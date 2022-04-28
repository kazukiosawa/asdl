import warnings
from typing import Callable, List
from contextlib import nullcontext

import torch
from torch import nn
from torch.cuda import Stream, nvtx
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..core import module_wise_assignments, modules_to_assign, no_centered_cov
from ..operations import OperationContext
from ..matrices import *
from ..symmatrix import SymMatrix
from ..vector import ParamVector
from ..fisher import LOSS_CROSS_ENTROPY, get_fisher_class

_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)
_invalid_ema_decay = -1
_module_level_shapes = [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG]

__all__ = [
    'NaturalGradient', 'FullNaturalGradient', 'LayerWiseNaturalGradient', 'KFAC',
    'UnitWiseNaturalGradient', 'DiagNaturalGradient', 'EmpiricalNaturalGradient'
]


class NaturalGradient:
    """
    Args:
        model: base model that contains multiple modules
        fisher_shape: shape of Fisher
    """
    def __init__(
        self,
        model,
        fisher_type=FISHER_EXACT,
        fisher_shape=SHAPE_FULL,
        loss_type=LOSS_CROSS_ENTROPY,
        damping=1e-5,
        ema_decay=_invalid_ema_decay,
        grad_scale=1.,
        ignore_modules=None,
        sync_group: dist.ProcessGroup = None,
        sync_group_ranks: List[int] = None,
        module_partitions: List[List[nn.Module]] = None,
        record_mode=False,
        nvtx_tag='',
        **kwargs
    ):
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(model, DDP), f'{DDP} is not supported.'
        del DDP
        self.model = model
        self.fisher_type = fisher_type
        self.loss_type = loss_type
        self.damping = damping
        self.ema_decay = ema_decay
        self.grad_scale = grad_scale
        if isinstance(fisher_shape, str):
            fisher_shape = [fisher_shape]
        self.named_modules_for_curvature = []
        self.modules_for_curvature = []
        self.shape_for = {}
        for name, module, shapes in module_wise_assignments(model,
                                                            *fisher_shape,
                                                            ignore_modules=ignore_modules,
                                                            named=True):
            assert len(shapes) == 1, f'Each module has to be assigned one Fisher shape. ' \
                                     f'{name} is assigned {len(shapes)} shapes.'
            self.modules_for_curvature.append(module)
            self.named_modules_for_curvature.append((name, module))
            self.shape_for[module] = shapes[0]
            self.shape_for[name] = shapes[0]
        self.ignore_modules = ignore_modules
        self._named_modules_for = {}
        if module_partitions is not None:
            assert dist.is_initialized(), 'torch.distributed has to be initialized ' \
                                          'when module_partitions is specified.'
            world_size = dist.get_world_size(sync_group)
            assert len(module_partitions) == world_size
            assert all(len(module_partitions[0]) == len(module_partitions[i]) for i in range(1, world_size))
            self.partitioned_modules = [m for partition in module_partitions for m in partition]
            self.num_modules_per_partition = len(module_partitions[0])
        else:
            self.partitioned_modules = []
            self.num_modules_per_partition = None
        self.module_partitions = module_partitions

        self.fisher_shape = fisher_shape
        fisher_cls = get_fisher_class(fisher_type, loss_type)
        self.fisher_manager = fisher_cls(model, **kwargs)

        if sync_group is not None:
            assert sync_group_ranks is not None
            assert sync_group.size() == len(sync_group_ranks)
        self.sync_group = sync_group
        self.sync_group_ranks = sync_group_ranks
        self.record_mode = record_mode
        self._nvtx_tag = nvtx_tag

        self.curvature_sync_handles = []
        self.grad_sync_handles = []
        self.grads = []
        self.packed_grads = []

    def named_modules_for(self, shape):
        if shape not in self._named_modules_for:
            self._named_modules_for[shape] = list(modules_to_assign(self.model,
                                                                    shape,
                                                                    *self.fisher_shape,
                                                                    ignore_modules=self.ignore_modules,
                                                                    named=True))
        return self._named_modules_for[shape]

    def modules_for(self, shape):
        return [m for _, m in self.named_modules_for(shape)]

    def parameters_for(self, shape):
        for module in self.modules_for(shape):
            for p in module.parameters():
                if p.requires_grad:
                    yield p

    @property
    def _fisher_attr(self):
        return self.fisher_type

    def _get_module_fisher(self, module, postfix=None):
        if postfix is None:
            attr = self._fisher_attr
        else:
            attr = f'{self._fisher_attr}_{postfix}'
        fisher = getattr(module, attr, None)
        return fisher

    def _set_module_fisher(self, module, fisher, postfix=None):
        if postfix is None:
            attr = self._fisher_attr
        else:
            attr = f'{self._fisher_attr}_{postfix}'
        setattr(module, attr, fisher)

    def _get_full_fisher(self):
        return self._get_module_fisher(self.model)

    def _get_module_symmatrix(self, module, shape, postfix=None) -> SymMatrix:
        fisher = self._get_module_fisher(module, postfix)
        if fisher is None:
            return None
        if shape in [SHAPE_FULL, SHAPE_LAYER_WISE]:
            return fisher
        elif shape == SHAPE_KRON:
            return fisher.kron
        elif shape == SHAPE_UNIT_WISE:
            return fisher.unit
        elif shape == SHAPE_DIAG:
            return fisher.diag
        else:
            raise ValueError(f'Invalid shape: {shape}.')

    def _scale_fisher(self, scale):
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                matrix = self._get_module_symmatrix(module, shape)
                if matrix is not None:
                    matrix.mul_(scale)
        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.mul_(scale)

    def nvtx_tag(self, keyword):
        if self.record_mode:
            return f':{keyword}' + self._nvtx_tag
        else:
            return '' + self._nvtx_tag

    @nvtx.range('update_curvature')
    def update_curvature(self,
                         inputs=None,
                         targets=None,
                         data_loader=None,
                         cxt: OperationContext = None,
                         closure: Callable = None,
                         accumulate=False,
                         ema_decay=None,
                         data_average=True,
                         calc_emp_loss_grad=False,
                         seed=None,
                         scale=1,
                         stream: Stream = None,
                         module_name=None,
                         num_batches=None,
                         kron=None,
                         no_save=False):
        if ema_decay is None:
            ema_decay = self.ema_decay
        if ema_decay != _invalid_ema_decay:
            accumulate = True
            scale *= ema_decay
            self._scale_fisher(1 - ema_decay)

        if cxt is not None or closure is not None:
            assert self.fisher_type == FISHER_EMP, f'fisher_type needs to be {FISHER_EMP} ' \
                                                   f'for computation based on {OperationContext} or a closure.'
            if not accumulate:
                self.fisher_manager.zero_fisher()
            if cxt is None:
                with no_centered_cov(self.model, self.fisher_shape, ignore_modules=self.ignore_modules, stream=stream) as cxt:
                    closure()
                    if not no_save:
                        self.save_curvature(cxt, scale)
            else:
                stream_cxt = torch.cuda.stream(stream) if stream is not None else nullcontext()
                with stream_cxt:
                    for shape in _module_level_shapes:
                        for name, module in self.named_modules_for(shape):
                            if module_name is not None and name != module_name:
                                continue
                            cxt.calc_cov(module,
                                         shape,
                                         clear_in_out=True,
                                         kron=kron,
                                         tag=self.nvtx_tag(name),
                                         num_batches=num_batches)
                            if not no_save:
                                self.save_curvature(cxt, scale, module=module)
        else:
            rst = self.fisher_manager.calculate_fisher(self.fisher_shape,
                                                       inputs=inputs,
                                                       targets=targets,
                                                       data_loader=data_loader,
                                                       accumulate=accumulate,
                                                       data_average=data_average,
                                                       calc_emp_loss_grad=calc_emp_loss_grad,
                                                       seed=seed,
                                                       scale=scale,
                                                       stream=stream)
            return rst[0], rst[1]  # loss and outputs

    def save_curvature(self, cxt, scale=1., module=None, module_name=None):
        self.fisher_manager.accumulate(cxt, scale, target_module=module, target_module_name=module_name)

    def accumulate_curvature(self,
                             inputs=None,
                             targets=None,
                             data_loader=None,
                             cxt: OperationContext = None,
                             closure: Callable = None,
                             ema_decay=None,
                             data_average=True,
                             calc_emp_loss_grad=False,
                             seed=None,
                             scale=1,
                             stream: Stream = None,
                             module_name=None,
                             num_batches=None,
                             kron=None,
                             no_save=False):
        return self.update_curvature(inputs=inputs,
                                     targets=targets,
                                     data_loader=data_loader,
                                     cxt=cxt,
                                     closure=closure,
                                     accumulate=True,
                                     ema_decay=ema_decay,
                                     data_average=data_average,
                                     calc_emp_loss_grad=calc_emp_loss_grad,
                                     seed=seed,
                                     scale=scale,
                                     stream=stream,
                                     module_name=module_name,
                                     num_batches=num_batches,
                                     kron=kron,
                                     no_save=no_save)

    def refresh_curvature(self,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          cxt: OperationContext = None,
                          closure: Callable = None,
                          data_average=True,
                          calc_emp_loss_grad=False,
                          seed=None,
                          scale=1,
                          stream: Stream = None,
                          module_name=None,
                          num_batches=None,
                          kron=None,
                          no_save=False):
        if self.ema_decay != _invalid_ema_decay:
            warnings.warn(f'ema_decay ({self.ema_decay}) will be ignored.')
        return self.update_curvature(inputs=inputs,
                                     targets=targets,
                                     data_loader=data_loader,
                                     cxt=cxt,
                                     closure=closure,
                                     accumulate=False,
                                     ema_decay=_invalid_ema_decay,
                                     data_average=data_average,
                                     calc_emp_loss_grad=calc_emp_loss_grad,
                                     seed=seed,
                                     scale=scale,
                                     stream=stream,
                                     module_name=module_name,
                                     num_batches=num_batches,
                                     kron=kron,
                                     no_save=no_save)

    @nvtx.range('update_inv')
    def update_inv(self, damping=None, module_name=None, kron=None, zero_curvature=False, partition_aware=False):
        if kron is None:
            kron = ['A', 'B']
        if damping is None:
            damping = self.damping

        for shape in _module_level_shapes:
            for name, module in self.named_modules_for(shape):
                if module_name is not None:
                    if name != module_name:
                        continue
                    if partition_aware and module in self.partitioned_modules:
                        partition_id = self.partitioned_modules.index(module) // self.num_modules_per_partition
                        module_id_in_partition = self.module_partitions[partition_id].index(module)
                        rank_in_group = dist.get_rank(self.sync_group)
                        modified_partition_id = (partition_id + rank_in_group) % len(self.module_partitions)
                        module = self.module_partitions[modified_partition_id][module_id_in_partition]

                matrix = self._get_module_symmatrix(module, shape)
                if matrix is None:
                    continue

                event = f'inv_{shape}'
                if shape == SHAPE_KRON:
                    for A_or_B in kron:
                        event += f'_{A_or_B}'
                nvtx.range_push(event + self.nvtx_tag(name))

                if self.is_module_for_inv_and_precondition(module):
                    if shape == SHAPE_KRON:
                        matrix.update_inv(damping,
                                          calc_A_inv='A' in kron,
                                          calc_B_inv='B' in kron)
                    else:
                        matrix.update_inv(damping)

                if zero_curvature:
                    with torch.no_grad():
                        if shape == SHAPE_KRON:
                            if 'A' in kron:
                                matrix.A.mul_(0)
                            if 'B' in kron:
                                matrix.B.mul_(0)
                        else:
                            matrix.mul_(0)

                nvtx.range_pop()

                if module_name is not None:
                    break

        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.update_inv(damping)
            if zero_curvature:
                with torch.no_grad():
                    fisher.mul_(0)

    @nvtx.range('precondition')
    def precondition(self, vectors: ParamVector = None, grad_scale=None):
        if grad_scale is None:
            grad_scale = self.grad_scale
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                if not self.is_module_for_inv_and_precondition(module):
                    continue
                self.precondition_module(module, shape, vectors, grad_scale=grad_scale)
        params = [p for p in self.parameters_for(SHAPE_FULL)]
        if len(params) > 0:
            fisher = self._get_full_fisher()
            assert fisher is not None, f'Fisher of shape {SHAPE_FULL} has not been calculated.'
            if vectors is None:
                vectors = ParamVector(params, [p.grad for p in params])
            assert vectors is not None, 'gradient has not been calculated.'
            if grad_scale != 1:
                vectors.mul_(grad_scale)
            fisher.mvp(vectors=vectors, use_inv=True, inplace=True)

    def precondition_module(self, module, shape=None, vectors: ParamVector = None,
                            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None, grad_scale=None):
        if grad_scale is None:
            grad_scale = self.grad_scale
        if shape is None:
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
        if vectors is not None:
            vec_weight = vectors.get_vector_by_param(module.weight, None)
            vec_bias = vectors.get_vector_by_param(module.bias, None)
        assert shape is not None, f'No shape is assigned to module: {module}.'
        matrix = self._get_module_symmatrix(module, shape)
        assert matrix is not None, f'Matrix of shape {shape} for module {module} has not been calculated.'
        if vec_weight is None and module.weight.requires_grad:
            vec_weight = module.weight.grad
        assert vec_weight is not None, 'gradient has not been calculated.'
        if _bias_requires_grad(module):
            if vec_bias is None:
                vec_bias = module.bias.grad
            assert vec_bias is not None, 'gradient has not been calculated.'
        if grad_scale != 1:
            vec_weight.data.mul_(grad_scale)
            if vec_bias is not None:
                vec_bias.data.mul_(grad_scale)
        matrix.mvp(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=True, inplace=True)

    def is_module_for_inv_and_precondition(self, module: nn.Module):
        if module not in self.modules_for_curvature:
            return False
        module_partitions = self.module_partitions
        if module_partitions is None:
            return True
        if module not in self.partitioned_modules:
            return True
        else:
            rank = dist.get_rank(self.sync_group)
            return module in module_partitions[rank]

    @nvtx.range('sync_curvature')
    def sync_curvature(self, module_name=None, kron=None, diag=None, with_grad=False, enabled=True, async_op=False):
        if not enabled:
            return
        handles = []
        if self.module_partitions is not None:
            if module_name is not None:
                handles += self.reduce_curvature(module_name, kron=kron, diag=diag, with_grad=with_grad)
            else:
                handles += self.reduce_scatter_curvature(kron=kron, diag=diag, with_grad=with_grad)
        handles += self.all_reduce_undivided_curvature(module_name=module_name, kron=kron, diag=diag, with_grad=with_grad)
        if async_op:
            self.curvature_sync_handles += handles
        else:
            for handle in handles:
                handle.wait()

    def sync_grad_pre_precondition(self, enabled=True, async_op=False):
        if not enabled:
            return
        if self.module_partitions is not None:
            self.reduce_scatter_grad(async_op=async_op)
        self.all_reduce_undivided_grad(async_op=async_op)

    def sync_grad_post_precondition(self, enabled=True, async_op=False):
        if not enabled:
            return
        if self.module_partitions is not None:
            self.all_gather_grad(async_op=async_op)
        self.all_reduce_no_curvature_grad(async_op=async_op)

    @nvtx.range('reduce_scatter_curvature')
    def reduce_scatter_curvature(self, kron=None, diag=None, with_grad=False):
        module_partitions = self.module_partitions
        assert module_partitions is not None, 'module_partitions is not specified.'
        handles = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape, kron=kron, diag=diag)
            for keys in keys_list:
                handles += self.fisher_manager.reduce_scatter_fisher(module_partitions,
                                                                     *keys,
                                                                     with_grad=with_grad,
                                                                     group=self.sync_group,
                                                                     async_op=True)
        return handles

    @nvtx.range('reduce_curvature')
    def reduce_curvature(self, module_name, kron=None, diag=None, with_grad=False):
        module_partitions = self.module_partitions
        assert module_partitions is not None, 'module_partitions is not specified.'
        try:
            module = next(m for name, m in self.named_modules_for_curvature if name == module_name)
            dst = next(i for i, partition in enumerate(module_partitions) if module in partition)
            if self.sync_group is not None:
                dst = self.sync_group_ranks[dst]
        except StopIteration:
            return []
        keys_list = self._keys_list_from_shape(self.shape_for[module], kron=kron, diag=diag)
        handles = []
        for keys in keys_list:
            handles += self.fisher_manager.reduce_fisher([module],
                                                         *keys,
                                                         all_reduce=False,
                                                         dst=dst,
                                                         with_grad=with_grad,
                                                         group=self.sync_group,
                                                         async_op=True)
        return handles

    @nvtx.range('all_reduce_undivided_curvature')
    def all_reduce_undivided_curvature(self, module_name=None, kron=None, diag=None, with_grad=False):
        modules = []
        for name, module in self.named_modules_for_curvature:
            if module in self.partitioned_modules:
                continue
            if module_name is not None and name != module_name:
                continue
            modules.append(module)
        handles = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape, kron=kron, diag=diag)
            for keys in keys_list:
                handles += self.fisher_manager.reduce_fisher(modules,
                                                             *keys,
                                                             all_reduce=True,
                                                             with_grad=with_grad,
                                                             group=self.sync_group,
                                                             async_op=True)
        return handles

    @staticmethod
    def _keys_list_from_shape(shape, kron=None, diag=None):
        if shape == SHAPE_FULL:
            return [['data']]
        elif shape == SHAPE_LAYER_WISE:
            return [['data']]
        elif shape == SHAPE_KRON:
            if kron is None:
                kron = ['A', 'B']
            assert all(A_or_B in ['A', 'B'] for A_or_B in kron)
            return [['kron', A_or_B] for A_or_B in kron]
        elif shape == SHAPE_UNIT_WISE:
            return [['unit', 'data']]
        elif shape == SHAPE_DIAG:
            if diag is None:
                diag = ['weight', 'bias']
            assert all(w_or_b in ['weight', 'bias'] for w_or_b in diag)
            return [['diag', w_or_b] for w_or_b in diag]

    @nvtx.range('reduce_scatter_grad')
    def reduce_scatter_grad(self, async_op=False):
        self._scatter_or_gather_grad('scatter', async_op=async_op)

    @nvtx.range('all_gather_grad')
    def all_gather_grad(self, async_op=False):
        self._scatter_or_gather_grad('gather', async_op=async_op)

    def _scatter_or_gather_grad(self, scatter_or_gather, async_op=False):
        assert dist.is_initialized()
        group = self.sync_group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        module_partitions = self.module_partitions
        assert module_partitions is not None, 'module_partitions is not specified.'
        assert len(module_partitions) == world_size
        num_modules_per_partition = len(module_partitions[0])
        assert all(len(module_partitions[i]) == num_modules_per_partition for i in range(1, world_size))
        for i in range(num_modules_per_partition):
            tensor_list = []
            grads_list = []
            for j in range(world_size):
                grads = [p.grad for p in module_partitions[j][i].parameters() if p.requires_grad and p.grad is not None]
                grads_list.append(grads)
                tensor_list.append(parameters_to_vector(grads))
            if scatter_or_gather == 'scatter':
                handle = dist.reduce_scatter(tensor_list[rank], tensor_list, group=group, async_op=async_op)
                if async_op:
                    self.grad_sync_handles.append(handle)
                    self.grads.append(grads_list[rank])
                    self.packed_grads.append(tensor_list[rank])
                else:
                    vector_to_parameters(tensor_list[rank], grads_list[rank])
            else:
                handle = dist.all_gather(tensor_list, tensor_list[rank], group=group, async_op=async_op)
                if async_op:
                    self.grad_sync_handles.append(handle)
                    self.grads.append([grads_list[j] for j in range(world_size)])
                    self.packed_grads.append([tensor_list[j] for j in range(world_size)])
                else:
                    for j in range(world_size):
                        vector_to_parameters(tensor_list[j], grads_list[j])

    @nvtx.range('all_reduce_undivided_grad')
    def all_reduce_undivided_grad(self, async_op=False):
        assert dist.is_initialized()
        module_list = nn.ModuleList([m for m in self.modules_for_curvature if m not in self.partitioned_modules])
        self._all_reduce_grad(module_list, async_op=async_op)

    @nvtx.range('all_reduce_no_curvature_grad')
    def all_reduce_no_curvature_grad(self, async_op=False):
        module_list = nn.ModuleList([m for m in self.model.modules()
                                     if len(list(m.children())) == 0 and m not in self.modules_for_curvature])
        self._all_reduce_grad(module_list, async_op=async_op)

    def _all_reduce_grad(self, module: nn.Module, async_op=False):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if len(grads) == 0:
            return
        packed_tensor = parameters_to_vector(grads)
        handle = dist.all_reduce(packed_tensor, group=self.sync_group, async_op=async_op)
        if async_op:
            self.grad_sync_handles.append(handle)
            self.grads.append(grads)
            self.packed_grads.append(packed_tensor)
        else:
            vector_to_parameters(packed_tensor, grads)

    def wait_all_curvature_sync(self):
        for _ in range(len(self.curvature_sync_handles)):
            self.curvature_sync_handles.pop(0).wait()

    def wait_all_grad_sync(self):
        for _ in range(len(self.grad_sync_handles)):
            self.grad_sync_handles.pop(0).wait()
            grads = self.grads.pop(0)
            packed_grads = self.packed_grads.pop(0)
            if isinstance(grads, list) and isinstance(grads[0], list):
                assert isinstance(packed_grads, list)
                for p, g in zip(packed_grads, grads):
                    vector_to_parameters(p, g)
            else:
                vector_to_parameters(packed_grads, grads)


class FullNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 **kwargs):
        super().__init__(model, fisher_type, SHAPE_FULL, loss_type, damping, ema_decay, **kwargs)


class LayerWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 ignore_modules=None,
                 **kwargs):
        super().__init__(model, fisher_type, SHAPE_LAYER_WISE, loss_type, damping, ema_decay, ignore_modules=ignore_modules, **kwargs)


class KFAC(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 ignore_modules=None,
                 **kwargs):
        fisher_shape = [SHAPE_KRON,
                        (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                        (nn.BatchNorm2d, SHAPE_UNIT_WISE)]
        super().__init__(model, fisher_type, fisher_shape, loss_type, damping, ema_decay, ignore_modules=ignore_modules, **kwargs)


class UnitWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 ignore_modules=None,
                 **kwargs,):
        super().__init__(model, fisher_type, SHAPE_UNIT_WISE, loss_type, damping, ema_decay, ignore_modules=ignore_modules, **kwargs)


class DiagNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 ignore_modules=None,
                 **kwargs):
        super().__init__(model, fisher_type, SHAPE_DIAG, loss_type, damping, ema_decay, ignore_modules=ignore_modules, **kwargs)


class EmpiricalNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_shape=SHAPE_FULL,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 grad_scale=1.,
                 ignore_modules=None,
                 sync_group: dist.ProcessGroup = None,
                 module_partitions: List[List[nn.Module]] = None,
                 record_mode=False,
                 nvtx_tag='',
                 **kwargs):
        super().__init__(model,
                         fisher_type=FISHER_EMP,
                         fisher_shape=fisher_shape,
                         damping=damping,
                         ema_decay=ema_decay,
                         grad_scale=grad_scale,
                         ignore_modules=ignore_modules,
                         sync_group=sync_group,
                         module_partitions=module_partitions,
                         record_mode=record_mode,
                         nvtx_tag=nvtx_tag,
                         **kwargs)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
