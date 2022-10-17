from typing import Callable, List, Tuple, Union, Any
from dataclasses import dataclass
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..utils import nvtx_range
from ..core import module_wise_assignments, modules_to_assign
from ..matrices import *
from ..symmatrix import SymMatrix
from ..vector import ParamVector
from ..fisher import LOSS_CROSS_ENTROPY, get_fisher_maker, FisherConfig
from ..grad_maker import GradientMaker
from ..interval import IntervalScheduler

_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)
_invalid_ema_decay = -1
_module_level_shapes = [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_SWIFT_KRON, SHAPE_KFE, SHAPE_UNIT_WISE, SHAPE_DIAG]

__all__ = [
    'NaturalGradientConfig', 'NaturalGradientMaker', 'FullNaturalGradientMaker', 'LayerWiseNaturalGradientMaker',
    'KfacGradientMaker', 'EkfacGradientMaker', 'UnitWiseNaturalGradientMaker', 'DiagNaturalGradientMaker', 'EmpNaturalGradientMaker'
]


@dataclass
class NaturalGradientConfig:
    data_size: int
    fisher_type: str = FISHER_MC
    fisher_shape: Union[str, List[Any]] = SHAPE_FULL
    loss_type: str = LOSS_CROSS_ENTROPY
    damping: float = 1e-5
    ema_decay: float = _invalid_ema_decay
    scale: float = 1.
    grad_scale: float = 1.
    upd_curvature_interval: int = 1
    upd_inv_interval: int = 1
    interval_scheduler:IntervalScheduler =None
    ignore_modules: List[any] = None
    sync_group: dist.ProcessGroup = None
    sync_group_ranks: List[int] = None
    module_partitions: List[List[nn.Module]] = None
    record_mode: bool = False
    nvtx_tag: str = ''
    n_mc_samples: int = 1
    var: float = 1
    seed: int = None


class NaturalGradientMaker(GradientMaker):
    def __init__(self, model, config):
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(model, DDP), f'{DDP} is not supported.'
        del DDP
        super().__init__(model)
        if isinstance(config.fisher_shape, str):
            config.fisher_shape = [config.fisher_shape]
        self.config = config

        self.named_modules_for_curvature = []
        self.modules_for_curvature = []
        self.shape_for = {}
        for name, module, shapes in module_wise_assignments(model,
                                                            *config.fisher_shape,
                                                            ignore_modules=config.ignore_modules,
                                                            named=True):
            assert len(shapes) == 1, f'Each module has to be assigned one Fisher shape. ' \
                                     f'{name} is assigned {len(shapes)} shapes.'
            self.modules_for_curvature.append(module)
            self.named_modules_for_curvature.append((name, module))
            self.shape_for[module] = shapes[0]
            self.shape_for[name] = shapes[0]
        self._named_modules_for = {}

        module_partitions = config.module_partitions
        sync_group = config.sync_group
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

        fisher_config = FisherConfig(
            fisher_type=config.fisher_type,
            fisher_shapes=config.fisher_shape,
            loss_type=config.loss_type,
            n_mc_samples=config.n_mc_samples,
            var=config.var,
            seed=config.seed,
        )
        self.fisher_maker = get_fisher_maker(model, fisher_config)

        if sync_group is not None:
            assert config.sync_group_ranks is not None
            assert sync_group.size() == len(config.sync_group_ranks)

        self.curvature_sync_handles = []
        self.grad_sync_handles = []
        self.grads = []
        self.packed_grads = []

        self._step = 0

    def forward_and_backward(self, accumulate=False) -> Union[Tuple[Any, Tensor], Any]:
        config = self.config
        if not accumulate:
            assert config.upd_curvature_interval == config.upd_inv_interval, \
                'upd_curvature_interval and upd_inv_interval needs to be the same when no curvature accumulation is performed.'
        
        if config.interval_scheduler is not None:
            update_TF = config.interval_scheduler.updateTF()
        else:
            update_TF = (self._step % self.config.upd_curvature_interval == 0)
        
        if update_TF:
            self.update_curvature(accumulate=accumulate,
                              calc_inv=not accumulate)
        else:
            self.forward()
            self._loss.backward()

        if accumulate and self._step % self.config.upd_inv_interval == 0:
            self.update_inv()
        self.precondition()
        self._step += 1
        if self._loss_fn is None:
            return self._model_output
        else:
            return self._model_output, self._loss

    def named_modules_for(self, shape):
        if shape not in self._named_modules_for:
            self._named_modules_for[shape] = list(modules_to_assign(self.model,
                                                                    shape,
                                                                    *self.config.fisher_shape,
                                                                    ignore_modules=self.config.ignore_modules,
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
        return self.fisher_maker.config.fisher_attr

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
        elif shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
            return fisher.kron
        elif shape == SHAPE_KFE:
            return fisher.kfe
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
        if self.config.record_mode:
            return f':{keyword}' + self.config.nvtx_tag
        else:
            return '' + self.config.nvtx_tag

    @nvtx_range('update_curvature')
    def update_curvature(self,
                         accumulate=False,
                         calc_inv=False):
        config = self.config
        fisher_maker = self.fisher_maker
        scale = config.scale

        ema_decay = config.ema_decay
        if ema_decay != _invalid_ema_decay:
            accumulate = True
            scale *= ema_decay
            self._scale_fisher(1 - ema_decay)

        self.delegate_forward_and_backward(fisher_maker,
                                           data_size=self.config.data_size,
                                           scale=scale,
                                           accumulate=accumulate,
                                           calc_loss_grad=True,
                                           calc_inv=calc_inv,
                                           damping=self.config.damping
                                           )

    @nvtx_range('update_inv')
    def update_inv(self, damping=None, module_name=None, kron=None, zero_curvature=False, partition_aware=False):
        if kron is None:
            kron = ['A', 'B']
        if damping is None:
            damping = self.config.damping

        for shape in _module_level_shapes:
            for name, module in self.named_modules_for(shape):
                if module_name is not None:
                    if name != module_name:
                        continue
                    if partition_aware and module in self.partitioned_modules:
                        partition_id = self.partitioned_modules.index(module) // self.num_modules_per_partition
                        module_id_in_partition = self.config.module_partitions[partition_id].index(module)
                        rank_in_group = dist.get_rank(self.config.sync_group)
                        modified_partition_id = (partition_id + rank_in_group) % len(self.config.module_partitions)
                        module = self.config.module_partitions[modified_partition_id][module_id_in_partition]

                matrix = self._get_module_symmatrix(module, shape)
                if matrix is None:
                    continue

                event = f'inv_{shape}'
                if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                    for A_or_B in kron:
                        event += f'_{A_or_B}'

                with nvtx_range(event + self.nvtx_tag(name)):
                    if self.is_module_for_inv_and_precondition(module):
                        if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                            matrix.update_inv(damping, calc_A_inv='A' in kron, calc_B_inv='B' in kron)
                        else:
                            matrix.update_inv(damping)

                    if zero_curvature:
                        with torch.no_grad():
                            if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                                if 'A' in kron:
                                    matrix.A.mul_(0)
                                if 'B' in kron:
                                    matrix.B.mul_(0)
                            else:
                                matrix.mul_(0)

                if module_name is not None:
                    break

        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.update_inv(damping)
            if zero_curvature:
                with torch.no_grad():
                    fisher.mul_(0)

    @nvtx_range('precondition')
    def precondition(self, vectors: ParamVector = None, grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.config.grad_scale
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                if not self.is_module_for_inv_and_precondition(module):
                    continue
                self.precondition_module(module, shape, vectors, grad_scale=grad_scale, use_inv=use_inv)
        params = [p for p in self.parameters_for(SHAPE_FULL)]
        if len(params) > 0:
            fisher = self._get_full_fisher()
            assert fisher is not None, f'Fisher of shape {SHAPE_FULL} has not been calculated.'
            if vectors is None:
                vectors = ParamVector(params, [p.grad for p in params])
            assert vectors is not None, 'gradient has not been calculated.'
            if grad_scale != 1:
                vectors.mul_(grad_scale)
            fisher.mvp(vectors=vectors, use_inv=use_inv, inplace=True)

    def precondition_module(self, module, shape=None, vectors: ParamVector = None,
                            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None,
                            grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.config.grad_scale
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
        kwargs = dict(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=use_inv, inplace=True)
        if shape == SHAPE_KFE:
            kwargs['eps'] = self.config.damping
        matrix.mvp(**kwargs)

    def is_module_for_inv_and_precondition(self, module: nn.Module):
        if module not in self.modules_for_curvature:
            return False
        module_partitions = self.config.module_partitions
        if module_partitions is None:
            return True
        if module not in self.partitioned_modules:
            return True
        else:
            rank = dist.get_rank(self.config.sync_group)
            return module in module_partitions[rank]

    @nvtx_range('sync_curvature')
    def sync_curvature(self, module_name=None, kron=None, diag=None, with_grad=False, enabled=True, async_op=False):
        if not enabled:
            return
        handles = []
        if self.config.module_partitions is not None:
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
        if self.config.module_partitions is not None:
            self.reduce_scatter_grad(async_op=async_op)
        self.all_reduce_undivided_grad(async_op=async_op)

    def sync_grad_post_precondition(self, enabled=True, async_op=False):
        if not enabled:
            return
        if self.config.module_partitions is not None:
            self.all_gather_grad(async_op=async_op)
        self.all_reduce_no_curvature_grad(async_op=async_op)

    @nvtx_range('reduce_scatter_curvature')
    def reduce_scatter_curvature(self, kron=None, diag=None, with_grad=False):
        module_partitions = self.config.module_partitions
        assert module_partitions is not None, 'module_partitions is not specified.'
        handles = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape, kron=kron, diag=diag)
            for keys in keys_list:
                handles += self.fisher_maker.reduce_scatter_fisher(module_partitions,
                                                                   *keys,
                                                                   with_grad=with_grad,
                                                                   group=self.config.sync_group,
                                                                   async_op=True)
        return handles

    @nvtx_range('reduce_curvature')
    def reduce_curvature(self, module_name, kron=None, diag=None, with_grad=False):
        module_partitions = self.config.module_partitions
        assert module_partitions is not None, 'module_partitions is not specified.'
        try:
            module = next(m for name, m in self.named_modules_for_curvature if name == module_name)
            if module not in self.partitioned_modules:
                return []
            dst = next(i for i, partition in enumerate(module_partitions) if module in partition)
            if self.config.sync_group is not None:
                dst = self.config.sync_group_ranks[dst]
        except StopIteration:
            return []
        keys_list = self._keys_list_from_shape(self.shape_for[module], kron=kron, diag=diag)
        handles = []
        for keys in keys_list:
            handles += self.fisher_maker.reduce_fisher([module],
                                                       *keys,
                                                       all_reduce=False,
                                                       dst=dst,
                                                       with_grad=with_grad,
                                                       group=self.config.sync_group,
                                                       async_op=True)
        return handles

    @nvtx_range('all_reduce_undivided_curvature')
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
                handles += self.fisher_maker.reduce_fisher(modules,
                                                           *keys,
                                                           all_reduce=True,
                                                           with_grad=with_grad,
                                                           group=self.config.sync_group,
                                                           async_op=True)
        return handles

    @staticmethod
    def _keys_list_from_shape(shape, kron=None, diag=None):
        if shape == SHAPE_FULL:
            return [['data']]
        elif shape == SHAPE_LAYER_WISE:
            return [['data']]
        elif shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
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

    @nvtx_range('reduce_scatter_grad')
    def reduce_scatter_grad(self, async_op=False):
        self._scatter_or_gather_grad('scatter', async_op=async_op)

    @nvtx_range('all_gather_grad')
    def all_gather_grad(self, async_op=False):
        self._scatter_or_gather_grad('gather', async_op=async_op)

    def _scatter_or_gather_grad(self, scatter_or_gather, async_op=False):
        assert dist.is_initialized()
        group = self.config.sync_group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        module_partitions = self.config.module_partitions
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

    @nvtx_range('all_reduce_undivided_grad')
    def all_reduce_undivided_grad(self, async_op=False):
        assert dist.is_initialized()
        module_list = nn.ModuleList([m for m in self.modules_for_curvature if m not in self.partitioned_modules])
        self._all_reduce_grad(module_list, async_op=async_op)

    @nvtx_range('all_reduce_no_curvature_grad')
    def all_reduce_no_curvature_grad(self, async_op=False):
        module_list = nn.ModuleList([m for m in self.model.modules()
                                     if len(list(m.children())) == 0 and m not in self.modules_for_curvature])
        self._all_reduce_grad(module_list, async_op=async_op)

    def _all_reduce_grad(self, module: nn.Module, async_op=False):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if len(grads) == 0:
            return
        packed_tensor = parameters_to_vector(grads)
        handle = dist.all_reduce(packed_tensor, group=self.config.sync_group, async_op=async_op)
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


class FullNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: NaturalGradientConfig):
        config.fisher_shape = SHAPE_FULL
        super().__init__(model, config)


class LayerWiseNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: NaturalGradientConfig):
        config.fisher_shape = SHAPE_LAYER_WISE
        super().__init__(model, config)


class KfacGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: NaturalGradientConfig, swift=False):
        config.fisher_shape = [SHAPE_SWIFT_KRON if swift else SHAPE_KRON,
                               (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                               (nn.BatchNorm2d, SHAPE_UNIT_WISE),
                               (nn.LayerNorm, SHAPE_UNIT_WISE)]
        super().__init__(model, config)


class EkfacGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: NaturalGradientConfig):
        assert config.fisher_type == FISHER_EMP, f'{EkfacGradientMaker} supports only {FISHER_EMP}.'
        if config.upd_inv_interval > 1:
            warnings.warn(f'{EkfacGradientMaker} ignores upd_inv_interval ({config.upd_inv_interval} is specified.)')
        config.fisher_shape = [SHAPE_KFE]
        super().__init__(model, config)

    def update_inv(self, *args, **kwargs):
        pass

    def precondition(self, vectors: ParamVector = None, grad_scale=None, use_inv=False):
        assert not use_inv, 'EKFAC does not calculate the inverse matrix.'
        super().precondition(vectors=vectors, grad_scale=grad_scale, use_inv=False)


class UnitWiseNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: NaturalGradientConfig):
        config.fisher_shape = SHAPE_UNIT_WISE
        super().__init__(model, config)


class DiagNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: NaturalGradientConfig):
        config.fisher_shape = SHAPE_DIAG
        super().__init__(model, config)


class EmpNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: NaturalGradientConfig):
        config.fisher_type = FISHER_EMP
        super().__init__(model, config)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
