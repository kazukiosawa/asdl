import warnings
from typing import Callable, List
from contextlib import nullcontext

import torch
from torch import nn
from torch.cuda import Stream
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
        ignore_modules=None,
        sync_group: dist.ProcessGroup = None,
        module_partitions_for_inv: List[List[nn.Module]] = None,
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
        if isinstance(fisher_shape, str):
            fisher_shape = [fisher_shape]
        for name, module, shapes in module_wise_assignments(model,
                                                            *fisher_shape,
                                                            ignore_modules=ignore_modules,
                                                            named=True):
            assert len(shapes) == 1, f'Each module has to be assigned one Fisher shape. ' \
                                     f'{name} is assigned {len(shapes)} shapes.'
        self.fisher_shape = fisher_shape
        self.ignore_modules = ignore_modules
        self._modules_for = {}
        fisher_cls = get_fisher_class(fisher_type, loss_type)
        self.fisher_manager = fisher_cls(model, **kwargs)
        self.sync_group = sync_group
        if module_partitions_for_inv is not None:
            assert dist.is_initialized(), 'torch.distributed has to be initialized ' \
                                          'when module_partitions_for_inv is specified.'
            world_size = dist.get_world_size(sync_group)
            assert all(len(module_partitions_for_inv[0]) == len(module_partitions_for_inv[i]) for i in range(1, world_size))
        self.module_partitions_for_inv = module_partitions_for_inv

    def modules_for(self, shape):
        if shape not in self._modules_for:
            self._modules_for[shape] = list(modules_to_assign(self.model,
                                                              shape,
                                                              *self.fisher_shape,
                                                              ignore_modules=self.ignore_modules))
        return self._modules_for[shape]

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
                         stream: Stream = None):
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
                    self.fisher_manager.accumulate(cxt, scale)
            else:
                stream_cxt = torch.cuda.stream(stream) if stream is not None else nullcontext()
                with stream_cxt:
                    for shape in _module_level_shapes:
                        for module in self.modules_for(shape):
                            cxt.calc_cov(module, shape, clear_in_out=True)
                    self.fisher_manager.accumulate(cxt, scale)
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
                             stream: Stream = None):
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
                                     stream=stream)

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
                          stream: Stream = None):
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
                                     stream=stream)

    def reduce_curvature(self, all_reduce=True):
        self.fisher_manager.reduce_matrices(all_reduce=all_reduce)

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                if not self.is_module_for_inv(module):
                    continue
                matrix = self._get_module_symmatrix(module, shape)
                if matrix is None:
                    continue
                matrix.update_inv(damping)
        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.update_inv(damping)

    def precondition(self, vectors: ParamVector = None, grad_scale=1.):
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                if not self.is_module_for_inv(module):
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
                            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None, grad_scale=1.):
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
            vec_bias.data.mul_(grad_scale)
        matrix.mvp(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=True, inplace=True)

    def is_module_for_inv(self, module: nn.Module):
        if self.module_partitions_for_inv is None:
            return True
        else:
            rank = dist.get_rank(self.sync_group)
            return module in self.module_partitions_for_inv[rank]

    def reduce_scatter_fisher(self, *keys, with_grad=False):
        module_partitions = self.module_partitions_for_inv
        assert module_partitions is not None, 'module_partitions_for_inv is not specified.'
        self.fisher_manager.reduce_scatter_fisher(module_partitions,
                                                  *keys,
                                                  with_grad=with_grad,
                                                  group=self.sync_group)

    def all_gather_grad(self):
        assert dist.is_initialized()
        group = self.sync_group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        module_partitions = self.module_partitions_for_inv
        assert module_partitions is not None, 'module_partitions_for_inv is not specified.'
        assert len(module_partitions) == world_size
        num_modules_per_partition = len(module_partitions[0])
        for i in range(num_modules_per_partition):
            tensor_list = []
            grads_list = []
            for j in range(world_size):
                grads = [p.grad for p in module_partitions[j][i].parameters() if p.requires_grad and p.grad is not None]
                grads_list.append(grads)
                tensor_list.append(parameters_to_vector(grads))
            dist.all_gather(tensor_list, tensor_list[rank], group=group)
            for j in range(world_size):
                vector_to_parameters(tensor_list[j], grads_list[j])


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
        super().__init__(model, fisher_type, SHAPE_LAYER_WISE, loss_type, damping, ema_decay, ignore_modules, **kwargs)


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
        super().__init__(model, fisher_type, fisher_shape, loss_type, damping, ema_decay, ignore_modules, **kwargs)


class UnitWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 ignore_modules=None,
                 **kwargs,):
        super().__init__(model, fisher_type, SHAPE_UNIT_WISE, loss_type, damping, ema_decay, ignore_modules, **kwargs)


class DiagNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 ignore_modules=None,
                 **kwargs):
        super().__init__(model, fisher_type, SHAPE_DIAG, loss_type, damping, ema_decay, ignore_modules, **kwargs)


class EmpiricalNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_shape=SHAPE_FULL,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay,
                 ignore_modules=None):
        super().__init__(model,
                         fisher_type=FISHER_EMP,
                         fisher_shape=fisher_shape,
                         damping=damping,
                         ema_decay=ema_decay,
                         ignore_modules=ignore_modules)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
