import warnings
from torch import nn

from .core import module_wise_assignments, modules_to_assign
from .matrices import *
from .symmatrix import SymMatrix
from .fisher import calculate_fisher, LOSS_CROSS_ENTROPY

_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)
_invalid_ema_decay = -1
_module_level_shapes = [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG]

__all__ = [
    'NaturalGradient', 'FullNaturalGradient', 'LayerWiseNaturalGradient', 'KFAC',
    'UnitWiseNaturalGradient', 'DiagNaturalGradient'
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
        n_mc_samples=1,
        damping=1e-5,
        ema_decay=_invalid_ema_decay,
    ):
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(model, DDP), f'{DDP} is not supported.'
        del DDP
        self.model = model
        self.fisher_type = fisher_type
        self.loss_type = loss_type
        self.n_mc_samples = n_mc_samples
        self.damping = damping
        self.ema_decay = ema_decay
        self.fisher_manager = None
        if isinstance(fisher_shape, str):
            fisher_shape = [fisher_shape]
        for name, module, shapes in module_wise_assignments(model, *fisher_shape, named=True):
            assert len(shapes) == 1, f'Each module has to be assigned one Fisher shape. ' \
                                     f'{name} is assigned {len(shapes)} shapes.'
        self.fisher_shape = fisher_shape

    def modules_for(self, shape):
        return modules_to_assign(self.model, shape, *self.fisher_shape)

    def parameters_for(self, shape):
        for module in self.modules_for(shape):
            for p in module.parameters():
                yield p

    def _get_module_fisher(self, module, postfix=None):
        if postfix is None:
            attr = self.fisher_type
        else:
            attr = f'{self.fisher_type}_{postfix}'
        fisher = getattr(module, attr, None)
        return fisher

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
                    matrix.scaling(scale)
        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.scaling(scale)

    def _update_curvature(self,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          accumulate=False,
                          ema_decay=None,
                          data_average=True,
                          calc_emp_loss_grad=False,
                          seed=None,
                          scale=1):
        if ema_decay is None:
            ema_decay = self.ema_decay
        if ema_decay != _invalid_ema_decay:
            assert accumulate, 'ema_decay cannot be set when accumulate=False.'
            scale *= ema_decay
            self._scale_fisher(1 - ema_decay)

        rst = calculate_fisher(self.model,
                               loss_type=self.loss_type,
                               fisher_type=self.fisher_type,
                               fisher_shapes=self.fisher_shape,
                               inputs=inputs,
                               targets=targets,
                               data_loader=data_loader,
                               accumulate=accumulate,
                               data_average=data_average,
                               calc_emp_loss_grad=calc_emp_loss_grad,
                               return_loss=True,
                               seed=seed,
                               scale=scale,
                               n_mc_samples=self.n_mc_samples)
        self.fisher_manager = rst[0]
        return rst[1]  # loss value

    def accumulate_curvature(self,
                             inputs=None,
                             targets=None,
                             data_loader=None,
                             ema_decay=None,
                             data_average=True,
                             calc_emp_loss_grad=False,
                             seed=None,
                             scale=1):
        return self._update_curvature(inputs,
                                      targets,
                                      data_loader,
                                      accumulate=True,
                                      ema_decay=ema_decay,
                                      data_average=data_average,
                                      calc_emp_loss_grad=calc_emp_loss_grad,
                                      seed=seed,
                                      scale=scale)

    def refresh_curvature(self,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          data_average=True,
                          calc_emp_loss_grad=False,
                          seed=None,
                          scale=1):
        if self.ema_decay != _invalid_ema_decay:
            warnings.warn(f'ema_decay ({self.ema_decay}) will be ignored.')
        return self._update_curvature(inputs,
                                      targets,
                                      data_loader,
                                      accumulate=False,
                                      ema_decay=_invalid_ema_decay,
                                      data_average=data_average,
                                      calc_emp_loss_grad=calc_emp_loss_grad,
                                      seed=seed,
                                      scale=scale)

    def reduce_curvature(self, all_reduce=True):
        self.fisher_manager.reduce_matrices(all_reduce=all_reduce)

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                matrix = self._get_module_symmatrix(module, shape)
                if matrix is None:
                    continue
                matrix.update_inv(damping)
        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.update_inv(damping)

    def precondition(self, vecs=None):
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                self.precondition_module(module, shape)
        fisher = self._get_full_fisher()
        if fisher is not None:
            if vecs is None:
                vecs = [p.grad for p in self.parameters_for(SHAPE_FULL) if p.requires_grad]
            fisher.mvp(vecs=vecs, use_inv=True, inplace=True)

    def precondition_module(self, module, shape=None, vec_weight=None, vec_bias=None):
        if shape is None:
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
        assert shape is not None, f'No shape is assigned to module: {module}.'
        matrix = self._get_module_symmatrix(module, shape)
        if vec_weight is None:
            vec_weight = module.weight.grad
            if vec_bias is None and _bias_requires_grad(module):
                vec_bias = module.bias.grad
        matrix.mvp(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=True, inplace=True)


class FullNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_FULL, loss_type, n_mc_samples, damping, ema_decay)


class LayerWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_LAYER_WISE, loss_type, n_mc_samples, damping, ema_decay)


class KFAC(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_KRON, loss_type, n_mc_samples, damping, ema_decay)


class UnitWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_UNIT_WISE, loss_type, n_mc_samples, damping, ema_decay)


class DiagNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_DIAG, loss_type, n_mc_samples, damping, ema_decay)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
