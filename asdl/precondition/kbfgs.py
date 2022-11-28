import math

import torch
import torch.nn as nn
from torch import Tensor

from ..core import extend
from ..operations import OP_MEAN_INPUTS, OP_SPATIAL_MEAN_OUTPUTS, OP_SPATIAL_MEAN_OUTGRADS,\
    OP_OUT_SPATIAL_SIZE, OP_COV_KRON, OP_BFGS_KRON_S_AS, OperationContext
from ..utils import cholesky_inv
from ..symmatrix import SymMatrix
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig


__all__ = ['KronBfgsGradientMaker']


class KronBfgsGradientMaker(PreconditionedGradientMaker):
    _supported_classes = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, config: PreconditioningConfig,
                 minibatch_hessian_action: bool = False, mu1: float = 0.2):
        super().__init__(model, config)
        self._last_model_args = ()
        self._last_model_kwargs = dict()
        self._curr_model_args = ()
        self._curr_model_kwargs = dict()
        self._A_inv_exists = False
        self._B_inv_exists = False
        self.minibatch_hessian_action = minibatch_hessian_action
        self.bfgs_attr = 'bfgs'
        self.mu1 = mu1
        self.mean_outputs_attr = 'mean_outputs'
        self.mean_outgrads_attr = 'mean_outgrads'

    def do_forward_and_backward(self, step=None):
        return not self.do_update_preconditioner(step)

    def _startup(self):
        step = self.state['step']
        if step > 0 and self.do_update_preconditioner(step - 1):
            self._post_preconditioner_update()

    def update_preconditioner(self):
        if self.minibatch_hessian_action and self._A_inv_exists:
            op_names = (OP_BFGS_KRON_S_AS, OP_SPATIAL_MEAN_OUTPUTS, OP_OUT_SPATIAL_SIZE)
        else:
            op_names = (OP_COV_KRON, OP_MEAN_INPUTS, OP_SPATIAL_MEAN_OUTPUTS, OP_OUT_SPATIAL_SIZE)
        op_names += (OP_SPATIAL_MEAN_OUTGRADS,)
        with extend(self.module_dict, *op_names) as cxt:
            rst = self.forward()
            self._update_A_inv(cxt)
            self._store_mean(cxt, is_forward=True)
            self._loss.backward()
            self._store_mean(cxt, is_forward=False)
        self._record_model_args_kwargs()
        return rst

    def _post_preconditioner_update(self):
        self._restore_last_model_args_kwargs()
        # another forward and backward using the previous model_args, kwargs
        op_names = (OP_SPATIAL_MEAN_OUTPUTS, OP_SPATIAL_MEAN_OUTGRADS, OP_OUT_SPATIAL_SIZE)
        with extend(self.module_dict, *op_names) as cxt:
            self.forward()
            self._loss.backward()
            self._update_B_inv(cxt)
        self._restore_curr_model_args_kwargs()
        self._B_inv_exists = True

    def precondition(self):
        if not self._B_inv_exists:
            return
        for module in self.module_dict.values():
            matrix: SymMatrix = getattr(module, self.bfgs_attr)
            vec_weight = module.weight.grad
            if vec_weight is None:
                raise ValueError('gradient has not been calculated.')
            if module.bias is not None and module.bias.requires_grad:
                vec_bias = module.bias.grad
                if vec_bias is None:
                    raise ValueError('gradient has not been calculated.')
            else:
                vec_bias = None
            matrix.kron.mvp(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=True, inplace=True)

    def _record_model_args_kwargs(self):
        self._last_model_args = self._model_args
        self._last_model_kwargs = self._model_kwargs.copy()

    def _restore_last_model_args_kwargs(self):
        self._curr_model_args = self._model_args
        self._curr_model_kwargs = self._model_kwargs.copy()
        self._model_args = self._last_model_args
        self._model_kwargs = self._last_model_kwargs.copy()

    def _restore_curr_model_args_kwargs(self):
        self._model_args = self._curr_model_args
        self._model_kwargs = self._curr_model_kwargs.copy()

    def _update_A_inv(self, cxt: OperationContext):
        config = self.config
        for module in self.module_dict.values():
            damping = self._get_damping(cxt, module, is_A=True)
            bfgs = getattr(module, self.bfgs_attr, None)
            if self.minibatch_hessian_action and self._A_inv_exists:
                s, As = cxt.bfgs_kron_s_As(module)
                y = As + damping * s
            else:
                new_bfgs = cxt.cov_symmatrix(module, pop=True).mul_(1/config.data_size)
                if bfgs is None:
                    setattr(module, self.bfgs_attr, new_bfgs)
                    bfgs = new_bfgs
                else:
                    # update the exponential moving average (EMA) of A
                    new_bfgs.mul_(config.ema_decay)
                    bfgs.mul_(1 - config.ema_decay)
                    bfgs += new_bfgs  # this must be __iadd__ to preserve inv
                A = bfgs.kron.A
                if bfgs.kron.A_inv is None:
                    bfgs.kron.A_inv = cholesky_inv(A, damping)
                mean_in_data = cxt.mean_in_data(module)
                s = torch.mv(bfgs.kron.A_inv, mean_in_data)
                y = torch.mv(A, s) + damping * s
            if bfgs is None:
                raise ValueError(f'Matrix for {module} is not calculated yet.')
            H = bfgs.kron.A_inv
            bfgs_inv_update_(H, s, y)
        self._A_inv_exists = True

    def _store_mean(self, cxt: OperationContext, is_forward=True):
        for module in self.module_dict.values():
            if is_forward:
                setattr(module, self.mean_outputs_attr, cxt.spatial_mean_out_data(module))
            else:
                setattr(module, self.mean_outgrads_attr, cxt.spatial_mean_out_grads(module))

    def _update_B_inv(self, cxt: OperationContext):
        for module in self.module_dict.values():
            damping = self._get_damping(cxt, module, is_A=False)
            bfgs = getattr(module, self.bfgs_attr)
            s = cxt.spatial_mean_out_data(module) - getattr(module, self.mean_outputs_attr)
            y = cxt.spatial_mean_out_grads(module) - getattr(module, self.mean_outgrads_attr)
            if bfgs.kron.B_inv is None:
                bfgs.kron.B_inv = torch.eye(s.shape[0], device=s.device)
            H = bfgs.kron.B_inv
            if isinstance(module, nn.Conv2d):
                s = s.mean(dim=0)
                y = y.mean(dim=0)
            powell_lm_damping_(H, s, y, mu1=self.mu1, mu2=damping)
            bfgs_inv_update_(H, s, y)

    def _get_damping(self, cxt: OperationContext, module: nn.Module, is_A=True):
        damping = self.config.damping
        sqrt_damping = math.sqrt(damping)
        if isinstance(module, nn.Conv2d):
            spatial_size = cxt.out_spatial_size(module)
            sqrt_spatial_size = math.sqrt(spatial_size)
            if is_A:
                # for A
                return sqrt_damping * sqrt_spatial_size
            else:
                # for B
                return sqrt_damping / sqrt_spatial_size
        else:
            return sqrt_damping


def powell_lm_damping_(H: Tensor, s: Tensor, y: Tensor, mu1: float, mu2: float):
    if mu1 <= 0 or 1 <= mu1:
        raise ValueError(f'mu1 has to be in (0, 1). Got {mu1}.')
    if mu2 <= 0:
        raise ValueError(f'mu2 has to be > 0. Got {mu2}.')
    Hy = torch.mv(H, y)
    ytHy = torch.dot(y, Hy)
    sty = torch.dot(s, y)
    if sty < mu1 * ytHy:
        theta = (1 - mu1) * ytHy / (ytHy - sty)
    else:
        theta = 1
    s.mul_(theta).sub_(Hy, alpha=1 - theta)  # Powell's damping on H
    y.add_(s, alpha=mu2)  # Levenberg-Marquardt damping on H^{-1}


def bfgs_inv_update_(H: Tensor, s: Tensor, y: Tensor):
    """
    The update of H=B^{-1} in BFGS by using the Sherman-Morrison formula explained in
    https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    """
    msg = f'H has to be a {Tensor} containing a symmetric matrix.'
    if H.ndim != 2 or torch.any(H.T != H):
        raise ValueError(msg)
    d1, d2 = H.shape
    if d1 != d2:
        raise ValueError(msg)
    msg = f' has to be a {Tensor} containing a vector of same dimension as H.'
    if s.ndim != 1 or s.shape[0] != d1:
        raise ValueError('s' + msg)
    if y.ndim != 1 or y.shape[0] != d1:
        raise ValueError('s' + msg)

    sty = torch.dot(s, y)  # s^ty
    Hy = torch.mv(H, y)  # Hy
    Hyst = torch.outer(Hy, s)  # Hys^t
    ytHy = torch.dot(y, Hy)  # y^tHy
    sst = torch.outer(s, s)  # ss^t
    H.add_(sst.mul_(sty + ytHy).div_(sty ** 2))
    H.sub_((Hyst + Hyst.T) / sty)
