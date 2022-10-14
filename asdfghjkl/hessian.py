from typing import Tuple, Union, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .symmatrix import SymMatrix, Diag
from .matrices import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_DIAG
from .mvp import power_method, conjugate_gradient_method, quadratic_form
from .vector import ParamVector
from .grad_maker import GradientMaker

__all__ = ['HessianConfig', 'HessianMaker']
_supported_shapes = [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_DIAG]


@dataclass
class HessianConfig:
    hessian_attr: str = 'hessian'
    tmp_hessian_attr: str = 'tmp_hessian'
    hvp_attr: str = 'hvp'
    tmp_hvp_attr: str = 'tmp_hvp'
    hessian_shapes: List[str] = None


class HessianMaker(GradientMaker):
    def __init__(self, model: nn.Module, config):
        super().__init__(model)
        self.config = config

    def zero_hessian(self, hvp=False):
        attr = self.config.hvp_attr if hvp else self.config.hessian_attr
        for module in self.model.modules():
            if hasattr(module, attr):
                delattr(module, attr)

    def forward_and_backward(self,
                             scale=1.,
                             accumulate=False,
                             calc_loss_grad=False,
                             hvp=False,
                             vec: ParamVector = None
                             ) -> Union[Tuple[Any, Tensor], Any]:
        if not accumulate:
            self.zero_hessian(hvp)

        self.forward()
        if hvp:
            params = [p for p in self.model.parameters() if p.requires_grad]
            v, g = _hvp(self._loss, params, vec)
            setattr(self.model, self.config.tmp_hvp_attr, v)
            if calc_loss_grad:
                for param, grad in zip(params, g.values()):
                    param.grad = grad
        else:
            self._hessian(calc_loss_grad)
        self.accumulate(scale)
        if self._loss_fn is None:
            return self._model_output
        else:
            return self._model_output, self._loss

    def _hessian(self, calc_loss_grad=False):
        model = self.model
        loss = self._loss
        hessian_shapes = self.config.hessian_shapes
        if isinstance(hessian_shapes, str):
            hessian_shapes = [hessian_shapes]
        # remove duplicates
        hessian_shapes = set(hessian_shapes)
        for hshape in hessian_shapes:
            assert hshape in _supported_shapes, f'Invalid hessian_shape: {hshape}. ' \
                                                f'hessian_shape must be in {_supported_shapes}.'
        save_attr = self.config.tmp_hessian_attr
        params = [p for p in model.parameters() if p.requires_grad]

        # full
        if SHAPE_FULL in hessian_shapes:
            full_hess = _hessian(loss, params, save_grad=calc_loss_grad)
            setattr(model, save_attr, SymMatrix(data=full_hess))
        else:
            full_hess = None

        if SHAPE_LAYER_WISE not in hessian_shapes \
                and SHAPE_DIAG not in hessian_shapes:
            return

        idx = 0
        for module in model.modules():
            w = getattr(module, 'weight', None)
            b = getattr(module, 'bias', None)
            params = [p for p in [w, b] if p is not None and p.requires_grad]
            if len(params) == 0:
                continue

            # module hessian
            if full_hess is None:
                m_hess = _hessian(loss, params, save_grad=calc_loss_grad)
            else:
                m_numel = sum([p.numel() for p in params])
                m_hess = full_hess[idx:idx + m_numel, idx:idx + m_numel]
                idx += m_numel

            # block-diagonal
            if SHAPE_LAYER_WISE in hessian_shapes:
                setattr(module, save_attr, SymMatrix(data=m_hess))

            # diagonal
            if SHAPE_DIAG in hessian_shapes:
                m_hess = torch.diag(m_hess)
                _idx = 0
                w_hess = b_hess = None
                if w is not None and w.requires_grad:
                    w_numel = w.numel()
                    w_hess = m_hess[_idx:_idx + w_numel].view_as(w)
                    _idx += w_numel
                if b is not None and b.requires_grad:
                    b_numel = b.numel()
                    b_hess = m_hess[_idx:_idx + b_numel].view_as(b)
                    _idx += b_numel
                diag = Diag(weight=w_hess, bias=b_hess)
                if hasattr(module, save_attr):
                    getattr(module, save_attr).diag = diag
                else:
                    setattr(module, save_attr, SymMatrix(diag=diag))

    def _extract_tmp_hessian(self, module: nn.Module, hvp=False):
        attr = self.config.tmp_hvp_attr if hvp else self.config.tmp_hessian_attr
        tmp_hessian = getattr(module, attr, None)
        if tmp_hessian is not None:
            delattr(module, attr)
        return tmp_hessian

    def _extract_tmp_hvp(self, module: nn.Module):
        return self._extract_tmp_hessian(module, hvp=True)

    def accumulate(self, scale=1.):
        model = self.model
        for module in model.modules():
            self._accumulate_hessian(module, self._extract_tmp_hessian(module), scale)
            self._accumulate_hvp(module, self._extract_tmp_hvp(module), scale)
        self._accumulate_hessian(model, self._extract_tmp_hessian(model), scale)
        self._accumulate_hvp(model, self._extract_tmp_hvp(model), scale)

    def _accumulate_hessian(self, module: nn.Module, new_hessian, scale=1., hvp=False):
        if new_hessian is None:
            return
        if scale != 1:
            new_hessian.mul_(scale)
        dst_attr = self.config.hvp_attr if hvp else self.config.hessian_attr
        dst_hessian = getattr(module, dst_attr, None)
        if dst_hessian is None:
            setattr(module, dst_attr, new_hessian)
        else:
            # this must be __iadd__ to preserve inv
            dst_hessian += new_hessian

    def _accumulate_hvp(self, module: nn.Module, new_hessian, scale=1.):
        self._accumulate_hessian(module, new_hessian, scale, hvp=True)

    def _get_hvp_fn(self):
        def hvp_fn(vec: ParamVector) -> ParamVector:
            self.forward_and_backward(hvp=True, vec=vec)
            return getattr(self.model, self.config.hvp_attr)
        return hvp_fn

    def hessian_eig(self,
                    top_n=1,
                    max_iters=100,
                    tol=1e-7,
                    is_distributed=False,
                    print_progress=False
                    ):
        eigvals, eigvecs = power_method(self._get_hvp_fn(),
                                        self.model,
                                        top_n=top_n,
                                        max_iters=max_iters,
                                        tol=tol,
                                        is_distributed=is_distributed,
                                        print_progress=print_progress)

        return eigvals, eigvecs

    def hessian_free(self,
                     b=None,
                     init_x=None,
                     damping=1e-3,
                     max_iters=None,
                     tol=1e-8,
                     print_progress=False,
                     ):
        if b is None:
            grads = {p: p.grad for p in self.model.parameters() if p.requires_grad}
            b = ParamVector(grads.keys(), grads.values())
        return conjugate_gradient_method(self._get_hvp_fn(),
                                         b,
                                         init_x=init_x,
                                         damping=damping,
                                         max_iters=max_iters,
                                         tol=tol,
                                         print_progress=print_progress)

    def hessian_quadratic_form(self, vec: ParamVector = None):
        if vec is None:
            grads = {p: p.grad for p in self.model.parameters() if p.requires_grad}
            vec = ParamVector(grads.keys(), grads.values())

        return quadratic_form(self._get_hvp_fn(), vec)


def _hvp(output, inputs, vec: ParamVector):
    grads = torch.autograd.grad(output, inputs=inputs, create_graph=True)
    v = torch.autograd.grad(grads, inputs=inputs, grad_outputs=tuple(vec.values()))
    return ParamVector(inputs, v), ParamVector(inputs, grads)


# adopted from https://github.com/mariogeiger/hessian/blob/master/hessian/hessian.py
def _hessian(output, inputs, out=None, allow_unused=False, create_graph=False, save_grad=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    hessian((x * y).sum(), [x, y])
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(
            output, inp, create_graph=True, allow_unused=allow_unused
        )
        if save_grad and inp.requires_grad:
            inp.grad = grad
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = _gradient(
                    grad[j],
                    inputs[i:],
                    retain_graph=True,
                    create_graph=create_graph
                )[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out


# adopted from https://github.com/mariogeiger/hessian/blob/master/hessian/gradient.py
def _gradient(
    outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False
):
    '''
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph
    )
    grads = [
        x if x is not None else torch.zeros_like(y) for x,
        y in zip(grads, inputs)
    ]
    return torch.cat([x.contiguous().view(-1) for x in grads])
