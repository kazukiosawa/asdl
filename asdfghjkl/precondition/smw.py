from typing import Tuple, Any
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn

from ..utils import has_reduction
from ..core import extend
from ..operations import OP_GRAM_HADAMARD
from ..grad_maker import GradientMaker

__all__ = ['SmwEmpNaturalGradientConfig', 'SmwEmpNaturalGradientMaker']


def zero_kernel(model: nn.Module, n_data1: int, n_data2: int):
    p = next(iter(model.parameters()))
    kernel = torch.zeros(n_data1, n_data2, device=p.device, dtype=p.dtype)
    setattr(model, 'kernel', kernel)


def cholesky_solve(A, b, eps=1e-8):
    diag = torch.diagonal(A)
    diag += eps
    if A.ndim > b.ndim:
        b = b.unsqueeze(dim=-1)
    u = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b, u).squeeze(dim=-1)


@dataclass
class SmwEmpNaturalGradientConfig:
    data_size: int
    damping: float = 1.e-5


class SmwEmpNaturalGradientMaker(GradientMaker):
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config

    def forward_and_backward(self) -> Tuple[Any, Tensor]:
        assert has_reduction(self._loss_fn), 'loss_fn has to have "reduction" option'
        if isinstance(self._loss_fn, nn.Module):
            data_average = self._loss_fn.reduction == 'mean'
        else:
            data_average = self._loss_fn_kwargs.get('reduction', 'mean') == 'mean'
        model = self.model
        n = self.config.data_size
        damping = self.config.damping

        with extend(model, OP_GRAM_HADAMARD):
            zero_kernel(model, n, n)
            self.forward()
            batch_loss = self._loss
            params = [p for p in model.parameters() if p.requires_grad]
            torch.autograd.grad(batch_loss.sum(), params, retain_graph=True)
        UtU = model.kernel  # n x n
        Utg = UtU.sum(dim=1)  # n
        if data_average:
            UtU.div_(n)
        b = cholesky_solve(UtU, Utg, damping)
        ones = torch.ones_like(b)
        if data_average:
            b /= n ** 2
            ones /= n
        batch_loss.backward(gradient=(ones - b) / damping)
        self._loss = batch_loss.mean() if data_average else batch_loss.sum()
        return self._model_output, self._loss

    def _call_loss_fn(self) -> Tensor:
        assert has_reduction(self._loss_fn), 'loss_fn has to have "reduction" option'
        if isinstance(self._loss_fn, nn.Module):
            self._loss_fn.reduction = 'none'
        else:
            self._loss_fn_kwargs['reduction'] = 'none'
        args, kwargs = self._get_mapped_loss_fn_args_kwargs()
        return self._loss_fn(*args, **kwargs)
