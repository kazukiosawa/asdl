from typing import Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..core import extend
from ..operations import OP_GRAM_HADAMARD
from .grad_maker import GradientMaker

torch_function_class = F.cross_entropy.__class__


def has_reduction(func):
    if isinstance(func, nn.Module):
        return hasattr(func, 'reduction')
    elif isinstance(func, torch_function_class):
        return 'reduction' in func.__code__.co_varnames
    return False


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
class SmwEmpiricalNaturalGradientMakerConfig:
    data_size: int
    damping: float = 1.e-5
    data_average: bool = True


class SmwEmpiricalNaturalGradientMaker(GradientMaker):
    def __init__(self, model, config=None, data_size: int = None):
        super().__init__(model)
        if config is None:
            assert data_size is not None, f'data_size has to be specified ' \
                                          f'when config is not given.'
            config = SmwEmpiricalNaturalGradientMakerConfig(data_size=data_size)
        else:
            assert isinstance(config, SmwEmpiricalNaturalGradientMakerConfig)
        self.config = config

    def forward_and_backward(self) -> Tuple[Tensor, Tensor]:
        model = self.model
        n = self.config.data_size
        data_average = self.config.data_average
        damping = self.config.damping

        with extend(model, OP_GRAM_HADAMARD):
            zero_kernel(model, n, n)
            logits, batch_loss = self._forward()
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
        if data_average:
            return logits, batch_loss.mean()
        else:
            return logits, batch_loss.sum()

    def _call_loss_fn(self) -> Tensor:
        assert has_reduction(self._loss_fn), 'loss_fn has to have "reduction" option'
        if isinstance(self._loss_fn, nn.Module):
            self._loss_fn.reduction = 'none'
        else:
            self._loss_fn_kwargs['reduction'] = 'none'
        args, kwargs = self._get_mapped_loss_fn_args_kwargs()
        return self._loss_fn(*args, **kwargs)
