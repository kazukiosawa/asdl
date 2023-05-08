from dataclasses import dataclass

import torch
import torch.nn as nn
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig

__all__ = ['CurveBallGradientMaker', 'CurveBallGradientConfig']


@dataclass
class CurveBallGradientConfig(PreconditioningConfig):
    damping: float = 1.e-7
    momentum = 0.9
    momentum_decay = 1.


class CurveBallGradientMaker(PreconditionedGradientMaker):
    def __init__(self, model: nn.Module, config: PreconditioningConfig,
                 momentum: float = 0.9, momentum_decay: float = 1):
        super().__init__(model, config)
        self._params = [p for p in model.parameters() if p.requires_grad]
        self._momentum = tuple([torch.zeros_like(p) for p in self._params])
        self.momentum = momentum
        self.momentum_decay = momentum_decay

    def do_forward_and_backward(self, step=None) -> bool:
        return False

    def _precondition(self):
        config = self.config
        hvps = self.loss_hvp(tangents=self._momentum)
        grads = [p.grad for p in self._params]
        if not (len(hvps) == len(grads) == len(self._momentum)):
            raise ValueError(f'len(hvps), len(grads), and len(momentum) have to be the same. '
                             f'Got {len(hvps)}, {len(grads)}, and {len(self._momentum)}')
        for p, m, hvp, grad in zip(self._params, self._momentum, hvps, grads):
            hvp.add_(m, alpha=config.damping)
            grad_of_quadratic = hvp + grad
            if self.momentum_decay != 1:
                m.mul_(self.momentum_decay)
            m.add_(grad_of_quadratic, alpha=self.momentum)
            if p.grad is None:
                p.grad = m.clone()
            else:
                p.grad.add_(m)
