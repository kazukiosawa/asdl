from dataclasses import dataclass

import torch
import torch.nn as nn
from .prec_grad_maker import PreconditionedGradientMaker, PreconditionedGradientConfig

__all__ = ['CurveBallGradientMaker', 'CurveBallGradientConfig']


@dataclass
class CurveBallGradientConfig(PreconditionedGradientConfig):
    damping: float = 1.e-7
    momentum = 0.9
    momentum_decay = 1.


class CurveBallGradientMaker(PreconditionedGradientMaker):
    def __init__(self, model: nn.Module, config: CurveBallGradientConfig):
        super().__init__(model, config)
        self.config: CurveBallGradientConfig = config
        self._params = [p for p in model.parameters() if p.requires_grad]
        self._momentum = tuple([torch.zeros_like(p) for p in self._params])

    def do_forward_and_backward(self, step=None) -> bool:
        return False

    def _precondition(self):
        config = self.config
        hvps = self.loss_hvp(tangents=self._momentum)
        grads = [p.grad for p in self._params]
        assert len(hvps) == len(grads) == len(self._momentum)
        for p, m, hvp, grad in zip(self._params, self._momentum, hvps, grads):
            hvp.add_(m, alpha=config.damping)
            grad_of_quadratic = hvp + grad
            if config.momentum_decay != 1:
                m.mul_(config.momentum_decay)
            m.add_(grad_of_quadratic, alpha=config.momentum)
            if p.grad is None:
                p.grad = m.clone()
            else:
                p.grad.add_(m)
