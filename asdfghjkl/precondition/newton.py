from typing import Tuple, Union, Any
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ..grad_maker import GradientMaker
from ..hessian import HessianMaker, HessianConfig
from ..matrices import SHAPE_FULL


__all__ = ['NewtonGradientConfig', 'NewtonGradientMaker']


@dataclass
class NewtonGradientConfig:
    damping: float = 1.e-8
    absolute: bool = False


class NewtonGradientMaker(GradientMaker):
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        hessian_config = HessianConfig(hessian_shapes=[SHAPE_FULL])
        self.hessian_maker = HessianMaker(model, hessian_config)

    def forward_and_backward(self,
                             scale=1.,
                             accumulate=False
                             ) -> Union[Tuple[Any, Tensor], Any]:
        self.delegate_forward_and_backward(self.hessian_maker,
                                           scale=scale,
                                           accumulate=accumulate,
                                           calc_loss_grad=True)
        self.precondition()
        if self._loss_fn is None:
            return self._model_output
        else:
            return self._model_output, self._loss

    def precondition(self):
        hessian = self.model.hessian.data
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if self.config.absolute:
            L, Q = torch.linalg.eigh(hessian)
            hessian = Q @ torch.abs(torch.diag(L)) @ Q.T
        diag = torch.diagonal(hessian)
        diag += self.config.damping
        g = parameters_to_vector(grads)
        vector_to_parameters(torch.linalg.solve(hessian, g), grads)

