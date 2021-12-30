import torch
from torch import nn

from .operation import Operation


class LayerNorm(Operation):
    """
    module.weight: normalized_shape
    module.bias: normalized_shape

    Argument shapes
    in_data: n x normalized_shape
    out_grads: n x normalized_shape

    normalized_shape: f[0] x f[1] x ... x f[-1]
    """
    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        return in_data.mul(out_grads) # n x normalized_shape
    
    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads
    
    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        grads = in_data.mul(out_grads)
        return grads.mul(grads).sum(dim=0) # normalized_shape
    
    @staticmethod
    def cov_diag_bias(module, out_grads):
        return out_grads.mul(out_grads).sum(dim=0) # normalized_shape
    