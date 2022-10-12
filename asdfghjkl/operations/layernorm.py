import torch
from torch import nn

from .operation import Operation, OP_COV_KRON, OP_COV_UNIT_WISE, OP_GRAM_HADAMARD, OP_GRAM_DIRECT


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
    def preprocess_in_data(module, in_data, out_data):
        # restore normalized input
        in_data_norm = (out_data - module.bias).div(module.weight)
        in_data = in_data_norm
        # n x * x norm_shape -> n x norm_shape
        norm_shape_len = len(module.weight.shape)
        in_data_shape_len = len(in_data.shape)
        if norm_shape_len < in_data_shape_len - 1:
            in_data = in_data.flatten(end_dim=-norm_shape_len - 1)
        return in_data

    @staticmethod
    def preprocess_out_grads(module, out_grads):
        # n x * x norm_shape -> n x norm_shape
        norm_shape_len = len(module.weight.shape)
        out_grads_shape_len = len(out_grads.shape)
        if norm_shape_len < out_grads_shape_len - 1:
            out_grads = out_grads.flatten(end_dim=-norm_shape_len - 1)
        return out_grads

    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        return in_data.mul(out_grads)  # n x normalized_shape
    
    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads

    @staticmethod
    def grad_weight(module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor):
        return in_data.mul(out_grads).sum(dim=0)

    @staticmethod
    def grad_bias(module: nn.Module, out_grads: torch.Tensor):
        return out_grads.sum(dim=0)

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        grads = in_data.mul(out_grads)
        return grads.mul(grads).sum(dim=0)  # normalized_shape
    
    @staticmethod
    def cov_diag_bias(module, out_grads):
        return out_grads.mul(out_grads).sum(dim=0)  # normalized_shape
    
    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        n_features = in_data.flatten(start_dim=1).shape[1]  # (f[0] x f[1] x ... x f[-1])
        grads_w = in_data.mul(out_grads)  # n x normalized_shape
        grads_b = out_grads  # n x normalized_shape
        cov_ww = (grads_w ** 2).sum(0).flatten()  # n_features x 1
        cov_bb = (grads_b ** 2).sum(0).flatten()  # n_features x 1
        cov_wb = (grads_w * grads_b).sum(0).flatten()  # n_features x 1
        blocks = torch.vstack([cov_ww, cov_wb, cov_wb, cov_bb]).reshape(2, 2, n_features).transpose(0, 2)
        return blocks  # n_features x 2 x 2

    @staticmethod
    def cov_kron_A(module, in_data):
        raise ValueError(
            f'{OP_COV_KRON} operation is not supported in LayerNorm.'
        )

    @staticmethod
    def cov_kron_B(module, out_grads):
        raise ValueError(
            f'{OP_COV_KRON} operation is not supported in LayerNorm.'
        )

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        raise ValueError(
            f'{OP_GRAM_HADAMARD} operation is not supported in LayerNorm.'
        )

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        raise ValueError(
            f'{OP_GRAM_HADAMARD} operation is not supported in LayerNorm.'
        )
