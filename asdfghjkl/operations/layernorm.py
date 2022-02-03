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
    def __init__(self, module, op_names, model_for_kernel=None):
        if OP_COV_KRON in op_names:
            op_names = op_names.copy()
            # kron operation is not supported. unit_wise will be used instead.
            op_names.remove(OP_COV_KRON)
            op_names.append(OP_COV_UNIT_WISE)

        if OP_GRAM_HADAMARD in op_names:
            op_names = op_names.copy()
            # gram hadamard operation is not supported. gram direct will be used instead.
            op_names.remove(OP_GRAM_HADAMARD)
            op_names.append(OP_GRAM_DIRECT)

        super().__init__(module, op_names, model_for_kernel)

    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        return in_data.mul(out_grads)  # n x normalized_shape
    
    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads
    
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
        blocks = torch.zeros(n_features, 2, 2).to(in_data.device)
        for i in range(n_features):
            blocks[i][0][0] = cov_ww[i]
            blocks[i][1][1] = cov_bb[i]
            blocks[i][0][1] = blocks[i][1][0] = cov_wb[i]
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
