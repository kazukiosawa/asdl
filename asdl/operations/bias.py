import torch
from torch import nn

from .operation import Operation
from .operation import OP_COV_KRON,OP_COV_SWIFT_KRON,OP_BATCH_GRADS,SHAPE_DIAG,OP_RFIM_RELU,OP_RFIM_SOFTMAX,OP_MEAN_INPUTS,OP_MEAN_OUTPUTS,OP_SPATIAL_MEAN_OUTPUTS,OP_OUT_SPATIAL_SIZE,OP_BFGS_KRON_S_AS,OP_COV_UNIT_WISE, OP_COV_UNIT_WISE_INV,OP_COV_DIAG, OP_COV_DIAG_INV,OP_GRAM_HADAMARD,OP_GRAM_DIRECT,OP_MEAN_OUTGRADS,OP_SPATIAL_MEAN_OUTGRADS,OP_SKETCHED_GRAM


class Bias(nn.Module):
    _supported_operations = []
    def __init__(self):
        super(Bias, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0)

    def forward(self, input):
        return input + self.weight


class BiasExt(Operation):
    """
    module.fixup_bias: 1

    Argument shapes
    in_data: n x f_in
    out_grads: n x f_out
    """
    _supported_operations = set([OP_COV_KRON,OP_COV_SWIFT_KRON,
                            OP_COV_DIAG, OP_COV_DIAG_INV,
                            OP_GRAM_DIRECT,OP_BATCH_GRADS])
    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return out_grads.view(N, -1).sum(dim=1).unsqueeze(-1)

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return out_grads.view(N, -1).sum(dim=1).square().sum()

    @staticmethod
    def cov_kron_A(module, in_data):
        return torch.ones((1, 1), device=in_data.device)

    @staticmethod
    def cov_kron_B(module, out_grads):
        N = out_grads.size(0)
        grad_grad = out_grads.view(N, -1).sum(dim=1).square().sum()
        return grad_grad.unsqueeze((0))
