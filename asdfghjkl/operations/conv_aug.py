import torch
from torch import nn

from .conv import Conv2d
from ..utils import im2col_2d_aug, arr2col_1d_aug


class Conv2dAug(nn.Conv2d):
    
    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


class Conv1dAug(nn.Conv1d):
    
    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


# TODO: add improvements vom .conv to diag and other operations, check kfac
class Conv2dAugExt(Conv2d):
    """
    module.weight: c_out x c_in x k_h x k_w
    module.bias: c_out x 1

    Argument shapes
    in_data: n x (c_in)(kernel_size) x out_size
    out_grads: n x c_out x out_size

    kernel_size = (k_h)(k_w)
    out_size = output feature map size
    """
    @staticmethod
    def preprocess_in_data(module, in_data, out_data):
        if isinstance(module, nn.Conv2d):
            return im2col_2d_aug(in_data, module)
        elif isinstance(module, nn.Conv1d):
            return arr2col_1d_aug(in_data, module)
        raise ValueError('Invalid operation for module', module)

    @staticmethod
    def preprocess_out_grads(module, out_grads):
        # n x c x h_out x w_out -> n x c x (h_out)(w_out)
        if isinstance(module, Conv2dAug):
            return out_grads.flatten(start_dim=3)
        return out_grads

    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        grads = torch.matmul(
            out_grads, in_data.transpose(-1, -2)
        ).sum(dim=1)  # n x c_out x (c_in)(kernel_size)
        return grads.view(
            -1, *module.weight.size()
        )  # n x c_out x c_in x k_h x k_w

    @staticmethod
    def batch_grads_bias(module: nn.Module, out_grads: torch.tensor):
        return out_grads.sum(axis=[1, 3])  # n x c_out

    @staticmethod
    def batch_grads_aug_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        out_grads = out_grads.sum(dim=1)
        in_data = in_data.mean(dim=1)
        grads = torch.matmul(
            out_grads, in_data.transpose(-1, -2)
        )  # n x c_out x (c_in)(kernel_size)
        return grads.view(
            -1, *module.weight.size()
        )  # n x c_out x c_in x k_h x k_w

    @staticmethod
    def batch_grads_aug_bias(module: nn.Module, out_grads: torch.tensor):
        return out_grads.sum(axis=[1, 3])  # n x c_out

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        # efficient reduction for augmentation (sum and mean of in and out)
        grads = torch.matmul(
            out_grads.sum(dim=1), in_data.mean(dim=1).transpose(-1, -2)
        ) # n x k_aug x c_out x (c_in)(kernel_size)
        rst = grads.mul(grads).sum(dim=0)  # c_out x (c_in)(kernel_size)
        return rst.view_as(module.weight)  # c_out x c_in x k_h x k_w

    @staticmethod
    def cov_diag_bias(module, out_grads):
        grads = out_grads.sum(axis=[1, 3])  # n x c_out
        return grads.mul(grads).sum(axis=0)  # c_out x 1

    @staticmethod
    def cov_kron_A(module, in_data):
        m = in_data.sum(dim=1).transpose(0, 1).flatten(
            start_dim=1
        )  # (c_in)(kernel_size) x n(out_size)
        return torch.matmul(
            m, m.T
        )  # (c_in)(kernel_size) x (c_in)(kernel_size)

    @staticmethod
    def cov_kron_B(module, out_grads):
        out_size = out_grads.shape[-1]
        m = out_grads.mean(dim=1).transpose(0, 1).flatten(start_dim=1)  # c_out x n(out_size)
        return torch.matmul(m, m.T).div(out_size)  # c_out x c_out


class Conv1dAug(nn.Conv1d):
    
    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])
