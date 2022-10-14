import torch
from torch import nn

from ..utils import im2col_2d
from .operation import Operation


class Conv2d(Operation):
    """
    module.weight: c_out x c_in x k_h x k_w
    module.bias: c_out x 1

    Argument shapes
    in_data: n x (c_in)(kernel_size) x out_size
    out_data: n x c_out x out_size
    out_grads: n x c_out x h_out x w_out

    kernel_size = (k_h)(k_w)
    out_size = output feature map size
    """
    @staticmethod
    def preprocess_in_data(module, in_data, out_data):
        # n x c x h_in x w_in -> n x c(kh)(kw) x (h_out)(w_out)
        return im2col_2d(in_data, module)

    @staticmethod
    def extend_in_data(in_data):
        # Extend in_data with ones.
        # conv2d: n x (c_in)(kernel_size) x out_size
        #      -> n x {(c_in)(kernel_size) + 1} x out_size
        shape = list(in_data.shape)
        shape[1] = 1
        ones = in_data.new_ones(shape)
        return torch.cat((in_data, ones), dim=1)

    @staticmethod
    def preprocess_out_grads(module, out_grads):
        # n x c x h_out x w_out -> n x c x (h_out)(w_out)
        return out_grads.flatten(start_dim=2)

    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        grads = torch.bmm(
            out_grads, in_data.transpose(2, 1)
        )  # n x c_out x (c_in)(kernel_size)
        return grads.view(
            -1, *module.weight.size()
        )  # n x c_out x c_in x k_h x k_w

    @staticmethod
    def batch_grads_bias(module: nn.Module, out_grads: torch.tensor):
        return out_grads.sum(axis=2)  # n x c_out

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        c_out, c_in, kh, kw = module.weight.shape
        out_size = in_data.shape[-1]

        # select a calculation method that consumes less memory
        if out_size * (c_out + c_in * kh * kw) < c_out * c_in * kh * kw:
            in_in = torch.square(in_data).transpose(0, 1).flatten(start_dim=1)  # (c_in)(kernel_size) x n(out_size)
            grad_grad = torch.square(out_grads).transpose(0, 1).flatten(start_dim=1)  # c_out x n(out_size)
            rst = torch.matmul(grad_grad, in_in.T)  # c_out x (c_in)(kernel_size)
            return rst.view_as(module.weight)  # c_out x c_in x k_h x k_w
        else:
            bg = Conv2d.batch_grads_weight(module, in_data, out_grads)  # n x c_out x c_in x k_h x k_w
            return torch.square(bg).sum(dim=0)  # c_out x c_in x k_h x k_w

    @staticmethod
    def cov_diag_bias(module, out_grads):
        grads = out_grads.sum(axis=2)  # n x c_out
        return grads.mul(grads).sum(axis=0)  # c_out x 1

    @staticmethod
    def cov_kron_A(module, in_data):
        out_size = in_data.shape[-1]
        m = in_data.transpose(0, 1).flatten(
            start_dim=1
        )  # (c_in)(kernel_size) x n(out_size)
        return torch.matmul(
            m, m.T
        ).div(out_size)  # (c_in)(kernel_size) x (c_in)(kernel_size)

    @classmethod
    def cov_swift_kron_A(cls, module, in_data):
        n, cin_ks, _ = in_data.shape
        if n < cin_ks:
            return in_data.sum(dim=-1)  # n x (c_in)(kernel_size)
        else:
            return cls.cov_kron_A(module, in_data)  # (c_in)(kernel_size) x (c_in)(kernel_size)

    @staticmethod
    def cov_kron_B(module, out_grads):
        m = out_grads.transpose(0,
                                1).flatten(start_dim=1)  # c_out x n(out_size)
        return torch.matmul(m, m.T)  # c_out x c_out

    @classmethod
    def cov_swift_kron_B(cls, module, out_grads):
        n, c_out, _ = out_grads.shape
        if n < c_out:
            return out_grads.sum(dim=-1)  # n x c_out
        else:
            return cls.cov_kron_B(module, out_grads)  # c_out x c_out

    @classmethod
    def cov_kfe_A(cls, module, in_data):
        n, cin_ks, out_size = in_data.shape
        if n * out_size < cin_ks:
            m = in_data.permute(0, 2, 1).view(n * out_size, -1)
            _, _, Vt = torch.linalg.svd(m, full_matrices=True)
            return Vt.T  # (c_in)(kernel_size) x (c_in)(kernel_size)
        else:
            A = cls.cov_kron_A(module, in_data)
            _, U = torch.linalg.eigh(A)
            return U  # (c_in)(kernel_size) x (c_in)(kernel_size)

    @classmethod
    def cov_kfe_B(cls, module, out_grads):
        n, c_out, out_size = out_grads.shape  # n x c_out x out_size
        if n * out_size < c_out:
            m = out_grads.permute(0, 2, 1).view(n * out_size, -1)
            _, _, Vt = torch.linalg.svd(m, full_matrices=True)
            return Vt.T  # c_out x c_out
        else:
            B = cls.cov_kron_B(module, out_grads)
            _, U = torch.linalg.eigh(B)
            return U  # c_out x c_out

    @classmethod
    def cov_kfe_scale(cls, module, in_data, out_grads, Ua, Ub, bias=True):
        n, cin_ks, out_size = in_data.shape
        in_data = in_data.permute(0, 2, 1).contiguous().view(n * out_size, -1)  # n(out_size) x (c_in)(kernel_size)
        _, c_out, _ = out_grads.shape
        out_grads = out_grads.permute(0, 2, 1).contiguous().view(n * out_size, -1)  # n(out_size) x c_out
        in_data_kfe = in_data.mm(Ua)
        out_grads_kfe = out_grads.mm(Ub)
        scale_w = torch.mm(out_grads_kfe.T ** 2, in_data_kfe ** 2) / n
        if bias:
            scale_b = (out_grads_kfe ** 2).mean(dim=0)
            return scale_w, scale_b
        return scale_w,

    @staticmethod
    def gram_A(module, in_data1, in_data2=None):
        # n x (c_in)(kernel_size)(out_size)
        m1 = in_data1.flatten(start_dim=1)
        if in_data2 is None:
            return torch.matmul(m1, m1.T)  # n x n
        m2 = in_data2.flatten(start_dim=1)
        return torch.matmul(m1, m2.T)  # n x n

    @staticmethod
    def gram_B(module, out_grads1, out_grads2=None):
        out_size = out_grads1.shape[-1]
        # n x (c_out)(out_size)
        m1 = out_grads1.flatten(start_dim=1)
        if out_grads2 is None:
            return torch.matmul(m1, m1.T).div(out_size)  # n x n
        m2 = out_grads2.flatten(start_dim=1)
        return torch.matmul(m1, m2.T).div(out_size)  # n x n

    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        m = torch.bmm(out_grads, in_data.transpose(1, 2))  # n x c_out x cin_ks
        m = m.permute(1, 2, 0)  # c_out x cin_ks x n
        return torch.matmul(m, m.transpose(1, 2))  # c_out x cin_ks x cin_ks

    @staticmethod
    def in_data_mean(module, in_data):
        in_data = in_data.mean(dim=2)  # n x (c_in)(kernel_size)
        return in_data.mean(dim=0)  # (c_in)(kernel_size)

    @staticmethod
    def out_data_mean(module, out_data):
        # n x c_out x h_out x w_out -> n x c_out x (h_out)(w_out)
        out_data = out_data.flatten(start_dim=2)
        return out_data.mean(dim=2)  # n x c_out

    @staticmethod
    def out_spatial_size(module, out_data):
        return out_data.shape[-2] * out_data.shape[-1]

    @staticmethod
    def out_grads_mean(module, out_grads):
        return out_grads.mean(dim=2)  # n x c_out

    @staticmethod
    def bfgs_kron_s_As(module, in_data):
        H = module.bfgs.kron.A_inv  # (c_in)(ks) x (c_in)(ks)
        in_data = in_data.mean(dim=2)  # n x (c_in)(ks)
        s = torch.mv(H, in_data.mean(dim=0))  # (c_in)(ks)
        indata_s = torch.mv(in_data, s)  # n
        As = torch.mv(in_data.T, indata_s)  # (c_in)(ks)
        return s, As
