import torch
from torch import nn
import torch.nn.functional as F

from .operation import Operation


class Linear(Operation):
    """
    module.weight: f_out x f_in
    module.bias: f_out x 1

    Argument shapes
    in_data: n x f_in
    out_grads: n x f_out
    """
    @staticmethod
    def preprocess_in_data(module, in_data, out_data):
        if in_data.ndim > 2:
            # n x * x f_in -> n x f_in
            in_data = in_data.flatten(end_dim=in_data.ndim - 2)
        return in_data

    @staticmethod
    def extend_in_data(in_data):
        # Extend in_data with ones.
        # linear: n x f_in
        #      -> n x (f_in + 1)
        shape = list(in_data.shape)
        shape[1] = 1
        ones = in_data.new_ones(shape)
        return torch.cat((in_data, ones), dim=1)

    @staticmethod
    def preprocess_out_grads(module, out_grads):
        if out_grads.ndim > 2:
            # n x * x f_out -> n x f_out
            out_grads = out_grads.flatten(end_dim=out_grads.ndim - 2)
        return out_grads

    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        return torch.bmm(
            out_grads.unsqueeze(2), in_data.unsqueeze(1)
        )  # n x f_out x f_in

    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads

    @staticmethod
    def grad_weight(module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor):
        return torch.matmul(out_grads.T, in_data)

    @staticmethod
    def grad_bias(module: nn.Module, out_grads: torch.Tensor):
        return torch.sum(out_grads, dim=0)

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        in_in = in_data.mul(in_data)  # n x f_in
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return torch.matmul(grad_grad.T, in_in)  # f_out x f_in

    @staticmethod
    def cov_diag_bias(module, out_grads):
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return grad_grad.sum(dim=0)  # f_out x 1

    @staticmethod
    def cov_kron_A(module, in_data):
        return torch.matmul(in_data.T, in_data)  # f_in x f_in

    @classmethod
    def cov_swift_kron_A(cls, module, in_data):
        n, f_in = in_data.shape
        if n < f_in:
            return in_data  # n x f_in
        else:
            return cls.cov_kron_A(module, in_data)  # f_in x f_in

    @staticmethod
    def cov_kron_B(module, out_grads):
        return torch.matmul(out_grads.T, out_grads)  # f_out x f_out

    @classmethod
    def cov_swift_kron_B(cls, module, out_grads):
        n, f_out = out_grads.shape
        if n < f_out:
            return out_grads  # n x f_out
        else:
            return cls.cov_kron_B(module, out_grads)  # f_out x f_out

    @classmethod
    def cov_kfe_A(cls, module, in_data):
        n, f_in = in_data.shape
        if n < f_in:
            _, _, Vt = torch.linalg.svd(in_data, full_matrices=True)
            return Vt.T  # f_in x f_in
        else:
            A = cls.cov_kron_A(module, in_data)
            _, U = torch.linalg.eigh(A)
            return U  # f_in x f_in

    @classmethod
    def cov_kfe_B(cls, module, out_grads):
        n, f_out = out_grads.shape
        if n < f_out:
            _, _, Vt = torch.linalg.svd(out_grads, full_matrices=True)
            return Vt.T  # f_out x f_out
        else:
            B = cls.cov_kron_B(module, out_grads)
            _, U = torch.linalg.eigh(B)
            return U  # f_out x f_out

    @classmethod
    def cov_kfe_scale(cls, module, in_data, out_grads, Ua, Ub, bias=True):
        n, f_in = in_data.shape
        _, f_out = out_grads.shape
        in_data_kfe = in_data.mm(Ua)
        out_grads_kfe = out_grads.mm(Ub)
        scale_w = torch.mm(out_grads_kfe.T ** 2, in_data_kfe ** 2) / n
        if bias:
            scale_b = (out_grads_kfe ** 2).mean(dim=0)
            return scale_w, scale_b
        return scale_w,

    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        n, f_in = in_data.shape[0], in_data.shape[1]
        in_in = torch.bmm(in_data.unsqueeze(2), in_data.unsqueeze(1)).view(n, -1)  # n x (f_in x f_in)
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return torch.matmul(grad_grad.T, in_in).view(-1, f_in, f_in)  # f_out x f_in x_fin

    @staticmethod
    def gram_A(module, in_data1, in_data2=None):
        if in_data2 is None:
            return torch.matmul(in_data1, in_data1.T)  # n x n
        return torch.matmul(in_data1, in_data2.T)  # n x n

    @staticmethod
    def gram_B(module, out_grads1, out_grads2=None):
        if out_grads2 is None:
            return torch.matmul(out_grads1, out_grads1.T)  # n x n
        return torch.matmul(out_grads1, out_grads2.T)  # n x n

    @staticmethod
    def rfim_relu(module, in_data, out_data):
        nu = torch.sigmoid(out_data) ** 2  # n x f_out
        xxt = torch.einsum('bi,bj->bij', in_data, in_data)  # n x f_in x f_in
        return torch.einsum('bi,bjk->ijk', nu, xxt)  # f_out x f_in x f_in

    @staticmethod
    def rfim_softmax(module, in_data, out_data):
        # equivalent to fisher_exact_for_cross_entropy
        probs = F.softmax(out_data, dim=1)  # n x f_out
        ppt = torch.bmm(probs.unsqueeze(2), probs.unsqueeze(1))  # n x f_out x f_out
        diag_p = torch.stack([torch.diag(p) for p in probs], dim=0)  # n x f_out x f_out
        f = diag_p - ppt  # n x f_out x f_out
        xxt = torch.einsum('bi,bj->bij', in_data, in_data)  # n x f_in x f_in
        return torch.einsum('bij,bkl->ikjl', f, xxt)  # (f_out)(f_in)(f_out)(f_in)

    @staticmethod
    def in_data_mean(module, in_data):
        return in_data.mean(dim=0)  # f_in

    @staticmethod
    def out_data_mean(module, out_data):
        return out_data.mean(dim=0)  # f_out

    @staticmethod
    def out_spatial_size(module, out_data):
        return 1

    @staticmethod
    def out_grads_mean(module, out_grads):
        return out_grads.mean(dim=0)  # f_out

    @staticmethod
    def bfgs_kron_s_As(module, in_data):
        H = module.bfgs.kron.A_inv  # f_in x f_in
        s = torch.mv(H, in_data.mean(dim=0))  # fin
        indata_s = torch.mv(in_data, s)  # n
        As = torch.mv(in_data.T, indata_s)  # f_in
        return s, As

