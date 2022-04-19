import torch
from torch import nn

from .operation import Operation


class Embedding(Operation):

    @staticmethod
    def batch_grads_weight(
        module: nn.Embedding, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        """
        in_data: n x *
        out_grads: n x * x embedding_dim

        Return:
            grads: n x num_embeddings x embedding_dim
        """
        in_data = in_data.flatten()  # n x * -> n
        out_grads = out_grads.flatten(end_dim=-2)  # n x * x embedding_dim -> n x embedding_dim

        size = in_data.shape + (module.num_embeddings, module.embedding_dim)
        grads = torch.zeros(size, device=module.weight.device)
        for i, index in enumerate(in_data):
            grads[i].index_put_((index,), out_grads[i])
            if module.padding_idx is not None:
                grads[i][module.padding_idx].fill_(0)
        return grads

    @staticmethod
    def cov_kron_A(module, in_data):
        """
        in_data: n x *

        Return:
            A: num_embeddings x num_embeddings
        """
        counts = torch.stack(
            [torch.bincount(in_data[i].int(), minlength=module.num_embeddings) for i in range(in_data.shape[0])])
        counts = counts.float().to(module.weight.device)
        return torch.matmul(counts.T, counts)  # num_embeddings x num_embeddings

    @staticmethod
    def cov_kron_B(module, out_grads):
        """
        out_grads: n x * x embedding_dim

        Return:
            B: embedding_dim x embedding_dim
        """
        out_grads = out_grads.flatten(end_dim=-2)  # n x * x embedding_dim -> n x embedding_dim
        return torch.matmul(out_grads.T, out_grads)  # embedding_dim x embedding_dim

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        """
        in_data: n x *
        out_grads: n x * x embedding_dim

        Return:
            cov: num_embeddings x embedding_dim
        """
        in_data = in_data.flatten()  # n x * -> n
        out_grads = out_grads.flatten(end_dim=-2)  # n x * x embedding_dim -> n x embedding_dim

        out_out = out_grads.mul(out_grads)  # n x embedding_dim
        cov = torch.zeros_like(module.weight)
        for i, index in enumerate(in_data):
            cov.index_put_((index,), out_out[i], accumulate=True)
            if module.padding_idx is not None:
                cov[module.padding_idx].fill_(0)
        return cov
