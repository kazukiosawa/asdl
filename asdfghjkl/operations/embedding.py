import torch
from torch import nn

from .operation import Operation


class Embedding(Operation):

    @staticmethod
    def batch_grads_weight(
        module: nn.Embedding, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        """
        in_data: n
        out_grads: n x embedding_dim

        Return:
            grads: n x num_embeddings x embedding_dim
        """
        size = in_data.shape + (module.num_embeddings, module.embedding_dim)
        grads = torch.zeros(size, device=module.weight.device)
        for i, index in enumerate(in_data):
            grads[i].index_put_((index,), out_grads[i])
            if module.padding_idx is not None:
                grads[i][module.padding_idx].fill_(0)
        return grads

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        out_out = out_grads.mul(out_grads)  # n x embedding_dim
        cov = torch.zeros_like(module.weight)
        for i, index in enumerate(in_data):
            cov.index_put_((index,), out_out[i], accumulate=True)
            if module.padding_idx is not None:
                cov[module.padding_idx].fill_(0)
        return cov
