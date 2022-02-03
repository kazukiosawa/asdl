import torch
from torch import nn

from .operation import Operation, OP_COV_KRON, OP_COV_UNIT_WISE, OP_GRAM_HADAMARD, OP_GRAM_DIRECT


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
        raise NotImplementedError

    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        raise NotImplementedError
