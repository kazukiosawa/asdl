import torch
from torch import nn
import torch.nn.functional as F

from .operation import Operation

class ViTEmbeddings(Operation):
    """
    module.cls_token: 1 x 1 x hidden_size
    module.position_embeddings: 1 x (num_pathces+1) x hidden_size

    Argument shapes
    in_data: n x c_in x h x w
    out_grads: n x (num_pathces+1) x hidden_size
    """
    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        # cls_token
        return out_grads[:, 0, :].view(-1, *module.cls_token.size()) # n x 1 x 1 x hidden_size
    
    @staticmethod
    def batch_grads_bias(module, out_grads):
        # position_embeddings
        return out_grads.unsqueeze(1) # n x 1 x (num_patches+1) x hidden_size
    
    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        # cls_token
        grad_grad = out_grads[:, 0, :] ** 2 # n x hidden_size
        return grad_grad.sum(dim=0).view_as(module.cls_token) # 1 x 1 x hidden_size
    
    @staticmethod
    def cov_diag_bias(module, out_grads):
        # position_embeddings
        grad_grad = out_grads ** 2 # n x (num_pathces+1) x hidden_size
        return grad_grad.sum(dim=0, keepdim=True) # 1 x (num_pathces+1) x hidden_size
    
    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        raise ValueError(
            f'{OP_COV_UNIT_WISE} operation is not supported in ViTEmbeddings.'
        )

    @staticmethod
    def cov_kron_A(module, in_data):
        raise ValueError(
            f'{OP_COV_KRON} operation is not supported in ViTEmbeddings.'
        )
    
    @staticmethod
    def cov_kron_B(module, out_grads):
        raise ValueError(
            f'{OP_COV_KRON} operation is not supported in ViTEmbeddings.'
        )

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        raise ValueError(
            f'{OP_GRAM_HADAMARD} operation is not supported in ViTEmbeddings.'
        )

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        raise ValueError(
            f'{OP_GRAM_HADAMARD} operation is not supported in ViTEmbeddings.'
        )