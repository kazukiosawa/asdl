import warnings
from .operation import *
from .linear import Linear
from .conv import Conv2d
from .batchnorm import BatchNorm1d, BatchNorm2d
from .layernorm import LayerNorm
from .embedding import Embedding
from .bias import Bias, BiasExt
from .scale import Scale, ScaleExt

__all__ = [
    'Linear',
    'Conv2d',
    'BatchNorm1d',
    'BatchNorm2d',
    'LayerNorm',
    'Bias',
    'Scale',
    'BiasExt',
    'ScaleExt',
    'get_op_class',
    'Operation',
    'OP_FULL_COV',
    'OP_FULL_CVP',
    'OP_COV',
    'OP_COV_INV',
    'OP_CVP',
    'OP_COV_KRON',
    'OP_COV_KRON_INV',
    'OP_COV_SWIFT_KRON',
    'OP_COV_SWIFT_KRON_INV',
    'OP_COV_KFE',
    'OP_COV_UNIT_WISE',
    'OP_COV_UNIT_WISE_INV',
    'OP_COV_DIAG',
    'OP_COV_DIAG_INV',
    'OP_RFIM_RELU',
    'OP_RFIM_SOFTMAX',
    'OP_GRAM_DIRECT',
    'OP_GRAM_HADAMARD',
    'OP_BATCH_GRADS',
    'OP_SAVE_INPUTS',
    'OP_SAVE_OUTGRADS',
    'OP_MEAN_INPUTS',
    'OP_MEAN_OUTPUTS',
    'OP_OUT_SPATIAL_SIZE',
    'OP_MEAN_OUTGRADS',
    'OP_BFGS_KRON_S_AS',
    'ALL_OPS',
    'FWD_OPS',
    'BWD_OPS',
    'BWD_OPS_WITH_INPUTS',
    'OperationContext'
]


def get_op_class(module):
    if isinstance(module, nn.Linear):
        return Linear
    elif isinstance(module, nn.Conv2d):
        return Conv2d
    elif isinstance(module, nn.BatchNorm1d):
        return BatchNorm1d
    elif isinstance(module, nn.BatchNorm2d):
        return BatchNorm2d
    elif isinstance(module, nn.LayerNorm):
        return LayerNorm
    elif isinstance(module, nn.Embedding):
        return Embedding
    elif isinstance(module, Bias):
        return BiasExt
    elif isinstance(module, Scale):
        return ScaleExt
    else:
        warnings.warn(f'Failed to lookup operations for Module {module}.')
        return None
