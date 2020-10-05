import torch

from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl import FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG

import pytorch_utils as pu

# get device (CUDA or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# setup data loader (MNIST)
train_loader, test_loader = pu.get_data_loader(pu.DATASET_MNIST, batch_size=128)

# initialize model (SmallSimpleCNN)
model, _ = pu.get_model(arch_name='SimpleCNN', arch_kwargs={'width_factor': 1})
model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())

print('===========================')
print(f'dataset: MNIST')
print(f'train set size: {len(train_loader.dataset)}')
print(f'network: SimpleCNN')
print(f'# params: {n_params}')
print(f'device: {device}')
print('===========================')

# evaluate matrices of various types and shapes using the entire training set
fisher_types = [FISHER_MC, COV]
fisher_shapes = [SHAPE_KRON, SHAPE_DIAG]
stats_name = 'full_batch'
print(f'Evaluating {fisher_types} of shape {fisher_shapes} ...')
matrix_manager = fisher_for_cross_entropy(model, fisher_types, fisher_shapes,
                                          data_loader=train_loader, stats_name=stats_name)
print(f'Done.')

# print eigenvalues summary
for ftype in fisher_types:
    for fshape in fisher_shapes:
        print('----------------')
        print(f'{fshape}.{ftype}')
        trace = matrix_manager.get_trace(ftype, fshape, stats_name)
        print(f'trace: {trace}')
        eigvals = matrix_manager.get_eigenvalues(ftype, fshape, stats_name)
        print(f'top-10 eigvals: {eigvals[:10].tolist()}')
