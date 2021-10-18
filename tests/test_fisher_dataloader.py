import copy

import torch
from torch import nn

from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG


def convnet(n_dim, n_channels, n_classes=10, kernel_size=3):
    n_features = int(n_dim / (2 ** 4)) ** 2 * n_channels
    model = nn.Sequential(
        nn.Conv2d(3, n_channels, kernel_size),
        nn.MaxPool2d(kernel_size),
        nn.ReLU(),
        nn.Conv2d(n_channels, n_channels, kernel_size),
        nn.MaxPool2d(kernel_size),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(n_features, n_classes),
    )
    return model


bs = 8
model1 = convnet(32, 16)
model2 = copy.deepcopy(model1)

# batch
x = torch.randn(bs, 3, 32, 32)
y = torch.tensor([0] * bs, dtype=torch.long)

# create a dataset which contains {n_batches} copies of the batch (x, y)
n_batches = 4
dataset = torch.utils.data.TensorDataset(x.repeat(n_batches, 1, 1, 1), y.repeat(n_batches))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)


def assert_equal(tensor1, tensor2):
    assert torch.equal(tensor1, tensor2), f'tensor1: {tensor1.norm().item()}, tensor2: {tensor2.norm().item()}'


for fisher_type in [FISHER_EMP, FISHER_MC, FISHER_EXACT]:
    print(fisher_type)
    for data_average in [True, False]:
        # Fisher by a single batch
        fisher_for_cross_entropy(model1,
                                 fisher_type=fisher_type,
                                 fisher_shapes=[SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG],
                                 inputs=x,
                                 targets=y,
                                 data_average=data_average,
                                 scale=1 if data_average else n_batches,
                                 seed=1 if fisher_type == FISHER_MC else None)

        # Fisher by data loader (contains copies of the same batch)
        fisher_for_cross_entropy(model2,
                                 fisher_type=fisher_type,
                                 fisher_shapes=[SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG],
                                 data_loader=dataloader,
                                 data_average=data_average,
                                 seed=1 if fisher_type == FISHER_MC else None)

        # full
        assert_equal(getattr(model1, fisher_type).data, getattr(model2, fisher_type).data)
        for m1, m2 in zip(model1.modules(), model2.modules()):
            if isinstance(m1, (nn.Conv2d, nn.Linear)):
                f1 = getattr(m1, fisher_type)
                f2 = getattr(m2, fisher_type)
                # layer-wise
                assert_equal(f1.data, f2.data)
                # kron
                assert_equal(f1.kron.A, f2.kron.A)
                assert_equal(f1.kron.B, f2.kron.B)
                # diag
                assert_equal(f1.diag.weight, f2.diag.weight)
                assert_equal(f1.diag.bias, f2.diag.bias)
