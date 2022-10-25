import copy

import torch
from torch import nn

from asdfghjkl import FISHER_EXACT, FISHER_EMP
from asdfghjkl import FullNaturalGradientMaker, LayerWiseNaturalGradientMaker, KfacGradientMaker, DiagNaturalGradientMaker
from asdfghjkl import PseudoBatchLoaderGenerator


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


# create a dataset
data_size = 200
batch_size = 10
x_all = torch.randn(data_size, 3, 32, 32)
y_all = torch.tensor([0] * data_size, dtype=torch.long)
dataset = torch.utils.data.TensorDataset(x_all, y_all)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
pbl_generator = PseudoBatchLoaderGenerator(dataloader, pseudo_batch_size=batch_size, batch_size=1)


def assert_equal(tensor1, tensor2, msg=''):
    diff = tensor1 - tensor2
    relative_norm = diff.norm().item() / tensor1.norm().item()
    assert relative_norm < 1e-4, f'{msg} tensor1: {tensor1.norm().item()}, ' \
                                 f'tensor2: {tensor2.norm().item()}, ' \
                                 f'relative_diff: {relative_norm}'


damping = 1e-2
for ngd_cls in [FullNaturalGradientMaker, LayerWiseNaturalGradientMaker, KfacGradientMaker, DiagNaturalGradientMaker]:
    for fisher_type in [FISHER_EMP, FISHER_EXACT]:
        print(ngd_cls.__name__, fisher_type)
        model1 = convnet(32, 16)
        model2 = copy.deepcopy(model1)

        ngd1 = ngd_cls(model1, fisher_type, damping=damping)
        # curvature by a batch
        for x, y in dataloader:
            ngd1.accumulate_curvature(x, y)
            break
        ngd1.update_preconditioner()

        # curvature by a pseudo-batch
        ngd2 = ngd_cls(model2, fisher_type, damping=damping)
        for pseudo_batch_loader in pbl_generator:
            ngd2.accumulate_curvature(data_loader=pseudo_batch_loader)
            break
        ngd2.update_preconditioner()

        if ngd_cls == FullNaturalGradientMaker:
            f1 = getattr(model1, fisher_type)
            f2 = getattr(model2, fisher_type)
            assert_equal(f1.data, f2.data, 'full_data')
            assert_equal(f1.inv, f2.inv, 'full_inv')
        else:
            for m1, m2 in zip(model1.modules(), model2.modules()):
                if isinstance(m1, (nn.Conv2d, nn.Linear)):
                    f1 = getattr(m1, fisher_type)
                    f2 = getattr(m2, fisher_type)
                    if ngd_cls == LayerWiseNaturalGradientMaker:
                        assert_equal(f1.data, f2.data, 'lw_data')
                        assert_equal(f1.inv, f2.inv, 'lw_inv')
                    elif ngd_cls == KfacGradientMaker:
                        assert_equal(f1.kron.A, f2.kron.A, 'A')
                        assert_equal(f1.kron.B, f2.kron.B, 'B')
                        assert_equal(f1.kron.A_inv, f2.kron.A_inv, 'A_inv')
                        assert_equal(f1.kron.B_inv, f2.kron.B_inv, 'B_inv')
                    else:
                        assert_equal(f1.diag.weight, f2.diag.weight, 'diag_w')
                        assert_equal(f1.diag.bias, f2.diag.bias, 'diag_b')
                        assert_equal(f1.diag.weight_inv, f2.diag.weight_inv, 'daig_w_inv')
                        assert_equal(f1.diag.bias_inv, f2.diag.bias_inv, 'diag_b_inv')
