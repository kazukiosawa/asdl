import copy

import torch
from torch import nn
import torch.nn.functional as F

from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP, LOSS_MSE, LOSS_CROSS_ENTROPY
from asdfghjkl import FullNaturalGradientMaker, LayerWiseNaturalGradientMaker, KfacGradientMaker, UnitWiseNaturalGradientMaker, DiagNaturalGradientMaker


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

# batch
x = torch.randn(bs, 3, 32, 32)
y = torch.tensor([0] * bs, dtype=torch.long)
n_classes = 10


def assert_equal(tensor1, tensor2, msg=''):
    diff = tensor1 - tensor2
    relative_norm = diff.norm().item() / tensor1.norm().item()
    assert torch.equal(tensor1, tensor2), f'{msg} tensor1: {tensor1.norm().item()}, tensor2: {tensor2.norm().item()}, relative_diff: {relative_norm}'


for ngd_cls in [FullNaturalGradientMaker, LayerWiseNaturalGradientMaker, KfacGradientMaker, UnitWiseNaturalGradientMaker, DiagNaturalGradientMaker]:
    for fisher_type in [FISHER_EMP, FISHER_MC, FISHER_EXACT]:
        for loss_type in [LOSS_MSE, LOSS_CROSS_ENTROPY]:
            print(ngd_cls.__name__, fisher_type, loss_type)
            model1 = convnet(32, 16, n_classes=n_classes)
            model2 = copy.deepcopy(model1)

            if loss_type == LOSS_MSE:
                target = F.one_hot(y, n_classes)
            else:
                target = y

            ngd1 = ngd_cls(model1, fisher_type, loss_type, ema_decay=1)
            # 1st set of accumulations
            for i in range(3):
                ngd1.accumulate_curvature(x, target,
                                          seed=1 if fisher_type == FISHER_MC else None)
            ngd1.update_preconditioner()
            # 2nd set of accumulations to make sure inv is preserved
            for i in range(3):
                ngd1.accumulate_curvature(x, target,
                                          seed=1 if fisher_type == FISHER_MC else None)

            ngd2 = ngd_cls(model2, fisher_type, loss_type)
            ngd2.accumulate_curvature(x, target,
                                      seed=1 if fisher_type == FISHER_MC else None)
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
                        elif ngd_cls == UnitWiseNaturalGradientMaker:
                            assert_equal(f1.unit.data, f2.unit.data, 'uw_data')
                            assert_equal(f1.unit.inv, f2.unit.inv, 'uw_inv')
                        else:
                            assert_equal(f1.diag.weight, f2.diag.weight, 'diag_w')
                            assert_equal(f1.diag.bias, f2.diag.bias, 'diag_b')
                            assert_equal(f1.diag.weight_inv, f2.diag.weight_inv, 'daig_w_inv')
                            assert_equal(f1.diag.bias_inv, f2.diag.bias_inv, 'diag_b_inv')
