import copy

import torch
from torch import nn
import torch.nn.functional as F

from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP, LOSS_MSE, LOSS_CROSS_ENTROPY
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG
from asdfghjkl import NaturalGradientMaker, FullNaturalGradientMaker, LayerWiseNaturalGradientMaker, KfacGradientMaker, DiagNaturalGradientMaker


def convnet(n_dim, n_channels, n_classes=10, kernel_size=3):
    n_features = int(n_dim / (2 ** 4)) ** 2 * n_channels
    model = nn.Sequential()
    model.add_module('conv1', nn.Conv2d(3, n_channels, kernel_size))
    model.add_module('pool1', nn.MaxPool2d(kernel_size))
    model.add_module('relu1', nn.ReLU())
    model.add_module('conv2', nn.Conv2d(n_channels, n_channels, kernel_size))
    model.add_module('pool2', nn.MaxPool2d(kernel_size))
    model.add_module('relu2', nn.ReLU())
    model.add_module('flatten', nn.Flatten())
    model.add_module('linear', nn.Linear(n_features, n_classes))
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


ngd_classes = [FullNaturalGradientMaker, LayerWiseNaturalGradientMaker, KfacGradientMaker, DiagNaturalGradientMaker]
shapes = [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG]

for ngd_cls, shape in zip(ngd_classes, shapes):
    for fisher_type in [FISHER_EMP, FISHER_MC, FISHER_EXACT]:
        for loss_type in [LOSS_MSE, LOSS_CROSS_ENTROPY]:
            print(ngd_cls.__name__, fisher_type, loss_type)
            model1 = convnet(32, 16, n_classes=n_classes)
            model2 = copy.deepcopy(model1)

            if loss_type == LOSS_MSE:
                target = F.one_hot(y, n_classes)
            else:
                target = y

            # class-wise shape selection
            fisher_shape = [(nn.Linear, shape), (nn.Conv2d, shape)]
            ngd1 = NaturalGradientMaker(model1, fisher_type, fisher_shape, loss_type)
            assert set(ngd1.modules_for(shape)) == set([model1.conv1, model1.conv2, model1.linear])

            # module-wise shape selection
            fisher_shape = [(model1.conv1, shape), (model1.conv2, shape), (model1.linear, shape)]
            ngd1 = NaturalGradientMaker(model1, fisher_type, fisher_shape, loss_type)
            assert set(ngd1.modules_for(shape)) == set([model1.conv1, model1.conv2, model1.linear])

            ngd2 = ngd_cls(model2, fisher_type, loss_type)
            assert set(ngd2.modules_for(shape)) == set([model2.conv1, model2.conv2, model2.linear])

            ngd1.refresh_curvature(x, target,
                                   seed=1 if fisher_type == FISHER_MC else None,
                                   calc_emp_loss_grad=True)
            ngd1.update_preconditioner()
            ngd1.precondition()

            ngd2.refresh_curvature(x, target,
                                   seed=1 if fisher_type == FISHER_MC else None,
                                   calc_emp_loss_grad=True)
            ngd2.update_preconditioner()
            ngd2.precondition()

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
