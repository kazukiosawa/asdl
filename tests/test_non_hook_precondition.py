import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

import asdfghjkl as asdl
from asdfghjkl import FISHER_EMP, LOSS_MSE, LOSS_CROSS_ENTROPY, NaturalGradientMaker


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


n_iters = 3
bs = 8

# batch
xs = []
ys = []
for _ in range(n_iters):
    xs.append(torch.randn(bs, 3, 32, 32))
    ys.append(torch.tensor([0] * bs, dtype=torch.long))
n_classes = 10


def assert_equal(tensor1, tensor2, msg=''):
    diff = tensor1 - tensor2
    relative_norm = diff.norm().item() / tensor1.norm().item()
    assert relative_norm < 1e-5, f'{msg} tensor1: {tensor1.norm().item()}, tensor2: {tensor2.norm().item()}, relative_diff: {relative_norm}'
    print(msg, 'OK')


for loss_type in [LOSS_CROSS_ENTROPY, LOSS_MSE]:
    print(loss_type)
    model1 = convnet(32, 16, n_classes=n_classes)
    model2 = copy.deepcopy(model1)
    model3 = copy.deepcopy(model1)
    fisher_shape = [(nn.Conv2d, asdl.SHAPE_KRON), (nn.Linear, asdl.SHAPE_DIAG)]

    if loss_type == LOSS_MSE:
        ys = [F.one_hot(y, n_classes).float() for y in ys]

        def loss_fn(x1, x2):
            return 0.5 * F.mse_loss(x1, x2, reduction='sum')
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

    ngd1 = NaturalGradientMaker(model1,
                                fisher_type=FISHER_EMP,
                                fisher_shape=fisher_shape,
                                loss_type=loss_type)
    model1.zero_grad()
    for i in range(n_iters):
        ngd1.accumulate_curvature(xs[i], ys[i], data_average=False, calc_emp_loss_grad=True)
    ngd1.update_preconditioner()
    ngd1.precondition()

    ngd2 = NaturalGradientMaker(model2,
                                fisher_type=FISHER_EMP,
                                fisher_shape=fisher_shape)
    model2.zero_grad()
    for i in range(n_iters):
        with asdl.save_inputs_outgrads(model2) as cxt:
            loss = loss_fn(model2(xs[i]), ys[i])
            loss.backward()
            ngd2.accumulate_curvature(cxt=cxt)
    ngd2.update_preconditioner()
    ngd2.precondition()

    ngd3 = NaturalGradientMaker(model3,
                                fisher_type=FISHER_EMP,
                                fisher_shape=fisher_shape)
    with asdl.save_inputs_outgrads(model3) as cxt:
        model3.zero_grad()
        for i in range(n_iters):
            loss = loss_fn(model3(xs[i]), ys[i])
            loss.backward()
        ngd3.refresh_curvature(cxt=cxt)
    ngd3.update_preconditioner()
    ngd3.precondition()

    g1 = parameters_to_vector([p.grad for p in model1.parameters()])
    g2 = parameters_to_vector([p.grad for p in model2.parameters()])
    g3 = parameters_to_vector([p.grad for p in model3.parameters()])

    assert_equal(g1, g2, 'g1 vs g2')
    assert_equal(g1, g3, 'g1 vs g3')
