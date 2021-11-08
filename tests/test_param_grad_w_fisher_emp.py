import copy
import time

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from asdfghjkl import FISHER_EMP
from asdfghjkl import FullNaturalGradient, LayerWiseNaturalGradient, KFAC, DiagNaturalGradient


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


def time_f(f, n_iters=5, n_warmups=5):
    for _ in range(n_warmups):
        f()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        f()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f'{f.__name__}: {elapsed / n_iters:.3f}s')


bs = 128

# batch
x = torch.randn(bs, 3, 32, 32)
y = torch.tensor([0] * bs, dtype=torch.long)


def assert_equal(tensor1, tensor2, msg=''):
    diff = tensor1 - tensor2
    relative_norm = diff.norm().item() / tensor1.norm().item()
    assert torch.equal(tensor1, tensor2), f'{msg} tensor1: {tensor1.norm().item()}, tensor2: {tensor2.norm().item()}, relative_diff: {relative_norm}'


# test param.grad by fisher_emp vs loss.backward()
model1 = convnet(32, 16)
model2 = copy.deepcopy(model1)

model1.zero_grad(set_to_none=True)
ngd1 = FullNaturalGradient(model1, FISHER_EMP)
ngd1.refresh_curvature(x, y, calc_emp_loss_grad=True)

model2.zero_grad(set_to_none=True)
loss = cross_entropy(model2(x), y)
loss.backward()

for p1, p2 in zip(model1.parameters(), model2.parameters()):
    assert_equal(p1.grad, p2.grad)


# time fisher_emp w/ and w/o param.grad vs loss.backward()
for ng_cls in [FullNaturalGradient, LayerWiseNaturalGradient, KFAC, DiagNaturalGradient]:
    print(ng_cls.__name__)

    def fisher_emp_and_param_grad():
        model1.zero_grad(set_to_none=True)
        ngd1 = ng_cls(model1, FISHER_EMP)
        ngd1.refresh_curvature(x, y, calc_emp_loss_grad=True)
        assert all(p.grad is not None for p in model1.parameters())

    def fisher_emp():
        model1.zero_grad(set_to_none=True)
        ngd1 = ng_cls(model1, FISHER_EMP)
        ngd1.refresh_curvature(x, y, calc_emp_loss_grad=False)
        assert all(p.grad is None for p in model1.parameters())

    def param_grad():
        model1.zero_grad(set_to_none=True)
        loss = cross_entropy(model1(x), y)
        loss.backward()
        assert all(p.grad is not None for p in model1.parameters())

    time_f(fisher_emp_and_param_grad)
    time_f(fisher_emp)
    time_f(param_grad)
