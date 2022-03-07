import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from asdfghjkl import LOSS_CROSS_ENTROPY, LOSS_MSE
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl import Scale, Bias

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 16)
        self.conv = nn.Conv2d(1, 2, 3)
        self.batchnorm2 = nn.BatchNorm2d(2)
        self.linear = nn.Linear(8, 5)
        self.batchnorm1 = nn.BatchNorm1d(5)
        self.scale = Scale()
        self.bias = Bias()
        self.layernorm = nn.LayerNorm(5)
    
    def forward(self, x):
        x = self.embedding(x).view(32, 1, 4, 4)
        x = F.relu(self.batchnorm2(self.conv(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.batchnorm1(self.linear(x)))
        x = self.bias(self.scale(x))
        x = self.layernorm(x)
        return x

@pytest.fixture()
def model():
    return Net()

@pytest.fixture
def data(loss_type):
    x = torch.randint(high=4, size=(32,))
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0]*32, dtype=torch.long)
    else:
        y = torch.randn(32, 5)
    return x, y

def pytest_addoption(parser):
    parser.addoption("--loss_type", action="extend", nargs="+", type=str, choices=[LOSS_CROSS_ENTROPY, LOSS_MSE])
    parser.addoption("--fisher_type", action="extend", nargs="+", type=str, choices=[FISHER_EXACT, FISHER_MC, FISHER_EMP])
    parser.addoption("--fisher_shape", action="extend", nargs="+", type=str,
                     choices=[SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG])

def pytest_generate_tests(metafunc):
    if "loss_type" in metafunc.fixturenames:
        loss_types = metafunc.config.getoption("loss_type")
        if loss_types is None:
            loss_types = [LOSS_CROSS_ENTROPY, LOSS_MSE]
        metafunc.parametrize("loss_type", loss_types)

    if "fisher_type" in metafunc.fixturenames:
        fisher_types = metafunc.config.getoption("fisher_type")
        if fisher_types is None:
            fisher_types = [FISHER_EXACT, FISHER_MC, FISHER_EMP]
        metafunc.parametrize("fisher_type", fisher_types)
    
    if "fisher_shape" in metafunc.fixturenames or "fisher_shapes" in metafunc.fixturenames:
        fisher_shapes = metafunc.config.getoption("fisher_shape")
        if fisher_shapes is None:
            fisher_shapes = [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG]
        if "fisher_shapes" in metafunc.fixturenames:
            fisher_shapes = [fisher_shapes]
        metafunc.parametrize("fisher_shape", fisher_shapes)
    