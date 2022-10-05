import argparse
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from asdfghjkl import LOSS_CROSS_ENTROPY, LOSS_MSE
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl import Scale, Bias

class Net(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.embedding = nn.Embedding(4, 16)
        self.conv = nn.Conv2d(1, 2, 3, bias=bias)
        self.batchnorm2 = nn.BatchNorm2d(2)
        self.linear = nn.Linear(8, 5, bias=bias)
        self.batchnorm1 = nn.BatchNorm1d(5)
        self.scale = Scale()
        self.bias = Bias()
        self.layernorm = nn.LayerNorm(5)
    
    def forward(self, x):
        x = self.embedding(x).view(-1, 1, 4, 4)
        x = F.relu(self.batchnorm2(self.conv(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.batchnorm1(self.linear(x)))
        x = self.bias(self.scale(x))
        x = self.layernorm(x)
        return x

@pytest.fixture
def model(cuda, bias):
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else "cpu")
    return Net(bias).to(device)

@pytest.fixture
def data(loss_type, batch_size):
    x = torch.randint(high=4, size=(batch_size,), dtype=torch.long)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0]*batch_size, dtype=torch.long)
    else:
        y = torch.randn(batch_size, 5)
    return x, y

def pytest_addoption(parser):
    parser.addoption("--loss_type", action="extend", nargs="+", type=str, choices=[LOSS_CROSS_ENTROPY, LOSS_MSE])
    parser.addoption("--fisher_type", action="extend", nargs="+", type=str, choices=[FISHER_EXACT, FISHER_MC, FISHER_EMP])
    parser.addoption("--fisher_shape", action="extend", nargs="+", type=str,
                     choices=[SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG])
    parser.addoption("--cuda", action="store_true")
    parser.addoption("--data_size", action="store", type=int, default=16)
    parser.addoption("--padding", action="store", type=int, default=0)
    parser.addoption("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.addoption("--batch_size", type=int, default=32)

def pytest_generate_tests(metafunc):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    
    if "cuda" in metafunc.fixturenames:
        cuda = metafunc.config.getoption("cuda")
        if cuda is not None:
            metafunc.parametrize("cuda", [cuda])

    if "data_size" in metafunc.fixturenames:
        data_size = metafunc.config.getoption("data_size")
        if data_size is not None:
            metafunc.parametrize("data_size", [data_size])

    if "padding" in metafunc.fixturenames:
        padding = metafunc.config.getoption("padding")
        if padding is not None:
            metafunc.parametrize("padding", [padding])
    
    if "bias" in metafunc.fixturenames:
        bias = metafunc.config.getoption("bias")
        if bias is not None:
            metafunc.parametrize("bias", [bias])
    
    if "batch_size" in metafunc.fixturenames:
        batch_size = metafunc.config.getoption("batch_size")
        if batch_size is not None:
            metafunc.parametrize("batch_size", [batch_size])
