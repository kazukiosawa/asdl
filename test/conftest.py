from dataclasses import dataclass

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from asdl import LOSS_CROSS_ENTROPY


def get_mlp(in_dim, hid_dim, out_dim):
    layers = [nn.Linear(in_dim, hid_dim), nn.ReLU()]
    layers.append(nn.Linear(hid_dim, hid_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hid_dim, out_dim))
    return nn.Sequential(*layers)


def get_cnn(in_dim, hid_dim, out_dim):
    layers = [nn.Conv2d(in_dim, hid_dim, kernel_size=2), nn.ReLU()]
    layers.append(nn.Conv2d(hid_dim, hid_dim, kernel_size=2))
    layers.append(nn.ReLU())
    layers.append(nn.Flatten())
    layers.append(nn.Linear(hid_dim * 2 * 2, out_dim))
    return nn.Sequential(*layers)


class NetworkA(nn.Module):
    def __init__(self, network_type, in_dim, hid_dim, out_dim):
        super().__init__()
        if network_type == 'mlp':
            self.model = get_mlp(in_dim, hid_dim, out_dim)
        else:
            self.model = get_cnn(in_dim, hid_dim, out_dim)

    def forward(self, inputs, targets=None, flip=False):
        logits = self.model(inputs)
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            if flip:
                return loss, logits  # returns a tuple of (loss, logits)
            return logits, loss  # returns a tuple of (logits, loss)
        return logits  # returns only logits


@dataclass
class Output:
    loss: torch.Tensor
    logits: torch.Tensor


class NetworkB(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.model = get_mlp(in_dim, hid_dim, out_dim)

    def forward(self, inputs, targets=None, return_dict=True):
        logits = self.model(inputs)
        loss = None
        if targets is not None:
            # flatten (bs, seq) dimensions
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Hugging Face Transformers' style
        if not return_dict:
            return loss, logits if loss is not None else logits
        else:
            return Output(loss=loss, logits=logits)


@pytest.fixture
def model(network_type, in_dim, hid_dim, out_dim):
    return NetworkA(network_type, in_dim, hid_dim, out_dim)


@pytest.fixture
def sequence_model(in_dim, hid_dim, out_dim):
    return NetworkB(in_dim, hid_dim, out_dim)


@pytest.fixture
def loss_fn(loss_type):
    return F.cross_entropy if loss_type == LOSS_CROSS_ENTROPY else F.mse_loss


def get_single_data(in_dim, out_dim, loss_type, network_type):
    torch.random.manual_seed(0)
    if network_type == 'mlp':
        x = torch.randn(1, in_dim)
    else:
        x = torch.randn(1, in_dim, 4, 4)
    if loss_type == LOSS_CROSS_ENTROPY:
        t = torch.randint(out_dim, (1,))
    else:
        t = torch.randn(1, out_dim)
    return x, t


@pytest.fixture
def single_data(in_dim, out_dim, loss_type, network_type):
    return get_single_data(in_dim, out_dim, loss_type, network_type)


@pytest.fixture
def single_data_copy(batch_size, in_dim, out_dim, loss_type, network_type):
    x, t = get_single_data(in_dim, out_dim, loss_type, network_type)
    if network_type == 'mlp':
        x = x.repeat(batch_size, 1)
    else:
        x = x.repeat(batch_size, 1, 1, 1)
    if loss_type == LOSS_CROSS_ENTROPY:
        t = t.repeat(batch_size)
    else:
        t = t.repeat(batch_size, 1)
    return x, t


@pytest.fixture
def multi_data(batch_size, in_dim, out_dim, loss_type, network_type):
    if network_type == 'mlp':
        x = torch.randn(batch_size, in_dim)
    else:
        x = torch.randn(batch_size, in_dim, 4, 4)
    if loss_type == LOSS_CROSS_ENTROPY:
        t = torch.randint(out_dim, (batch_size,))
    else:
        t = torch.randn(batch_size, out_dim)
    return x, t


@pytest.fixture
def sequence_data(batch_size, seq_len, in_dim, out_dim):
    x = torch.randn(batch_size, seq_len, in_dim)
    t = torch.randint(-1, out_dim, (batch_size, seq_len))  # -1 for ignore_index
    return x, t
