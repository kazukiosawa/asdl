import torch.nn as nn


def mlp(num_layers, input_dim, hidden_dim, num_classes=10):
    assert num_layers >= 2
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(input_dim, hidden_dim))
    model.add_module('relu1', nn.ReLU())
    idx = 2
    for i in range(num_layers - 2):
        model.add_module(f'fc{idx}', nn.Linear(hidden_dim, hidden_dim))
        model.add_module(f'relu{idx}', nn.ReLU())
        idx += 1
    model.add_module(f'fc{idx}', nn.Linear(hidden_dim, num_classes))
    return model

