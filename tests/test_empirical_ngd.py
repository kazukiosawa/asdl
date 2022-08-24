import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from asdfghjkl import FISHER_EMP, SHAPE_FULL
from asdfghjkl import NaturalGradientMaker, NaturalGradientMakerConfig
from asdfghjkl import SmwEmpNaturalGradientMaker, SmwEmpNaturalGradientMakerConfig


torch.random.manual_seed(0)
datasize = 10
batchsize = 2
dim = 2
n_layers = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = nn.Sequential().to(device)
for i in range(n_layers):
    model.add_module(f'fc{i}', nn.Linear(dim, dim).to(device))
    model.add_module(f'bn{i}', nn.BatchNorm1d(dim).to(device))

inputs = torch.randn(datasize, dim)
targets = torch.tensor([0] * datasize, dtype=torch.long)
dataset = torch.utils.data.TensorDataset(inputs, targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)

damping = 1e-2
model1 = copy.deepcopy(model)
optim1 = torch.optim.SGD(model1.parameters(), lr=1)
config = NaturalGradientMakerConfig(fisher_type=FISHER_EMP,
                                    fisher_shape=SHAPE_FULL,
                                    upd_curvature_interval=1,
                                    upd_inv_interval=1,
                                    damping=damping)
grad_maker1 = NaturalGradientMaker(model1, config)

model2 = copy.deepcopy(model)
optim2 = torch.optim.SGD(model2.parameters(), lr=1)
config = SmwEmpNaturalGradientMakerConfig(damping=damping)
grad_maker2 = SmwEmpNaturalGradientMaker(model2, config)

for i, (x, t) in enumerate(dataloader):
    print('*******************')
    print('batch', i)
    x, t = x.to(device), t.to(device)
    optim1.zero_grad(set_to_none=True)
    dummy_y = grad_maker1.setup_model_call(model1, x)
    grad_maker1.setup_loss_call(F.cross_entropy, dummy_y, t)
    y1, loss1 = grad_maker1.forward_and_backward(data_size=batchsize)
    optim1.step()

    optim2.zero_grad(set_to_none=True)
    dummy_y = grad_maker2.setup_model_call(model2, x)
    grad_maker2.setup_loss_call(F.cross_entropy, dummy_y, t)
    y2, loss2 = grad_maker2.forward_and_backward(data_size=batchsize)
    optim2.step()

    print('logits', float(y1.norm()), float(y2.norm()))
    print('loss', float(loss1), float(loss2))
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.requires_grad:
            print('grad', float(p1.grad.norm()), float(p2.grad.norm()))
