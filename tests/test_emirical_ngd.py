import copy

import torch
from torch.nn import Linear, Sequential

from asdfghjkl import FISHER_EMP
from asdfghjkl import empirical_natural_gradient, FullNaturalGradient


torch.random.manual_seed(0)
datasize = 10
batchsize = 2
dim = 2
n_layers = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Sequential().to(device)
for i in range(n_layers):
    model.add_module(f'fc{i}', Linear(dim, dim).to(device))

inputs = torch.randn(datasize, dim)
targets = torch.tensor([0] * datasize, dtype=torch.long)
dataset = torch.utils.data.TensorDataset(inputs, targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)

damping = 1e-3
model1 = copy.deepcopy(model)
model2 = copy.deepcopy(model)
ngd = FullNaturalGradient(model1, fisher_type=FISHER_EMP, damping=damping)
optim1 = torch.optim.SGD(model1.parameters(), lr=1)
optim2 = torch.optim.SGD(model2.parameters(), lr=1)

for x, t in dataloader:
    x, t = x.to(device), t.to(device)
    model1.zero_grad(set_to_none=True)
    ngd.refresh_curvature(x, t, calc_emp_loss_grad=True)
    ngd.update_inv()
    ngd.precondition()
    optim1.step()

    model2.zero_grad(set_to_none=True)
    empirical_natural_gradient(model2, x, t, damping=damping)
    optim2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.requires_grad:
            torch.testing.assert_allclose(p1.grad, p2.grad, rtol=1e-4, atol=1e-3)
