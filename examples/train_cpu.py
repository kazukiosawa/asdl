import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import asdl

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adam'
OPTIM_KFAC = 'kfac'
OPTIM_SMW_NGD = 'smw_ngd'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'kron_psgd'
OPTIM_KBFGS = 'kbfgs'
OPTIM_SENG = 'seng'
OPTIM_SHAMPOO = 'shampoo'


def train(epoch):
    model.train()
    for batch_idx, (x, t) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        # y = model(x)
        # loss = F.mse_loss(y, t)
        # loss.backward()

        dummy_y = grad_maker.setup_model_call(model, x)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, t)
        y, loss = grad_maker.forward_and_backward()

        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(train_loader.dataset), 100. * batch_idx / num_steps_per_epoch, float(loss)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', default=OPTIM_KFAC)
    args = parser.parse_args()

    in_dim = 5
    hid_dim = 4
    out_dim = 3

    batch_size = 32
    num_iters = 10
    data_size = batch_size * num_iters
    epochs = 2

    model = nn.Sequential()
    model.append(nn.Linear(in_dim, hid_dim))
    model.append(nn.ReLU())
    model.append(nn.Linear(hid_dim, hid_dim))
    model.append(nn.ReLU())
    model.append(nn.Linear(hid_dim, out_dim))

    xs = torch.randn(data_size, in_dim)
    ts = torch.tensor([0] * data_size)
    train_set = torch.utils.data.TensorDataset(xs, ts)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    num_steps_per_epoch = len(train_loader)

    lr = 0.1
    weight_decay = 5.e-4
    damping = 1.

    if args.optim == OPTIM_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    config = asdl.PreconditioningConfig(data_size=batch_size, damping=damping)

    if args.optim == OPTIM_KFAC:
        grad_maker = asdl.KfacGradientMaker(model, config)
    elif args.optim == OPTIM_SMW_NGD:
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        grad_maker = asdl.PsgdGradientMaker(model, config)
    elif args.optim == OPTIM_KRON_PSGD:
        grad_maker = asdl.KronPsgdGradientMaker(model, config)
    elif args.optim == OPTIM_KBFGS:
        grad_maker = asdl.KronBfgsGradientMaker(model, config)
    elif args.optim == OPTIM_SENG:
        grad_maker = asdl.SengGradientMaker(model, config)
    elif args.optim == OPTIM_SHAMPOO:
        grad_maker = asdl.ShampooGradientMaker(model, config)
    else:
        raise ValueError(f'Invalid optim: {args.optim}')

    for epoch in range(1, epochs + 1):
        train(epoch)
