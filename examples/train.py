import os
import argparse
from collections import OrderedDict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import asdl

import wandb

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adam'
OPTIM_KFAC = 'kfac'
OPTIM_SMW_NGD = 'smw_ngd'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'kron_psgd'
OPTIM_KBFGS = 'kbfgs'
OPTIM_SENG = 'seng'


def main():
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(epoch)
        total_train_time += time.time() - start
        test(epoch)

    print(f'total_train_time: {total_train_time:.2f}s')
    print(f'avg_epoch_time: {total_train_time / args.epochs:.2f}s')
    print(f'avg_step_time: {total_train_time / args.epochs / num_steps_per_epoch * 1000:.2f}ms')
    if args.wandb:
        wandb.run.summary['total_train_time'] = total_train_time
        wandb.run.summary['avg_epoch_time'] = total_train_time / args.epochs
        wandb.run.summary['avg_step_time'] = total_train_time / args.epochs / num_steps_per_epoch


def train(epoch):
    model.train()
    for batch_idx, (x, t) in enumerate(train_loader):
        x, t = x.to(device), t.to(device)
        optimizer.zero_grad(set_to_none=True)

        # y = model(x)
        # loss = F.cross_entropy(y, t)
        # loss.backward()

        dummy_y = grad_maker.setup_model_call(model, x)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, t)
        y, loss = grad_maker.forward_and_backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            if args.wandb:
                log = {'epoch': epoch,
                       'iteration': (epoch - 1) * num_steps_per_epoch + batch_idx + 1,
                       'train_loss': float(loss),
                       'learning_rate': optimizer.param_groups[0]['lr']}
                wandb.log(log)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / num_steps_per_epoch, float(loss)))

        scheduler.step()


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    if args.wandb:
        log = {'epoch': epoch,
               'iteration': epoch * num_steps_per_epoch,
               'test_loss': test_loss,
               'test_accuracy': test_accuracy}
        wandb.log(log)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optim', default=OPTIM_KFAC)
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_args()
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.cuda.reset_accumulated_memory_stats()

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size, 'drop_last': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    common_kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_kwargs.update(common_kwargs)
    test_kwargs.update(common_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
    num_steps_per_epoch = len(train_loader)

    hidden_dim = args.hidden_dim
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(784, hidden_dim)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_dim, hidden_dim)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_dim, 10)),
    ])).to(device)

    if args.optim == OPTIM_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = asdl.PreconditioningConfig(data_size=args.batch_size, damping=args.damping)

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
    else:
        grad_maker = asdl.GradientMaker(model)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * num_steps_per_epoch)

    config = vars(args).copy()
    config.pop('wandb')
    if args.optim in [OPTIM_SGD, OPTIM_ADAM]:
        config.pop('damping')
    if args.wandb:
        wandb.init(config=config,
                   entity=os.environ.get('WANDB_ENTITY', None),
                   project=os.environ.get('WANDB_PROJECT', None),
                   )

    print('=====================')
    for key, value in config.items():
        print(f'{key}: {value}')
    print('=====================')

    torch.cuda.synchronize()
    try:
        main()
        max_memory = torch.cuda.max_memory_allocated()
    except RuntimeError as err:
        if 'CUDA out of memory' in str(err):
            print(err)
            max_memory = -1  # OOM
        else:
            raise RuntimeError(err)

    print(f'cuda_max_memory: {max_memory/float(1<<30):.2f}GB')
    if args.wandb:
        wandb.run.summary['cuda_max_memory'] = max_memory
