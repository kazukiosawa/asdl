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

import asdfghjkl as asdl
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl import empirical_natural_gradient, empirical_natural_gradient2
from asdfghjkl.precondition.psgd import KronPreconditoner

import wandb

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adam'
OPTIM_NGD = 'ngd'
OPTIM_WOODBURY_NGD = 'woodbury_ngd'
OPTIM_WOODBURY_NGD2 = 'woodbury_ngd2'
OPTIM_PSGD = 'psgd'


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
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if args.optim == OPTIM_NGD:
            loss, outputs = ngd.update_curvature(inputs, targets, calc_emp_loss_grad=True)
            ngd.update_inv()
            ngd.precondition()
        elif args.optim == OPTIM_WOODBURY_NGD:
            loss = empirical_natural_gradient(model, inputs, targets, damping=args.damping)
        elif args.optim == OPTIM_WOODBURY_NGD2:
            loss = empirical_natural_gradient2(model, inputs, targets, damping=args.damping)
        elif args.optim == OPTIM_PSGD:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            psgd.update_preconditioner()
            psgd.precondition()
        else:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.wandb:
                log = {'epoch': epoch,
                       'iteration': (epoch - 1) * num_steps_per_epoch + batch_idx + 1,
                       'train_loss': float(loss),
                       'learning_rate': optimizer.param_groups[0]['lr']}
                wandb.log(log)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
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
    parser.add_argument('--optim', choices=[OPTIM_SGD, OPTIM_ADAM, OPTIM_NGD, OPTIM_WOODBURY_NGD, OPTIM_WOODBURY_NGD2, OPTIM_PSGD], default=OPTIM_NGD)
    parser.add_argument('--fisher_type', choices=[FISHER_EXACT, FISHER_MC, FISHER_EMP], default=FISHER_EXACT)
    parser.add_argument('--fisher_shape', choices=[SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG], default=SHAPE_FULL)
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

    ngd = psgd = None
    if args.optim == OPTIM_NGD:
        fisher_shape = [args.fisher_shape]
        ngd = asdl.NaturalGradient(model,
                                   fisher_type=args.fisher_type,
                                   fisher_shape=fisher_shape,
                                   damping=args.damping)
    elif args.optim == OPTIM_PSGD:
        psgd = KronPreconditoner(model)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * num_steps_per_epoch)

    config = vars(args).copy()
    config.pop('wandb')
    if args.optim != OPTIM_NGD:
        config.pop('fisher_type')
        config.pop('fisher_shape')
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
