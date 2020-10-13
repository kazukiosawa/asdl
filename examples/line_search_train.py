import argparse
import yaml
import random
import json
import warnings
import copy

import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import pytorch_utils as pu
from asdfghjkl.kernel import *
from asdfghjkl import FISHER_EXACT, SHAPE_FULL, SHAPE_BLOCK_DIAG
from asdfghjkl import fisher_free_for_cross_entropy, hessian_free
from asdfghjkl.precondition import NaturalGradient, LayerWiseNaturalGradient, KFAC, DiagNaturalGradient


# ignore warning from PIL/TiffImagePlugin.py
warnings.filterwarnings('ignore', message='.*(C|c)orrupt\sEXIF\sdata.*')

# yapf: disable
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=pu.DATASET_MNIST,
                    choices=[
                        pu.DATASET_CIFAR10, pu.DATASET_CIFAR100,
                        pu.DATASET_FMNIST, pu.DATASET_MNIST,
                        pu.DATASET_SVHN, pu.DATASET_IMAGENET
                    ],
                    help='name of dataset')
parser.add_argument('--dataset_root', type=str, default='./data',
                    help='root of dataset')
parser.add_argument('--train_root', type=str, default=None,
                    help='root of ImageNet train')
parser.add_argument('--val_root', type=str, default=None,
                    help='root of ImageNet val')
parser.add_argument('--epochs', type=int, default=200,
                    help='maximum number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=None,
                    help='input batch size for testing')
parser.add_argument('--kernel_batch_size', type=int, default=16)
parser.add_argument('--arch_file', type=str, default=None,
                    help='name of file which defines the architecture')
parser.add_argument('--arch', type=str,
                    help='name of the architecture')
parser.add_argument('--arch_args', type=json.loads, default={},
                    help='[JSON] arguments for the architecture')
parser.add_argument('--optim', type=str, default='SGD',
                    help='name of the optimizer')
parser.add_argument('--momentum', type=float, default=0.)
parser.add_argument('--damping', type=float, default=1e-2)
parser.add_argument('--condition_number', type=float, default=None)
parser.add_argument('--cg_tol', type=float, default=1e-3)
parser.add_argument('--interval', type=float, default=1e-5)
parser.add_argument('--max_lr', type=float, default=10)
# Options
parser.add_argument('--distributed_backend', type=str, default='nccl',
                    help='backend for distributed init_process_group')
parser.add_argument('--download', type=bool, default=False,
                    help='if True, downloads the dataset (CIFAR-10 or 100)')
parser.add_argument('--no_cuda', action='store_true',
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--n_workers', type=int, default=4,
                    help='number of sub processes for data loading')
parser.add_argument('--config', default=None, nargs='+',
                    help='config YAML file path')
parser.add_argument('--run_id', type=str, default=None,
                    help='If None, wandb.run.id will be used.')
parser.add_argument('--turn_off_wandb', action='store_true',
                    help='If True, no information will be sent to W&B.')
# yapf: enable


def topk_correct(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def evaluate(data_loader):
    data_size = len(data_loader.dataset)
    total_loss = 0
    correct_1 = correct_5 = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_fn = loss_fn.to(device)

    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += loss_fn(outputs, targets).item()
            corrects = topk_correct(outputs, targets, (1, 5))
            correct_1 += corrects[0]
            correct_5 += corrects[1]

    if is_distributed:
        # pack
        packed_tensor = torch.tensor([total_loss, correct_1,
                                      correct_5]).to(device)
        # all-reduce
        dist.all_reduce(packed_tensor)
        # unpack
        total_loss = packed_tensor[0].item()
        correct_1 = packed_tensor[1].item()
        correct_5 = packed_tensor[2].item()

    loss = total_loss / data_size
    accuracy1 = correct_1 / data_size
    accuracy5 = correct_5 / data_size

    return accuracy1, accuracy5, loss


def reduce_gradient():
    packed_tensor = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
    dist.all_reduce(packed_tensor)
    packed_tensor.div_(world_size)
    pointer = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        numel = p.numel()
        p.grad.copy_(packed_tensor[pointer: pointer + numel].view_as(p.grad))
        pointer += numel


def gradient(inputs, targets):
    model.zero_grad()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()

    if is_distributed:
        reduce_gradient()


def second_order_gradient(inputs, targets, damping=1e-5):
    gradient(inputs, targets)
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            grads.append(p.grad.clone().detach().requires_grad_(False))

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    sg = hessian_free(model,
                      loss_fn,
                      grads,
                      inputs=inputs,
                      targets=targets,
                      damping=damping,
                      max_iters=None,
                      is_distributed=is_distributed,
                      tol=args.cg_tol,
                      print_progress=False)

    i = 0
    for p in model.parameters():
        if p.requires_grad:
            p.grad.copy_(sg[i])
            i += 1


def natural_gradient_by_precondition(inputs, targets, precond_class, fisher_type=FISHER_EXACT, damping=1e-5):
    precond = precond_class(model, fisher_type=fisher_type, damping=damping)
    precond.update_curvature(inputs, targets)

    if is_distributed:
        # reduce curvature
        precond.reduce_curvature(all_reduce=True)

    precond.update_inv()
    gradient(inputs, targets)
    precond.precondition()


def natural_gradient_by_fisher_free(inputs, targets, fisher_type=FISHER_EXACT, fisher_shape=SHAPE_FULL, precond_class=None, damping=1e-5):
    if precond_class is not None:
        precond = precond_class(model, fisher_type=fisher_type, damping=damping)
        precond.update_curvature(inputs, targets)
        if is_distributed:
            # reduce curvature
            precond.reduce_curvature(all_reduce=True)
        precond.update_inv()
    else:
        precond = None

    gradient(inputs, targets)
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            grads.append(p.grad.clone().detach().requires_grad_(False))

    ng = fisher_free_for_cross_entropy(model,
                                       grads,
                                       fisher_type=fisher_type,
                                       fisher_shape=fisher_shape,
                                       inputs=inputs,
                                       targets=targets,
                                       damping=damping,
                                       max_iters=None,
                                       is_distributed=is_distributed,
                                       tol=args.cg_tol,
                                       preconditioner=precond,
                                       print_progress=False)

    i = 0
    for p in model.parameters():
        if p.requires_grad:
            p.grad.copy_(ng[i])
            i += 1


def natural_gradient_by_kernel_free(inputs, targets, damping=1e-5):
    kernel_free_cross_entropy(model,
                              inputs,
                              targets,
                              damping=damping,
                              is_distributed=is_distributed,
                              tol=args.cg_tol,
                              print_progress=False)


def natural_gradient_by_kernel(inputs, targets, data_loader, kernel_fn, damping=1e-5, efficient=False):
    if is_distributed:
        kernel = batch(kernel_fn, model, data_loader, is_distributed=True, store_on_device=False, gather_type='split')
        parallel_efficient_natural_gradient_cross_entropy(model, inputs, targets, kernel, damping=damping)
    else:
        kernel = batch(kernel_fn, model, data_loader, store_on_device=False)
        if efficient:
            assert kernel.ndim == 3
            kernel = kernel.permute(2, 0, 1)  # c x n x n
            efficient_natural_gradient_cross_entropy(model, inputs, targets, kernel, damping=damping)
        else:
            natural_gradient_cross_entropy(model, inputs, targets, kernel, damping=damping)


def get_mini_batch_gradient(inputs, targets, grad_fn, **grad_kwargs):
    grad_fn(inputs, targets, **grad_kwargs)

    grads = []
    for p in model.parameters():
        if p.requires_grad:
            grads.append(p.grad.clone().detach().requires_grad_(False))
        else:
            grads.append(torch.zeros_like(p))

    return grads


def line_search(vectors):
    # keep init params
    init_params = [p.clone().detach().requires_grad_(False) for p in model.parameters()]

    interval = args.interval
    max_lr = args.max_lr

    lr = interval
    last_acc = None
    while True:
        for i, param in enumerate(model.parameters()):
            param.data.copy_(init_params[i].sub(vectors[i].mul(lr)))
        new_acc = evaluate(test_loader)[0]
        if last_acc is not None:
            improvement = new_acc - last_acc
            if improvement / last_acc < -5e-2:
                break
            elif abs(improvement / last_acc) < 5e-2:
                interval *= 2
        last_acc = new_acc
        lr += interval
        if lr > max_lr:
            break

    # restore best params
    best_lr = lr - interval
    for i, param in enumerate(model.parameters()):
        param.data.copy_(init_params[i].sub(vectors[i].mul(best_lr)))

    return best_lr, last_acc


def train(epoch, iteration, **grad_kwargs):
    momentum = args.momentum
    global grads_momentum

    if is_distributed:
        # deterministically shuffle based on epoch
        train_loader.sampler.set_epoch(epoch)

    for batch_id, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        model.train()

        eigvals = kernel_eigenvalues(model,
                                     inputs,
                                     top_n=1,
                                     cross_entropy=True,
                                     eigenvectors=False,
                                     is_distributed=is_distributed,
                                     print_progress=False)
        top_eig = eigvals[0] / args.batch_size
        if args.condition_number:
            args.damping = top_eig / args.condition_number
            if args.optim != 'sgd':
                grad_kwargs['damping'] = args.damping

        grads = get_mini_batch_gradient(inputs, targets, **grad_kwargs)

        if grads_momentum is None or momentum == 0.:
            grads_momentum = copy.deepcopy(grads)
        else:
            for i in range(len(grads)):
                grads_momentum[i].mul_(momentum).add_(grads[i])

        lr, test_acc_1 = line_search(grads_momentum)

        log = {
            'epoch': epoch,
            'iteration': iteration,
            'learning_rate': lr,
            'test_acc@1': test_acc_1,
            'fisher_top_eig': top_eig,
            'damping': args.damping
        }

        if is_master:
            logger.print_report(log)
            logger.log(log)

        iteration += 1


def main():
    if is_master:
        logger.print_header()

    base_kwargs = {
        'sgd': {
            'grad_fn': gradient,
        },
        'newton': {
            'grad_fn': second_order_gradient,
        },
        'ngd': {
            'grad_fn': natural_gradient_by_precondition,
            'precond_class': NaturalGradient,
            'fisher_type': FISHER_EXACT,
        },
        'ngd_ntk': {
            'grad_fn': natural_gradient_by_kernel,
            'kernel_fn': empirical_direct_ntk,
        },
        'ngd_cg': {
            'grad_fn': natural_gradient_by_fisher_free,
            'fisher_type': FISHER_EXACT,
            'fisher_shape': SHAPE_FULL,
            'precond_class': None,
        },
        'ngd_ntk_cg': {
            'grad_fn': natural_gradient_by_kernel_free,
        },
        'lw_ngd': {
            'grad_fn': natural_gradient_by_precondition,
            'precond_class': LayerWiseNaturalGradient,
            'fisher_type': FISHER_EXACT,
        },
        'lw_ngd_cg': {
            'grad_fn': natural_gradient_by_fisher_free,
            'fisher_type': FISHER_EXACT,
            'fisher_shape': SHAPE_BLOCK_DIAG,
            'precond_class': KFAC,
        },
        'kfac': {
            'grad_fn': natural_gradient_by_precondition,
            'precond_class': KFAC,
            'fisher_type': FISHER_EXACT,
        },
        'diag_ngd': {
            'grad_fn': natural_gradient_by_precondition,
            'precond_class': DiagNaturalGradient,
            'fisher_type': FISHER_EXACT,
        },
        'cw_ngd': {
            'grad_fn': natural_gradient_by_kernel,
            'kernel_fn': empirical_class_wise_direct_ntk,
            'efficient': False,
        },
        'cw_ngd_eff': {
            'grad_fn': natural_gradient_by_kernel,
            'kernel_fn': empirical_class_wise_direct_ntk,
            'efficient': True,
        },
    }

    start_epoch = 1
    last_iteration = 0
    n_iters_per_epoch = len(train_loader)
    grad_kwargs = base_kwargs[args.optim]
    if args.optim != 'sgd':
        grad_kwargs['damping'] = args.damping
    for epoch in range(start_epoch, args.epochs + 1):
        iteration = last_iteration + 1
        train(epoch, iteration, **grad_kwargs)
        last_iteration = epoch * n_iters_per_epoch

        train_acc_1, train_acc_5, train_loss = evaluate(train_loader)
        test_acc_1, test_acc_5, test_loss = evaluate(test_loader)

        if is_master:
            log = {
                'epoch': epoch,
                'iteration': last_iteration,
                'train_acc@1': train_acc_1,
                'train_loss': train_loss,
                'test_acc@1': test_acc_1,
                'test_loss': test_loss,
            }
            logger.print_report(log)
            logger.log(log)


if __name__ == '__main__':
    args = parser.parse_args()
    dict_args = vars(args)

    # load config file (YAML)
    if args.config is not None:
        for path in args.config:
            with open(path) as f:
                config = yaml.full_load(f)
            dict_args.update(config)

    if args.seed is not None:
        # set random seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True  # NOTE: this will slow down training

    # [COMM] get MPI rank
    device, rank, world_size = pu.get_device_and_comm_rank(args.no_cuda,
                                                           args.distributed_backend)

    is_distributed = world_size > 1
    is_master = rank == 0

    train_loader, test_loader = pu.get_data_loader(
        args.dataset,
        args.batch_size,
        args.test_batch_size,
        dataset_root=args.dataset_root,
        train_root=args.train_root,
        val_root=args.val_root,
        download=args.download,
        n_workers=args.n_workers,
        world_size=world_size)

    # setup model
    arch_kwargs = args.arch_args
    model, args.arch_args = pu.get_model(args.arch, arch_kwargs,
                                         args.arch_file, args.dataset)
    model = model.to(device)

    # momentum of mini-batch gradients
    grads_momentum = None

    if args.optim not in ['ngd_ntk', 'cw_ngd', 'cw_ngd_eff']:
        dict_args.pop('kernel_batch_size')
    if args.optim not in ['ngd_cg', 'lw_ngd_cg', 'newton', 'ngd_ntk_cg']:
        dict_args.pop('cg_tol')

    if is_master:
        turn_off_wandb = dict_args.pop('turn_off_wandb')
        run_id = dict_args.pop('run_id')

        # initialize W&B as needed
        entries = [
            'epoch',
            'iteration',
            'train_loss',
            'test_loss',
            'train_acc@1',
            'test_acc@1',
            'learning_rate',
            'fisher_top_eig',
        ]
        if args.optim != 'sgd':
            entries.append('damping')
        logger = pu.Logger(args, entries=entries, turn_off_wandb=turn_off_wandb)
        if not is_distributed:
            # DDP disables hook functions that are required for
            # tracking torch graph
            logger.record_torch_graph(model)
        if run_id is None:
            run_id = logger.get_wandb_run_id()
        assert run_id is not None, \
            'run_id is not specified. Turn on W&B or specify --run_id.'

        print('================================')
        if is_distributed:
            print('Distributed training')
            print(f'world_size: {dist.get_world_size()}')
            print(f'backend: {dist.get_backend()}')
            print('---------------------------')
        for key, val in vars(args).items():
            if key == 'dataset':
                print(f'{key}: {val}')
                print(f'train set size: {len(train_loader.dataset)}')
                print(f'test set size: {len(test_loader.dataset)}')
            else:
                print(f'{key}: {val}')
        print(f'device: {device}')
        print('================================')
    else:
        logger = None

    main()
