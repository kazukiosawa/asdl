import argparse
import json
import yaml

import torch
from torch.nn.functional import mse_loss
import torchvision

from asdfghjkl import KfacGradientMaker, FISHER_EMP, LOSS_MSE
from asdfghjkl import empirical_natural_gradient
from asdfghjkl import LBFGS
from asdfghjkl.precondition import Shampoo
import profiling
import models


# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=str, default='32,32',
                    help='input size')
parser.add_argument('--optim', type=str, default='kfac',
                    help='name of the optimizer')
parser.add_argument('--arch', type=str,
                    help='name of the architecture')
parser.add_argument('--arch_args', type=json.loads, default={},
                    help='[JSON] arguments for the architecture')
parser.add_argument('--num_blocks', type=int, default=None)
parser.add_argument('--width_scale', type=int, default=None)
parser.add_argument('--num_iters', type=int, default=100,
                    help='number of benchmark iterations')
parser.add_argument('--num_warmups', type=int, default=5,
                    help='number of warmup iterations')
parser.add_argument('--config', default=None, nargs='+',
                    help='config YAML file path')
parser.add_argument('--decoder', action='store_true')
parser.add_argument('--emit_nvtx', action='store_true')
parser.add_argument('--wandb', action='store_true',
                    help='If True, record summary to W&B')
# yapf: enable


def time_sgd():
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = mse_loss(call_forward(), t)

    def bwd():
        loss.backward()

    def upd_param():
        optimizer.step()

    profiling.time_funcs([fwd, bwd, upd_param],
                         emit_nvtx=args.emit_nvtx,
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_adam():
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = mse_loss(call_forward(), t)

    def bwd():
        loss.backward()

    def upd_param():
        optimizer.step()

    profiling.time_funcs([fwd, bwd, upd_param],
                         emit_nvtx=args.emit_nvtx,
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_shampoo():
    optimizer = Shampoo(model.parameters())
    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = mse_loss(call_forward(), t)

    def bwd():
        loss.backward()

    def upd_param():
        optimizer.step()

    profiling.time_funcs([fwd, bwd, upd_param],
                         emit_nvtx=args.emit_nvtx,
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_kfac():
    ng = KfacGradientMaker(model, FISHER_EMP, loss_type=LOSS_MSE, damping=1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    def fwd_bwd_upd_curv():
        ng.refresh_curvature(x, t, calc_emp_loss_grad=True)

    def upd_inv():
        ng.update_preconditioner()

    def precond():
        ng.precondition()

    def upd_param():
        optimizer.step()

    profiling.time_funcs([fwd_bwd_upd_curv, upd_inv, precond, upd_param],
                         emit_nvtx=args.emit_nvtx,
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_smw():
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    def fwd_bwd_precond():
        empirical_natural_gradient(model, x, t, loss_fn=mse_loss)

    def upd_param():
        optimizer.step()

    profiling.time_funcs([fwd_bwd_precond, upd_param],
                         emit_nvtx=args.emit_nvtx,
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_lbfgs():
    hist_size = 20
    lbfgs = LBFGS(model.parameters(), max_hist_size=hist_size, rho_min=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = mse_loss(call_forward(), t)

    def bwd():
        loss.backward()

    def upd_hist():
        lbfgs.update_history()

    def precond():
        lbfgs.precondition()

    def upd_param():
        optimizer.step()

    fwd()
    bwd()
    # record histories for measuring time for precondition() with hist_size
    for _ in range(hist_size):
        upd_hist()

    profiling.time_funcs([fwd, bwd, upd_hist, precond, upd_param],
                         emit_nvtx=args.emit_nvtx,
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def call_forward():
    if args.decoder:
        return model(x, memory)
    else:
        return model(x)


if __name__ == '__main__':
    args = parser.parse_args()
    dict_args = vars(args)

    # load config file (YAML)
    if args.config is not None:
        for path in args.config:
            with open(path) as f:
                config = yaml.full_load(f)
            dict_args.update(config)

    for key in ['num_blocks', 'width_scale']:
        if dict_args[key] is not None:
            args.arch_args[key] = dict_args.pop(key)

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    # init model
    arch_cls = getattr(models, args.arch, None)
    if arch_cls is None:
        arch_cls = getattr(torchvision.models, args.arch)
    model = arch_cls(**args.arch_args)
    model.to(device)

    # prepare data
    input_size = [int(s) for s in args.input_size.split(',')]
    x = torch.rand(*input_size).to(device)
    memory = torch.rand(*input_size).to(device)
    y = call_forward()
    t = torch.rand(y.shape).to(device)

    torch.cuda.reset_peak_memory_stats()

    if args.optim == 'sgd':
        time_sgd()
    elif args.optim == 'adam':
        time_adam()
    elif args.optim == 'shampoo':
        time_shampoo()
    elif args.optim == 'kfac':
        time_kfac()
    elif args.optim == 'smw':
        time_smw()
    elif args.optim == 'lbfgs':
        time_lbfgs()

    max_memory = torch.cuda.max_memory_allocated()
    print('max memory allocated: ', max_memory)
    if args.wandb:
        import wandb
        wandb.init(config=dict_args)
        summary = {'max_memory_allocated': max_memory,
                   'num_params': sum(p.numel() for p in model.parameters())}
        wandb.summary.update(summary)

