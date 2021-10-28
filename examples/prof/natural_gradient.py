import argparse
import json
import yaml

import torch
from torch.nn.functional import cross_entropy
import torchvision

from asdfghjkl import KFAC, FISHER_EMP
from asdfghjkl import empirical_natural_gradient
from asdfghjkl import LBFGS
import profiling
import models

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size')
parser.add_argument('--input-size', type=str, default='32,32',
                    help='input size')
parser.add_argument('--optim', type=str, default='kfac',
                    help='name of the optimizer')
parser.add_argument('--arch', type=str,
                    help='name of the architecture')
parser.add_argument('--arch-args', type=json.loads, default={},
                    help='[JSON] arguments for the architecture')
parser.add_argument('--num-iters', type=int, default=100,
                    help='number of benchmark iterations')
parser.add_argument('--num-warmups', type=int, default=5,
                    help='number of warmup iterations')
parser.add_argument('--config', default=None, nargs='+',
                    help='config YAML file path')
# yapf: enable


def time_sgd():
    model = init_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = model(x)

    def bwd():
        loss.backward()

    def upd_param():
        optimizer.step()

    profiling.time_funcs([fwd, bwd, upd_param],
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_adam():
    model = init_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = model(x)

    def bwd():
        loss.backward()

    def upd_param():
        optimizer.step()

    profiling.time_funcs([fwd, bwd, upd_param],
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_kfac():
    model = init_model()
    ng = KFAC(model, FISHER_EMP)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    def upd_curv():
        ng.refresh_curvature(x, t, calc_emp_loss_grad=True)

    def upd_inv():
        ng.update_inv()

    def precond():
        ng.precondition()

    def upd_param():
        optimizer.step()

    profiling.time_funcs([upd_curv, upd_inv, precond, upd_param],
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_smw():
    model = init_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    def precond():
        empirical_natural_gradient(model, x, t, loss_fn=cross_entropy)

    def upd_param():
        optimizer.step()

    profiling.time_funcs([precond, upd_param],
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def time_lbfgs():
    model = init_model()
    hist_size = 20
    lbfgs = LBFGS(model.parameters(), max_hist_size=hist_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = model(x)

    def bwd():
        loss.backward()

    def upd_hist():
        lbfgs.update_history()

    def precondition():
        lbfgs.precondition()

    def upd_param():
        optimizer.step()

    fwd()
    bwd()
    # record histories for measuring time for precondition() with hist_size
    for _ in range(hist_size):
        upd_hist()

    profiling.time_funcs([fwd, bwd, upd_hist, precondition, upd_param],
                         num_iters=args.num_iters,
                         num_warmups=args.num_warmups)


def init_model():
    model = arch_cls(**args.arch_args)
    model.to(device)
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    dict_args = vars(args)

    # load config file (YAML)
    if args.config is not None:
        for path in args.config:
            with open(path) as f:
                config = yaml.full_load(f)
            dict_args.update(config)

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    # init model
    arch_cls = getattr(models, args.arch, None)
    if arch_cls is None:
        arch_cls = getattr(torchvision.models, args.arch)

    # prepare data
    input_size = [int(s) for s in args.input_size.split(',')]
    x = torch.rand(args.batch_size, *input_size).to(device)
    t = torch.tensor([0] * x.size(0)).long().to(device)

    if args.optim == 'kfac':
        time_kfac()


