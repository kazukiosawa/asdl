import argparse
import json
import yaml

import torch
from torch.nn.functional import cross_entropy
import torchvision

from asdfghjkl import KFAC, FISHER_EMP
import profiling
import models

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size')
parser.add_argument('--input-size', type=str, default='32,32',
                    help='input size')
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


def time_kfac(x,
              model,
              fisher_type=FISHER_EMP,
              name='KFAC',
              num_iters=100,
              num_warmups=5):
    target = torch.tensor([0] * x.size(0)).long().to(device)
    ng = KFAC(model, fisher_type)

    loss = torch.Tensor().to(device)

    def fwd():
        nonlocal loss
        loss = cross_entropy(model(x), target)

    def upd_curv():
        ng.update_curvature(x, target)

    def acc_curv():
        ng.accumulate_curvature()

    def upd_inv():
        ng.finalize_accumulation()
        ng.update_inv()

    def bwd():
        model.zero_grad(set_to_none=True)
        loss.backward()

    def acc_grad():
        for p in model.parameters():
            _ = p.grad.data + p.grad.data

    def precond():
        ng.precondition()

    profiling.time_funcs(
        [fwd, upd_curv, acc_curv, upd_inv, bwd, acc_grad, precond],
        name=name,
        num_iters=num_iters,
        num_warmups=num_warmups)


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
    model = arch_cls(**args.arch_args)
    model.to(device)

    # prepare an input
    input_size = [int(s) for s in args.input_size.split(',')]
    x = torch.rand(args.batch_size, *input_size).to(device)

    time_kfac(x, model, num_iters=args.num_iters, num_warmups=args.num_warmups)
