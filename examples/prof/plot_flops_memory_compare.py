import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

import torch
import torchvision
from asdfghjkl.counter import *
import models

_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

optim_colors = {
    'SGD': _colors[0],
    'Adam': _colors[1],
    'Shampoo': _colors[2],
    'K-FAC (emp)': _colors[3],
    'L-BFGS (m=20)': _colors[4],
    'SMW-NG': _colors[5],
                }

parser = argparse.ArgumentParser()
parser.add_argument('--archs', type=str, default='resnet50,resnet101')
parser.add_argument('--run_forward', action='store_true')
parser.add_argument('--input_size', type=str, default='2,3,256,256')
parser.add_argument('--fig_path', type=str, default='flops_memory.png')
parser.add_argument('--batch_sizes', type=str, default='32,128')
parser.add_argument('--seq_len', type=int, default=128)


def plot(ax, all_counts, optim, funcs, title=None, ylabel=True, legend=True, plot_flop=True):
    if ylabel:
        if plot_flop:
            ax.set_ylabel(f'TFLOPs')
        else:
            ax.set_ylabel(f'Memory (GB)')
    if title is not None:
        ax.set_title(title)

    xs = [f'bs:{bs}' for bs in batch_sizes]
    ys = []
    for bs in batch_sizes:
        counts = all_counts[bs]
        y = 0
        for func_name, counter in funcs.items():
            if plot_flop:
                y += counts[counter.__name__]['flop'] / float(1 << 40)  # TFlop
            else:
                y += counts[counter.__name__]['numel'] * 4 / float(1 << 30)  # fp32, GB
        ys.append(y)
    ax.plot(xs, ys, label=optim, color=optim_colors[optim], marker='o')
    ax.set_yscale('log')
    ax.grid('--')

    if legend:
        ax.legend(loc='upper left')


def main():
    optim_funcs = {
        'SGD': {'fwd': Forward,
                'bwd_weight': BackwardWeight,
                'bwd_input': BackwardInput},
        'Adam': {'fwd': Forward,
                 'bwd_weight': BackwardWeight,
                 'bwd_input': BackwardInput,
                 'stats': AdamStats,
                 'precond': AdamPrecond},
        'Shampoo': {'fwd': Forward,
                    'bwd_weight': BackwardWeight,
                    'bwd_input': BackwardInput,
                    'stats': ShampooStats,
                    'inv': ShampooInv,
                    'precond': ShampooPrecond},
        'K-FAC (emp)': {'fwd': Forward,
                        'bwd_weight': BackwardWeight,
                        'bwd_input': BackwardInput,
                        'stats': KFACStats,
                        'inv': KFACInv,
                        'precond': KFACPrecond},
        'L-BFGS (m=20)': {'fwd': Forward,
                          'bwd_weight': BackwardWeight,
                          'bwd_input': BackwardInput,
                          'precond': LBFGSPrecond},
        'SMW-NG': {'fwd': Forward,
                   'bwd_weight': BackwardWeight,
                   'bwd_input': BackwardInput,
                   'stats': SMWNGStats,
                   'precond': SMWNGPrecond},
    }
    fig = plt.figure(figsize=(len(archs) * 15, 15))
    gs = fig.add_gridspec(2, len(archs))

    counters = [Forward(), BackwardWeight(), BackwardInput(),
                AdamStats(), AdamPrecond(),
                ShampooStats(), ShampooInv(), ShampooPrecond(),
                KFACStats(), KFACInv(), KFACPrecond(),
                LBFGSPrecond(hist_size=20), SMWNGStats(), SMWNGPrecond()]

    all_counts = {}
    for arch in archs:
        all_counts[arch] = {}
        arch_class = getattr(torchvision.models, arch, None)
        if arch_class is None:
            arch_class = getattr(models, arch)
        model = arch_class()
        for bs in batch_sizes:
            print(f'{arch} bs:{bs}')
            with set_counter(model, counters, batch_size=bs, seq_len=args.seq_len) as counts:
                if args.run_forward:
                    x = torch.randn(*input_size)
                    model(x)
                all_counts[arch][bs] = counts

    for i, plot_flop in enumerate([True, False]):
        axes = []
        for j, arch in enumerate(archs):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
            for optim in optim_funcs.keys():
                plot(ax, all_counts[arch], optim, optim_funcs[optim],
                     title=arch, plot_flop=plot_flop)

        top_max = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(top=top_max)

    plt.tight_layout()
    plt.savefig(args.fig_path, bbox_inches='tight')


if __name__ == '__main__':
    args = parser.parse_args()
    archs = args.archs.split(',')
    input_size = [int(s) for s in args.input_size.split(',')]
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    main()
