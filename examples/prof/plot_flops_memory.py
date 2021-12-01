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

func_colors = {'fwd': _colors[0],
               'bwd_weight': _colors[1],
               'bwd_input': _colors[2],
               'stats': _colors[3],
               'inv': _colors[4],
               'precond': _colors[5],
               }

optim_colors = {'Adam': _colors[0],
                'Shampoo': _colors[1],
                'K-FAC (emp)': _colors[2],
                'L-BFGS (m=20)': _colors[3],
                'SMW-NG': _colors[4],
                }

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, choices=['flop', 'memory'])
parser.add_argument('--archs', type=str, default='resnet50,resnet101')
parser.add_argument('--run_forward', action='store_true')
parser.add_argument('--input_size', type=str, default='2,3,256,256')
parser.add_argument('--fig_path', type=str, default='flops_memory.png')
parser.add_argument('--batch_sizes', type=str, default='32,128')
parser.add_argument('--seq_len', type=int, default=128)


def plot_bar(ax, all_counts, optim, funcs, title=None, ylabel=True, legend=True, plot_flop=True):
    if ylabel:
        if plot_flop:
            ax.set_ylabel(f'TFLOPs')
        else:
            ax.set_ylabel('Memory (GB)')
    if title is not None:
        ax.set_title(title, color=optim_colors[optim])

    for idx, bs in enumerate(batch_sizes):
        counts = all_counts[bs]
        bottom = 0

        if not plot_flop:
            # byte count (params)
            y = counts['params'] * 4 / float(1 << 30)  # fp32, GB
            label = 'params' if idx == 0 else None
            ax.bar(idx, y, label=label, color='gray')
            bottom += y

        for func_name, counter in funcs.items():
            label = func_name if idx == 0 else None

            if plot_flop:
                # flop count
                y = counts[counter.__name__]['flop'] / float(1 << 40)  # TFlop
                ax.bar(idx, y, bottom=bottom, label=label, color=func_colors[func_name])
                bottom += y
            else:
                # byte count
                y = counts[counter.__name__]['numel'] * 4 / float(1 << 30)  # fp32, GB
                if y > 0:
                    ax.bar(idx, y, bottom=bottom, label=label, color=func_colors[func_name])
                    bottom += y

    ax.set_xticks(range(len(batch_sizes)))
    x_labels = [f'bs:{batch_size}' for batch_size in batch_sizes]
    ax.set_xticklabels(x_labels)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left')


def main():
    optim_funcs = {
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
    fig = plt.figure(figsize=(len(archs) * 15, len(optim_funcs) * 5))
    gs = fig.add_gridspec(len(optim_funcs), len(archs))

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

    for i, optim in enumerate(optim_funcs.keys()):
        axes = []
        for j, arch in enumerate(archs):
            ax = fig.add_subplot(gs[i, j])
            plot_bar(ax, all_counts[arch], optim, optim_funcs[optim],
                     title=f'{optim} {arch}', plot_flop=args.target == 'flop')
            axes.append(ax)

#        top_max = max(ax.get_ylim()[1] for ax in axes)
#        for ax in axes:
#            ax.set_ylim(top=top_max, bottom=0)

    plt.tight_layout()
    plt.savefig(args.fig_path, bbox_inches='tight')


if __name__ == '__main__':
    args = parser.parse_args()
    archs = args.archs.split(',')
    input_size = [int(s) for s in args.input_size.split(',')]
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    main()
