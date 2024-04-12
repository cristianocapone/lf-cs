import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from argparse import Namespace
from argparse import ArgumentParser

root = '<path_to_data>'

def main(args : Namespace):
    fast = np.stack([
        np.load(f'{args.load_dir}/tau-fast_rep_{i}.npy')
        for i in range(args.n_reps)
    ])

    eprop = np.stack([
        np.load(f'{args.load_dir}/tau-slow_rep_{i}.npy')
        for i in range(args.n_reps)
    ])

    colors = [get_cmap('cool')(i) for i in np.linspace(0.25, 1, num=2)]

    fig, ax = plt.subplots(figsize=(3.5, 3))

    x = np.arange(len(fast[0])) * 50

    lines = []
    for data, c in zip(
        (fast, eprop),
        colors
    ):
        m, e = np.mean(data, axis=0), np.std(data, axis=0)
        ax.fill_between(x, m + e, m - e, color=c, alpha=0.25)
        ax.plot(x, data.T, c=c, alpha=0.25, lw=0.5)
        l, = ax.plot(x, m, c=c)

        lines.append(l)

    ax.legend(lines, ['lf-cs', 'e-prop'], frameon=False, loc=2)

    ax.set_xticks((0, 500, 1000, 1500, 2000, 2500))
    ax.set_xlabel('iteration')
    ax.set_ylabel('reward pong-100')

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 2500)
    ax.spines['left'].set_bounds(-2.0, 1.0)

    ax.spines[['bottom', 'left']].set_linewidth(1.25)

    ax.xaxis.set_tick_params(width=1.25)
    ax.yaxis.set_tick_params(width=1.25)

    fig.tight_layout()
    fig.savefig(f'{args.save_dir}/lf-cs-pong-100.png', dpi=300)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n_rep', required=False, type=int, default=5, help='Number of experiment repetitions')
    parser.add_argument('-load_dir', required=False, type=str, default='.', help='Directory where data are stored')
    parser.add_argument('-save_dir', required=False, type=str, default='.', help='Directory to save the figures')
    
    args = parser.parse_args()
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(args.save_dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(args.save_dir)
        print("The new directory is created!")
    
    main(args)