import numpy as np
import matplotlib.pyplot as plt

from hastl import LOESS
from utils import *


def plot_single(ax, data, q):
    result = loess.fit_1d(data, q=q, jump=1)
    # ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-5, 505)
    ax.plot(x, data, label="data")
    ax.plot(x, result, label="LOESS")
    ax.set_title(r"$q = {}$".format(q))
    ax.set_ylabel("y")
    ax.legend(loc="lower left")


if __name__ == "__main__":
    plt.style.use("bmh")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14

    x_dim = 500
    n_p = 100

    data = gen_harmonic_data(out_len=x_dim, n_p=n_p, nans=False, trend_coeff=0).astype(np.float32)
    x = np.arange(1, x_dim + 1)

    loess = LOESS(backend="c")

    # qs = [11, 31, 101, 3001]
    qs = [11, 101, 1001]
    fig, axs = plt.subplots(len(qs))
    if type(axs) != np.ndarray:
        axs = np.array([axs])

    for (ax, q) in zip(axs, qs):
        plot_single(ax, data, q)
    plt.xlabel("x")

    # fig.set_size_inches(10, 10)
    # fig.set_size_inches(10, 7)
    fig.set_size_inches(10, 4)

    plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.12, right=0.99, top=0.93, wspace=None, hspace=0.6)
    plt.savefig("loess1.pdf", dpi=150)
    # plt.show()
