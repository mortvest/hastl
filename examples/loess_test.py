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
    ax.set_title("q = {}".format(q))
    ax.set_ylabel("y")
    ax.legend()


if __name__ == "__main__":
    plt.style.use("bmh")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 12

    x_dim = 500
    n_p = 100

    data = gen_harmonic_data(out_len=x_dim, n_p=n_p, nans=False, trend_coeff=0).astype(np.float32)
    x = np.arange(1, x_dim + 1)

    loess = LOESS(backend="c")

    qs = [11, 31, 101, 3001]
    fig, axs = plt.subplots(len(qs))
    if type(axs) != np.ndarray:
        axs = np.array([axs])

    for (ax, q) in zip(axs, qs):
        plot_single(ax, data, q)
    plt.xlabel("x")

    fig.set_size_inches(10, 10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.085)
    plt.savefig("loess_qs.png", dpi=150)
    # plt.show()
