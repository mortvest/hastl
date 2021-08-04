import random
random.seed(100)

import numpy as np
import matplotlib.pyplot as plt

from hastl import LOESS


def add_nans(arr, nan_frac=0.1):
    assert 0 <= nan_frac < 1
    dim = arr.shape[0]
    n_nans = int(dim * nan_frac)
    idxs = random.sample(range(dim), n_nans)
    arr1 = np.copy(arr)
    arr1[idxs] = np.nan
    return arr1

def gen_harmonic_data(out_len=500, noise_level=0.1, trend_coeff=0.00005, period=2*np.pi, n_p=52, nans=True, nan_frac=0.1):
    n_repeats = int(out_len / n_p)
    x = np.linspace(0, period, n_p)

    x1 = np.tile(x, n_repeats + 1)[:out_len]
    x2 = np.repeat(np.arange(n_repeats + 1) * period, n_p)[:out_len]
    x1 += x2

    n_res = x1.shape[0]
    noise = np.random.normal(0, noise_level, n_res)
    trend = np.arange(0, n_res) * trend_coeff
    # result = np.sin(x1) + trend + noise
    result = np.sin(x1) + noise
    if nans:
        result = add_nans(result, nan_frac)
    result = np.round(result, 4)
    return result


def plot_single(ax, data, q):
    result = loess.fit_1d(data, q=q)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-10, 510)
    ax.plot(x, data, label="data")
    ax.plot(x, result, label="LOESS")
    ax.set_title("q = {}".format(q))
    ax.set_ylabel("y")
    ax.legend()


if __name__ == "__main__":
    plt.style.use("bmh")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    x_dim = 500
    n_p = 100

    data = gen_harmonic_data(out_len=x_dim, n_p=n_p, nans=False).astype(np.float32)
    x = np.arange(1, x_dim + 1)

    loess = LOESS(debug=True, backend="c")

    # q = 3001
    # result = loess.fit_1d(data, q=q)

    fig, axs = plt.subplots(4)

    qs = [11, 51, 101, 3001]

    for (ax, q) in zip(axs, qs):
        plot_single(ax, data, q)
    plt.xlabel("x")

    fig.set_size_inches(10, 10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.085)
    plt.savefig("loess_qs.png", dpi=150)

    # plt.show()

