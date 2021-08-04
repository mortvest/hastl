"""
A small example of HaSTL usage. Apply STL to a generated harmonic time series
with missing values and plot the results
"""
import random
random.seed(100)

import numpy as np
import matplotlib.pyplot as plt

from hastl import STL


def add_nans(arr, nan_frac=0.1):
    assert 0 <= nan_frac < 1
    dim = arr.shape[0]
    n_nans = int(dim * nan_frac)
    idxs = random.sample(range(dim), n_nans)
    arr1 = np.copy(arr)
    arr1[idxs] = np.nan
    return arr1

def gen_harmonic_data(out_len=500, noise_level=0.05, trend_coeff=0.0005, period=2*np.pi, n_p=52, nans=True, nan_frac=0.1):
    n_repeats = int(out_len / n_p)
    x = np.linspace(0, period, n_p)

    x1 = np.tile(x, n_repeats + 1)[:out_len]
    x2 = np.repeat(np.arange(n_repeats + 1) * period, n_p)[:out_len]
    x1 += x2

    n_res = x1.shape[0]
    noise = np.random.normal(0, noise_level, n_res)
    trend = np.arange(0, n_res) * trend_coeff
    result = np.sin(x1) + trend + noise
    if nans:
        result = add_nans(result, nan_frac)
    result = np.round(result, 4)
    return result


def plot_single(ax, x, data, label):
    # ax.set_ylim(-2, 2)
    ax.set_xlim(-10, 510)
    ax.plot([], [])
    ax.scatter([], [])
    if label == "Remainder":
        ax.scatter(x, data, label="data")
    else:
        ax.plot(x, data, label="data")

    ax.set_title(label)
    ax.set_ylabel("y")
    # ax.legend()


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
    n_p = 52

    data = gen_harmonic_data(out_len=x_dim, n_p=n_p, nan_frac=0.01, trend_coeff=0.002).astype(np.float32)
    x = np.arange(1, x_dim + 1)

    stl = STL(debug=True, backend="c")
    seasonal, trend, remainder = stl.fit_1d(data, n_p=n_p)

    fig, axs = plt.subplots(4)
    labels = ["Input", "Seasonal", "Trend", "Remainder"]
    ds = [data, seasonal, trend, remainder]

    for ax, data, label in zip(axs, ds, labels):
        plot_single(ax, x, data, label)

    plt.xlabel("x")
    fig.set_size_inches(10, 10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.085)
    plt.savefig("stl1.png", dpi=150)
    # plt.show()
