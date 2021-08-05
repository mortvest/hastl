"""
A small example of HaSTL usage. Apply STL to a generated harmonic time series
with missing values and plot the results
"""
import random
# random.seed(100)
random.seed(101)

import numpy as np
import matplotlib.pyplot as plt

from hastl import STL
from utils import *


def plot_single(ax, x, data, label):
    # ax.set_ylim(-2, 2)
    ax.set_xlim(-5, 505)
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

    data = gen_harmonic_data(out_len=x_dim, n_p=n_p, nan_frac=0.01, trend_coeff=0.002, noise_level=0.05).astype(np.float32)
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
    # plt.savefig("stl1.png", dpi=150)
    plt.show()
