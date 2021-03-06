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


def nan_ds(y):
    nan_idx = np.isnan(y)
    x = lambda z: z.nonzero()[0]
    y_i = np.interp(x(nan_idx), x(~nan_idx), y[~nan_idx])
    return x(nan_idx) + 1, y_i

def plot_single(ax, x, data, label):
    # ax.set_ylim(-2, 2)
    ax.set_xlim(-5, 505)
    ax.plot([], [])
    ax.scatter([], [])
    x_nan, y_nan = nan_ds(data)

    if label == "Remainder":
        ax.scatter(x, data, label="data", s=12)
    else:
        ax.plot(x, data, label="time series")
    if label == "Input":
        ax.scatter(x_nan, y_nan, label="missing values", c="black", marker="x")
        ax.legend()

    ax.set_title(label)
    ax.set_ylabel("y")


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
    n_p = 52

    data = gen_harmonic_data(out_len=x_dim, n_p=n_p, nan_frac=0.05, trend_coeff=0.002, noise_level=0.05).astype(np.float32)
    x = np.arange(1, x_dim + 1)

    stl = STL(debug=True, backend="c")
    seasonal, trend, remainder = stl.fit_1d(data, n_p=n_p, q_s=19, d_s=0)

    fig, axs = plt.subplots(4)
    labels = ["Input", "Seasonal", "Trend", "Remainder"]
    ds = [data, seasonal, trend, remainder]

    for ax, data, label in zip(axs, ds, labels):
        plot_single(ax, x, data, label)

    plt.xlabel("x")
    fig.set_size_inches(10, 7)

    plt.tight_layout()

    plt.subplots_adjust(left=0.08, bottom=0.07, right=0.98, top=0.96, wspace=None, hspace=0.4)
    # plt.subplots_adjust(wspace=0.085)
    # plt.subplots_adjust(wspace=-1)
    plt.savefig("stl1.pdf", dpi=150)
    # plt.show()
