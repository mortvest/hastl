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
    seasonal_magn = stl.fit_1d(data, n_p=n_p, q_s=19, d_s=0)

    print(seasonal_magn)
