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


if __name__ == "__main__":
    x_dim = 500
    n_p = 52

    data = gen_harmonic_data(out_len=x_dim, n_p=n_p).astype(np.float32)
    x = np.arange(1, x_dim + 1)

    stl = STL(debug=True, backend="opencl")
    seasonal, trend, remainder = stl.fit_1d(data, n_p=n_p)

    fig, axs = plt.subplots(4)
    axs[0].plot(x, data)
    axs[0].set_title("Input")

    axs[1].plot(x, seasonal)
    axs[1].set_title("Seasonal")

    axs[2].plot(x, trend)
    axs[2].set_title("Trend")

    axs[3].plot(x, remainder)
    axs[3].set_title("Remainder")
    plt.show()
