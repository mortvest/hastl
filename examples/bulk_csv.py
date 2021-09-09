"""
A bigger example of HaSTL usage. Download and unzip three .csv files, then apply
STL to three columns of each file and save the results
"""
import itertools
import os
import wget
from zipfile import ZipFile

import numpy as np
import pandas as pd

from hastl import STL


def process_file(stl_obj, input_name, output_name, n_p):
    print("processing", output_name)
    data_df = pd.read_csv(input_name)
    data = data_df[["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]].to_numpy()
    data = data.astype(np.float32)
    data[np.isclose(data, 0)] = np.nan

    print("running stl")
    seasonal, trend, remainder = stl_obj.fit(data.T, n_p = n_p, s_window=999)
    seasonal = np.nan_to_num(seasonal)
    trend = np.nan_to_num(trend)
    remainder = np.nan_to_num(remainder)

    print("creating dataframe")
    data[np.isnan(data)] = 0
    res = pd.DataFrame({
        "dt": data_df["dt"],
        "Global_active_power_sesonal": seasonal[0,:],
        "Global_active_power_trend": trend[0,:],
        "Global_active_power_remainder": remainder[0,:],
        "Global_reactive_power_seasonal": seasonal[1,:],
        "Global_reactive_power_trend": trend[1,:],
        "Global_reactive_power_remainder": remainder[1,:],
        "Voltage_seasonal": seasonal[2,:],
        "Voltage_trend": trend[2,:],
        "Voltage_remainder": remainder[2,:],
        "Global_intensity_seasonal": seasonal[3, :],
        "Global_intensity_trend": trend[3, :],
        "Global_intensity_remainder": remainder[3, :],
        "Sub_metering_1": data_df["Sub_metering_1"],
        "Sub_metering_2": data_df["Sub_metering_2"],
        "Sub_metering_3": data_df["Sub_metering_3"]
        })
    print("saving csv")
    res.to_csv(output_name, index=False)

# download and parse input data
data_dir = "data/"
data_dir_p = data_dir + "power/"

test_file = data_dir_p + "test.csv"
train_file = data_dir_p + "train.csv"
val_file = data_dir_p + "val.csv"

archive_name = "power.zip"
archive_path = data_dir + archive_name

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(archive_path):
    print("downloading files")
    url = "https://sid.erda.dk/share_redirect/Gj7j1ZAFo9"
    wget.download(url, archive_path)

print()
if not (os.path.exists(test_file) and os.path.exists(train_file) and os.path.exists(val_file)):
    print("extracting zip")
    with ZipFile(archive_path, 'r') as zip_archive:
        zip_archive.extractall(data_dir)

print("initializing the environment")
stl_obj = STL(backend="opencl")

datasets = [train_file, test_file, val_file]
out_names = ["train", "test", "val"]

n_ps = [60 * 24, 60 * 24 * 7]
periodicities = ["daily", "weekly"]

for (dataset, out_name), (n_p, periodicity) in itertools.product(zip(datasets, out_names), zip(n_ps, periodicities)):
    process_file(stl_obj, dataset, "{}_decomp_{}.csv".format(out_name, periodicity), n_p)
