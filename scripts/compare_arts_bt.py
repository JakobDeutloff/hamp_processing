# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr
from src import readwrite_functions as rwfuncs
from src import load_data_functions as loadfuncs
from src.arts_functions import run_arts, setup_workspace
import pyarts
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %% set path to arts data
pyarts.cat.download.retrieve(verbose=True)
# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
path_saveplts = cfg["path_saveplts"]
flightname = cfg["flightname"]

# %% read dropsonde data
ds_dropsonde = xr.open_mfdataset(str(cfg["path_dropsondes_level3"])).load()

# %% create HAMP post-processed data
hampdata = loadfuncs.do_post_processing(
    cfg["path_bahamas"],
    cfg["path_radar"],
    cfg["path_radiometer"],
    cfg["radiometer_date"],
    is_planet=cfg["is_planet"],
    do_radar=True,
    do_183=True,
    do_11990=True,
    do_kv=True,
    do_cwv=True,
)
# %% define frequencies
freq_k = [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]
freq_v = [50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]
freq_90 = [90.0]
freq_119 = [118.75]
freq_183 = [183.31]
width_119 = [1.4, 2.3, 4.2, 8.5]
width_183 = [0.6, 1.5, 2.5, 3.5, 5.0, 7.5]

all_freqs = freq_k + freq_v + freq_90 + freq_119 + freq_183


# %% find hamp data at dropsonde location
ds_dropsonde_loc = ds_dropsonde.isel(sonde_id=28)
drop_time = ds_dropsonde_loc["interpolated_time"].dropna("gpsalt").min().values
radar_loc = hampdata.radar.sel(time=drop_time, method="nearest")
radio_183_loc = hampdata.radio183.sel(time=drop_time, method="nearest")
radio_11990_loc = hampdata.radio11990.sel(time=drop_time, method="nearest")
radio_kv_loc = hampdata.radiokv.sel(time=drop_time, method="nearest")
bahamas_loc = hampdata.flightdata.sel(time=drop_time, method="nearest")
height = float(bahamas_loc["IRS_ALT"])

# %% cut dropsonde to flightlevel
ds_dropsonde_cut = ds_dropsonde_loc.where(
    ds_dropsonde_loc["gpsalt"] < height, drop=True
)

# %% extrapolate dropsonde data to flightlevel


def exponential(x, a, b):
    return a * np.exp(b * x)


def linear(x, a, b):
    return a * x + b


def fit_exponential(x, y, p0):
    valid_idx = (~np.isnan(y)) & (~np.isnan(x))
    x = x[valid_idx]
    y = y[valid_idx]
    popt, pcov = curve_fit(exponential, x, y, p0=p0)
    return popt


def fit_linear(x, y):
    valid_idx = (~np.isnan(y)) & (~np.isnan(x))
    x = x[valid_idx]
    y = y[valid_idx]
    popt, pcov = curve_fit(linear, x, y)
    return popt


def fill_upper_levels(x, y, popt, func):
    offset = y.dropna("gpsalt").values[-1]
    nan_vals = np.isnan(y)
    idx_nan = np.where(nan_vals)[0]
    nan_vals[idx_nan - 1] = True  # get overlap of one
    filled = np.zeros_like(y)
    filled[~nan_vals] = y[~nan_vals]
    new_vals = func(x[nan_vals], *popt)
    filled[nan_vals] = new_vals - new_vals[0] + offset
    return filled


popt_p = fit_exponential(
    ds_dropsonde_cut["gpsalt"].values, ds_dropsonde_cut["p"].values, p0=[100000, -1]
)
popt_ta = fit_linear(ds_dropsonde_cut["gpsalt"].values, ds_dropsonde_cut["ta"].values)
popt_q = fit_exponential(
    ds_dropsonde_cut["gpsalt"].values, ds_dropsonde_cut["q"].values, p0=[0.02, -0.1]
)
p_extrap = fill_upper_levels(
    ds_dropsonde_cut["gpsalt"].values,
    ds_dropsonde_cut["p"].interpolate_na("gpsalt"),
    popt_p,
    exponential,
)
ta_extrap = fill_upper_levels(
    ds_dropsonde_cut["gpsalt"].values,
    ds_dropsonde_cut["ta"].interpolate_na("gpsalt"),
    popt_ta,
    linear,
)
q_extrap = ds_dropsonde_cut["q"].interpolate_na("gpsalt").values
q_extrap[np.isnan(q_extrap)] = q_extrap[~np.isnan(q_extrap)][-1]
# %% control plots
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(ds_dropsonde_cut["gpsalt"], ds_dropsonde_cut["q"], label="Pressure")
ax.plot(
    ds_dropsonde_cut["gpsalt"], q_extrap, label="Extrapolated Pressure", linestyle="--"
)

# %% setup arts workspace
ws = setup_workspace()

# %% run arts
run_arts(
    pressure_profile=p_extrap,
    temperature_profile=ta_extrap,
    h2o_profile=q_extrap,
    ws=ws,
    fnum=100,
    frequencies=np.array(all_freqs) * 1e9,
    zenith_angle=180,
    height=height,
)
# %% compare to hamp radiometers
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(ws.f_grid.value / 1e9, ws.y.value, label="ARTS", marker="x")
ax.scatter(
    radio_11990_loc["frequency"].values,
    radio_11990_loc["TBs"].values,
    label="HAMP",
    color="red",
)
ax.scatter(
    radio_kv_loc["frequency"].values,
    radio_kv_loc["TBs"].values,
    label="HAMP",
    color="red",
)
ax.scatter(
    radio_183_loc["frequency"].values,
    radio_183_loc["TBs"].values,
    label="HAMP",
    color="red",
)
ax.set_xlabel("Frequency / GHz")
ax.set_ylabel("Brightness Temperature / K")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
# %%
