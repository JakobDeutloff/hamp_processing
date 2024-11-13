# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr
from src import readwrite_functions as rwfuncs
from src import load_data_functions as loadfuncs
from src.arts_functions import (
    run_arts,
    setup_workspace,
    extrapolate_dropsonde,
    get_profiles,
    average_double_bands,
    get_surface_temperature,
    get_surface_windspeed,
)
from src.plot_functions import plot_arts_flux
from src.dropsonde_processing import get_all_clouds_flags_dropsondes
from orcestra.postprocess.level0 import bahamas
import pyarts
import numpy as np
import typhon
import pandas as pd
import matplotlib.pyplot as plt

# %% set path to arts data
pyarts.cat.download.retrieve(verbose=True)
# %% read config
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
path_saveplts = cfg["path_saveplots"]
flightname = cfg["flightname"]

# %% read dropsonde data
ds_dropsonde = xr.open_dataset(cfg["path_dropsondes"], engine="zarr")
ds_dropsonde = ds_dropsonde.where(
    (ds_dropsonde["interp_time"] > pd.to_datetime(cfg["date"]))
    & (
        ds_dropsonde["interp_time"]
        < pd.to_datetime(cfg["date"]) + pd.DateOffset(hour=23)
    )
).dropna(dim="sonde_id", how="all")

# %% create HAMP post-processed data
hampdata = loadfuncs.load_hamp_data(
    cfg["path_radar"], cfg["path_radiometers"], cfg["path_iwv"]
)

# %% read bahamas raw
ds_bahamas = xr.open_dataset(
    f'Data/Bahamas_Raw/QL_{cfg["flightname"]}_BAHAMAS_V01.nc'
).pipe(bahamas)

# %% define frequencies
freq_k = [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]
freq_v = [50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]
freq_90 = [90.0]
center_freq_119 = 118.75
center_freq_183 = 183.31
width_119 = [1.4, 2.3, 4.2, 8.5]
width_183 = [0.6, 1.5, 2.5, 3.5, 5.0, 7.5]
freq_119 = [center_freq_119 - w for w in width_119] + [
    center_freq_119 + w for w in width_119
]
freq_183 = [center_freq_183 - w for w in width_183] + [
    center_freq_183 + w for w in width_183
]
all_freqs = freq_k + freq_v + freq_90 + freq_119 + freq_183
all_freqs = np.sort(all_freqs)

# %% check if dropsondes are cloud free
ds_dropsonde = get_all_clouds_flags_dropsondes(ds_dropsonde)
cloud_free_idxs = (
    ds_dropsonde["sonde_id"].where(ds_dropsonde["cloud_flag"] == 0, drop=True).values
)

# %% cloud mask from radar
ds_dropsonde = ds_dropsonde.assign(
    radar_cloud_flag=(
        hampdata.radar.sel(time=ds_dropsonde.launch_time, method="nearest").sel(
            height=slice(200, None)
        )["dBZe"]
        > -30
    ).max("height")
    * 1
)
cloud_free_idxs_radar = (
    ds_dropsonde["sonde_id"]
    .where(ds_dropsonde["radar_cloud_flag"] == 0, drop=True)
    .values
)

# %% control plot of radar and cloud masks
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
hampdata.radar["dBZe"].where(hampdata.radar["dBZe"] > -30).plot.pcolormesh(
    ax=ax, x="time", y="height", cmap="viridis"
)

for id in ds_dropsonde.sonde_id:
    time = ds_dropsonde.sel(sonde_id=id).launch_time.values

    if ds_dropsonde.sel(sonde_id=id).cloud_flag == 1:
        ax.axvline(time, color="red")
    else:
        ax.axvline(time, color="green")

    if ds_dropsonde.sel(sonde_id=id).radar_cloud_flag == 1:
        ax.axvline(time, color="red", linestyle="-.")
    elif ds_dropsonde.sel(sonde_id=id).radar_cloud_flag == 0:
        ax.axvline(time, color="green", linestyle="-.")

plt.show()


# %% setup container dataframes to store data from all cloud free sondes
freqs_hamp = hampdata.radiometers.frequency.values

# %% get profiles
sonde_id = cloud_free_idxs[0]
ds_dropsonde_loc, hampdata_loc, height, drop_time = get_profiles(
    sonde_id, ds_dropsonde, hampdata
)

# check if dropsonde is broken (contains only nan values)
if ds_dropsonde_loc["ta"].isnull().mean().values == 1:
    print(f"Dropsonde {sonde_id} is broken, skipping")

#  get surface values
surface_temp = get_surface_temperature(ds_dropsonde_loc)
surface_ws = get_surface_windspeed(ds_dropsonde_loc)

# extrapolate dropsonde profiles
ds_dropsonde_extrap = extrapolate_dropsonde(ds_dropsonde_loc, height, ds_bahamas)

# %% plot dropsonde profiles
fig, axes = plt.subplots(1, 3, figsize=(10, 7), sharey=True)

axes[0].scatter(
    ds_bahamas["TS"].sel(time=drop_time, method="nearest"),
    height,
    marker="x",
    color="k",
)
ds_dropsonde_extrap["ta"].plot(y="alt", ax=axes[0])
ds_dropsonde_loc["ta"].plot(y="alt", ax=axes[0], linestyle="--")

axes[1].scatter(
    ds_bahamas["PS"].sel(time=drop_time, method="nearest") * 100,
    height,
    marker="x",
    color="k",
)
ds_dropsonde_extrap["p"].plot(y="alt", ax=axes[1])
ds_dropsonde_loc["p"].plot(y="alt", ax=axes[1], linestyle="--")

axes[2].scatter(
    ds_bahamas["MIXRATIOV"].sel(time=drop_time, method="nearest") / 1e6,
    height,
    marker="x",
    color="k",
)
typhon.physics.specific_humidity2vmr(ds_dropsonde_extrap["q"]).plot(y="alt", ax=axes[2])
typhon.physics.specific_humidity2vmr(ds_dropsonde_loc["q"]).plot(
    y="alt", ax=axes[2], linestyle="--"
)

plt.show()


# %% setup workspace
ws = setup_workspace()

# %% run arts
run_arts(
    pressure_profile=ds_dropsonde_extrap["p"].values,
    temperature_profile=ds_dropsonde_extrap["ta"].values,
    h2o_profile=typhon.physics.specific_humidity2vmr(ds_dropsonde_extrap["q"].values),
    surface_ws=surface_ws,
    surface_temp=surface_temp,
    ws=ws,
    frequencies=all_freqs * 1e9,
    zenith_angle=180,
    height=height,
)

TB_arts = pd.DataFrame(
    data=np.array(ws.y.value), index=np.float32(np.array(ws.f_grid.value) / 1e9)
)

# get according hamp data
TB_hamp = pd.DataFrame(data=hampdata_loc.radiometers.TBs.values, index=freqs_hamp)

# average double bands
TB_arts_av = average_double_bands(
    TB_arts,
    freqs_hamp,
)

# %% Plot to compare arts to hamp radiometers
fig, ax = plot_arts_flux(TB_hamp, TB_arts_av, dropsonde_id=sonde_id, time=drop_time)
plt.show()
# %%
