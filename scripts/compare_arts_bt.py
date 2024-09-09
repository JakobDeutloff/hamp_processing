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
)
from src.plot_functions import plot_arts_flux, get_hamp_TBs
from src.dropsonde_processing import get_all_clouds_flags_dropsondes
import pyarts
import numpy as np
import typhon
import pandas as pd

# %% set path to arts data
pyarts.cat.download.retrieve(verbose=True)
# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
path_saveplts = cfg["path_saveplts"]
flightname = cfg["flightname"]

# %% read dropsonde data
ds_dropsonde = xr.open_mfdataset(str(cfg["path_dropsonde_level3"])).load()
ds_dropsonde = ds_dropsonde.where(
    (ds_dropsonde["interp_time"] > pd.to_datetime(cfg["date"]))
    & (
        ds_dropsonde["interp_time"]
        < pd.to_datetime(cfg["date"]) + pd.DateOffset(hour=23)
    )
).dropna(dim="sonde_id", how="all")

# %% create HAMP post-processed data
hampdata = loadfuncs.do_post_processing(
    cfg["path_bahamas"],
    cfg["path_radar"],
    cfg["path_radiometer"],
    cfg["radiometer_date"],
    cfg["path_sea_land_mask"],
    is_planet=cfg["is_planet"],
    do_radar=False,
    do_183=True,
    do_11990=True,
    do_kv=True,
    do_cwv=False,
)
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

# %% setup workspace
ws = setup_workspace()

# %% loop over cloud free sondes
freqs_hamp, _ = get_hamp_TBs(
    hampdata.sel(timeslice=ds_dropsonde["interp_time"].values[0])
)
TBs_arts = pd.DataFrame(index=freqs_hamp, columns=cloud_free_idxs)
TBs_hamp = TBs_arts.copy()
dropsondes_extrap = []

for i, sonde_id in enumerate(cloud_free_idxs):
    # get profiles
    ds_dropsonde_loc, hampdata_loc, height, drop_time = get_profiles(
        sonde_id, ds_dropsonde, hampdata
    )
    # check if dropsonde is broken (contains only nan values)
    if ds_dropsonde_loc["ta"].isnull().mean().values == 1:
        print(f"Dropsonde {sonde_id} is broken, skipping")
        continue

    # extrapolate dropsonde data
    ds_dropsonde_extrap = extrapolate_dropsonde(ds_dropsonde_loc, height)
    dropsondes_extrap.append(ds_dropsonde_extrap)

    # run arts
    run_arts(
        pressure_profile=ds_dropsonde_extrap["p"].values,
        temperature_profile=ds_dropsonde_extrap["ta"].values,
        h2o_profile=typhon.physics.mixing_ratio2vmr(ds_dropsonde_extrap["q"].values),
        ws=ws,
        frequencies=all_freqs * 1e9,
        zenith_angle=180,
        height=height,
    )

    # get according hamp data
    freqs_hamp, TBs_hamp[sonde_id] = get_hamp_TBs(hampdata_loc)

    # average double bands
    TB_arts = pd.DataFrame(
        data=np.array(ws.y.value), index=np.array(ws.f_grid.value) / 1e9
    )
    TBs_arts[sonde_id] = average_double_bands(
        TB_arts,
        freqs_hamp,
    )

    #  compare to hamp radiometers
    fig, ax = plot_arts_flux(
        TBs_hamp[sonde_id], TBs_arts[sonde_id], dropsonde_id=sonde_id, time=drop_time
    )
    if not os.path.exists(f"quicklooks/{cfg['flightname']}/arts_calibration"):
        os.makedirs(f"quicklooks/{cfg['flightname']}/arts_calibration")
    fig.savefig(
        f'quicklooks/{cfg["flightname"]}/arts_calibration/TBs_{drop_time.strftime("%Y%m%d_%H%M")}.png',
        dpi=100,
    )

# %% concatenate datasets and save to netcdf
BTs_arts = xr.DataArray(
    np.array(list(TBs_arts)),
    dims=["sonde_id", "frequency"],
    coords={"frequency": all_freqs, "sonde_id": cloud_free_idxs},
)
BTs_arts.to_netcdf(f"arts_calibration_data/{cfg['flightname']}_arts_BTs.nc")
BTs_hamp = xr.DataArray(
    np.array(list(TBs_hamp)),
    dims=["sonde_id", "frequency"],
    coords={"frequency": freqs_hamp, "sonde_id": cloud_free_idxs},
)
BTs_hamp.to_netcdf(f"arts_calibration_data/{cfg['flightname']}_hamp_BTs.nc")
dropsondes_extrap = xr.concat(dropsondes_extrap, dim="sonde_id")
dropsondes_extrap.to_netcdf(
    f"arts_calibration_data/{cfg['flightname']}_dropsondes_extrap.nc"
)

# %% calculate difference
BTs_hamp_sel = BTs_hamp.sel(frequency=BTs_arts.frequency, method="nearest")
BTs_hamp_sel = BTs_hamp_sel.assign_coords(frequency=BTs_arts.frequency)
BTs_diff = BTs_hamp_sel - BTs_arts


# %% plot mean and std of HAMP BTs
import matplotlib.pyplot as plt

frequencies_no_bottom = [54.94, 56.66, 58, 183.31]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.boxplot(BTs_diff.sel(frequency=frequencies_no_bottom))
ax.axhline(0, color="black", linestyle="--")
ax.set_xticklabels(frequencies_no_bottom)
ax.set_xlabel("Frequency / GHz")
ax.set_ylabel("BT HAMP - ARTS / K")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"quicklooks/{cfg['flightname']}/arts_calibration/BT_diff.png", dpi=100)

# %%
