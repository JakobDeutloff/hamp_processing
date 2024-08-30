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
)
from src.plot_functions import plot_arts_flux, get_hamp_TBs
from src.dropsonde_processing import get_all_clouds_flags_dropsondes
import pyarts
import numpy as np
import typhon

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
    do_radar=False,
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

# %% check if dropsondes are cloud free
ds_dropsonde = get_all_clouds_flags_dropsondes(ds_dropsonde)
cloud_free_idxs = (
    ds_dropsonde["sonde_id"].where(ds_dropsonde["cloud_flag"] == 0, drop=True).values
)

# %% setup workspace
ws = setup_workspace()

# %% loop over cloud free sondes
TBs_arts = np.zeros((len(cloud_free_idxs), len(all_freqs)))
TBs_hamp = np.zeros((len(cloud_free_idxs), len(all_freqs)))
dropsondes_extrap = []

for i, sonde_id in enumerate(cloud_free_idxs):
    # get profiles
    ds_dropsonde_loc, hampdata_loc, height, drop_time = get_profiles(
        sonde_id, ds_dropsonde, hampdata
    )
    # extrapolate dropsonde data
    ds_dropsonde_extrap = extrapolate_dropsonde(ds_dropsonde_loc, height)
    dropsondes_extrap.append(ds_dropsonde_extrap)
    # run arts
    run_arts(
        pressure_profile=ds_dropsonde_extrap["p"].values,
        temperature_profile=ds_dropsonde_extrap["ta"].values,
        h2o_profile=typhon.physics.mixing_ratio2vmr(ds_dropsonde_extrap["q"].values),
        ws=ws,
        frequencies=np.array(all_freqs) * 1e9,
        zenith_angle=180,
        height=height,
    )
    TBs_arts[i, :] = np.array(ws.y.value)
    TBs_hamp[i, :] = get_hamp_TBs(hampdata_loc, freqs=all_freqs)
    #  compare to hamp radiometers
    fig, ax = plot_arts_flux(ws, TBs_hamp[i, :], dropsonde_id=sonde_id, time=drop_time)
    if not os.path.exists(f"quicklooks/{cfg['flightname']}/arts_calibration"):
        os.makedirs(f"quicklooks/{cfg['flightname']}/arts_calibration")
    fig.savefig(
        f'quicklooks/{cfg["flightname"]}/arts_calibration/TBs_{drop_time.strftime("%Y%m%d_%H%M")}.png',
        dpi=100,
    )

# %% concatenate datasets and save to netcdf
BTs_arts = xr.DataArray(
    np.array(list(TBs_arts.values())),
    dims=["sonde_id", "frequencies"],
    coords={"frequencies": all_freqs, "sonde_id": cloud_free_idxs},
)
BTs_arts.to_netcdf(f"arts_calibration_data/{cfg['flightname']}_arts_BTs.nc")
BTs_hamp = xr.DataArray(
    np.array(list(TBs_hamp.values())),
    dims=["sonde_id", "frequencies"],
    coords={"frequencies": all_freqs, "sonde_id": cloud_free_idxs},
)
BTs_hamp.to_netcdf(f"arts_calibration_data/{cfg['flightname']}_hamp_BTs.nc")
dropsondes_extrap = xr.concat(dropsondes_extrap, dim="sonde_id")
dropsondes_extrap.to_netcdf(
    f"arts_calibration_data/{cfg['flightname']}_dropsondes_extrap.nc"
)

# %% calculate mean BTs and differences
mean_BT_arts = BTs_arts.mean("sonde_id")
