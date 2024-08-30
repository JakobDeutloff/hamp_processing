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
from src.plot_functions import plot_arts_flux
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

# %% find hamp data at dropsonde location
drop_id = 29
ds_dropsonde_loc, hampdata_loc, height, drop_time = get_profiles(
    drop_id, ds_dropsonde, hampdata
)

# %% extrapolate dropsonde data
ds_dropsonde_extrap = extrapolate_dropsonde(ds_dropsonde_loc, height)

# %% setup workspace
ws = setup_workspace()

# %% run arts
run_arts(
    pressure_profile=ds_dropsonde_extrap["p"].values,
    temperature_profile=ds_dropsonde_extrap["ta"].values,
    h2o_profile=typhon.physics.mixing_ratio2vmr(ds_dropsonde_extrap["q"].values),
    ws=ws,
    frequencies=np.array(all_freqs) * 1e9,
    zenith_angle=180,
    height=height,
)

# %% compare to hamp radiometers
plot_arts_flux(ws, hampdata_loc, dropsonde_id=drop_id, time=drop_time)

# %%
