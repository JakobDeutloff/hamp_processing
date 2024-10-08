# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr
import pandas as pd
from src import plot_quicklooks as plotql
from src import load_data_functions as loadfuncs
from src import readwrite_functions as rwfuncs
from src import earthcare_functions as ecfuncs

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
flightname = cfg["flightname"]
path_saveplts = cfg["path_saveplts"]
### ------------------------------------------------------------- ###

# %% create HAMP post-processed data
hampdata = loadfuncs.do_post_processing(
    cfg["path_bahamas"],
    cfg["path_radar"],
    cfg["path_radiometer"],
    cfg["radiometer_date"],
    is_planet=cfg["is_planet"],
    do_radar=True,
    do_183=False,
    do_11990=False,
    do_kv=False,
    do_cwv=False,
)

# %% get earthcare track forecasts
ec_track = ecfuncs.get_earthcare_track(cfg["date"])

# %% find time when earthcare crosses halo
ec_under_time = ecfuncs.find_ec_under_time(ec_track, hampdata.flightdata)
window = "60min"
plot_duration = pd.Timedelta(window)
starttime, endtime = (
    ec_under_time - plot_duration / 2,
    ec_under_time + plot_duration / 2,
)

# %% Write timeslice of Level 1 post-processed radar data to .nc file
ncfilename = cfg["path_writedata"] / f"earthcare_level1_{window}_{flightname}.nc"
rwfuncs.write_level1data_timeslice(
    hampdata, "radar", slice(starttime, endtime), ncfilename
)

# %% produce radar-only single quicklook from sliced radardata .nc file
ds_radar = xr.open_mfdataset(ncfilename)
starttime, endtime = ds_radar.time[0].values, ds_radar.time[-1].values
savefig_format = "png"
savename = cfg["path_writedata"] / f"earthcare_level1_{window}_{flightname}.png"
dpi = 64
timeframe = slice(starttime, endtime)
plotql.radar_quicklook(
    None,
    timeframe,
    flightname,
    ec_under_time=ec_under_time,
    figsize=(12, 6),
    savefigparams=[savefig_format, savename, dpi],
    ds_radar=ds_radar,
    is_latllonaxes=False,
)
