# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr
import pandas as pd
from src import plot_quicklooks as plotql
from src import load_data_functions as loadfuncs
from src import helper_functions as helpfuncs

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = helpfuncs.extract_config_params(configfile)
flight = cfg["flight"]
path_saveplts = cfg["path_saveplts"]
radiometer_date = cfg["radiometer_date"]
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
ec_track = helpfuncs.get_earthcare_track(cfg["date"])

# %% find time when earthcare crosses halo
ec_under_time = helpfuncs.find_ec_under_time(ec_track, hampdata.flightdata)
window = "60min"
plot_duration = pd.Timedelta(window)
starttime, endtime = (
    ec_under_time - plot_duration / 2,
    ec_under_time + plot_duration / 2,
)

# %% Write timeslice of Level 1 post-processed radar data to .nc file
ncfilename = cfg["path_writeradar"] / f"earthcare_level1_{window}_{flight}.nc"
sliced_level1radar = hampdata.radar.sel(time=slice(starttime, endtime))
sliced_level1radar.to_netcdf(ncfilename)

sizemb_og = int(hampdata.radar.nbytes / 1024 / 1024)  # convert bytes to MB
sizemb_slice = int(sliced_level1radar.nbytes / 1024 / 1024)  # convert bytes to MB
print(
    f"Timeslice of radar data saved to: {ncfilename}\nRadar data original size = {sizemb_og}MB\nsize = {sizemb_slice}MB"
)

# %% produce radar-only single quicklook from sliced radardata .nc file
ds_radar = xr.open_mfdataset(ncfilename)
starttime, endtime = ds_radar.time[0].values, ds_radar.time[-1].values
savefig_format = "png"
savename = cfg["path_writeradar"] / f"earthcare_level1_{window}_{flight}.png"
dpi = 64
plotql.radar_quicklook(
    None,
    timeframe=slice(starttime, endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(12, 6),
    savefigparams=[savefig_format, savename, dpi],
    ds_radar=ds_radar,
)
