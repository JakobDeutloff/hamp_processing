# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src import plot_quicklooks as plotql
from src import load_data_functions as loadfuncs
from src import readwrite_functions as rwfuncs
from src import earthcare_functions as ecfuncs

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
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
    do_183=True,
    do_11990=True,
    do_kv=True,
    do_cwv=True,
)

# %% find time when earthcare crosses halo
ec_track = ecfuncs.get_earthcare_track(cfg["date"])
ec_under_time = ecfuncs.find_ec_under_time(ec_track, hampdata.flightdata)

# %% produce HAMP single quicklook between startime and endtime
starttime, endtime = hampdata["183"].time[0].values, hampdata["183"].time[-1].values
savefig_format = "png"
savename = path_saveplts / f"timesliceql_{flight}_hamp.png"
dpi = 72
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(28, 20),
    savefigparams=[savefig_format, savename, dpi],
)

# %% produce radar-only single quicklook between startime and endtime
starttime, endtime = hampdata.radar.time[0].values, hampdata.radar.time[-1].values
savefig_format = "png"
savename = path_saveplts / f"timesliceql_{flight}_radar.png"
dpi = 72
plotql.radar_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(12, 6),
    savefigparams=[savefig_format, savename, dpi],
)

# %% produce radiometer-only single quicklook between startime and endtime
starttime, endtime = hampdata["183"].time[0].values, hampdata["183"].time[-1].values
savefig_format = "png"
savename = path_saveplts / f"timesliceql_{flight}_radiometers.png"
dpi = 72
plotql.radiometer_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    figsize=(10, 14),
    savefigparams=[savefig_format, savename, dpi],
)

# %%
starttime, endtime = hampdata["kv"].time[0].values, hampdata["kv"].time[-1].values
timeframe = slice(starttime, endtime)
savefig_format = "png"
savename = path_saveplts / f"timesliceql_{flight}_columnwatervapour.png"
dpi = 72
fig, axes = plotql.plot_kvband_column_water_vapour_retrieval(
    hampdata,
    timeframe,
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(9, 12),
    savefigparams=[savefig_format, savename, dpi],
)

# %% produce hourly HAMP quicklooks
start_hour = pd.Timestamp(hampdata["183"].time[0].values).floor(
    "h"
)  # Round start time to full hour
end_hour = pd.Timestamp(hampdata["183"].time[-1].values).ceil(
    "h"
)  # Round end time to full hour
savefig_format = "png"
dpi = 72
plotql.hamp_hourly_quicklooks(
    hampdata,
    flight,
    start_hour,
    end_hour,
    savefigparams=[savefig_format, path_saveplts, dpi],
)
