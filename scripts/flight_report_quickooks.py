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
path_saveplts = cfg["path_saveplts"]
flightname = cfg["flightname"]
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

# %% mask points in 90 GHz radiometer with unrealistic values
hampdata.radio11990 = hampdata.radio11990.where(hampdata.radio11990.TBs < 300)

# %% find time when earthcare crosses halo
ec_track = ecfuncs.get_earthcare_track(cfg["date"])
ec_under_time = ecfuncs.find_ec_under_time(ec_track, hampdata.flightdata)

plot_duration = pd.Timedelta("30m")
ec_starttime, ec_endtime = (
    ec_under_time - plot_duration / 2,
    ec_under_time + plot_duration / 2,
)

# %% produce HAMP single quicklook between startime and endtime
flight_starttime, flight_endtime = (
    hampdata["183"].time[0].values,
    hampdata["183"].time[-1].values,
)
savefig_format = "png"
savename = path_saveplts / f"hamp_fullflight_{flightname}.png"
dpi = 40
timeframe = slice(flight_starttime, flight_endtime)
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe,
    flightname,
    ec_under_time=ec_under_time,
    figsize=(28, 20),
    savefigparams=[savefig_format, savename, dpi],
)

# %% produce ec_under single quicklook
savefig_format = "png"
savename = path_saveplts / f"hamp_ec_under_{flightname}.png"
dpi = 40
timeframe = slice(ec_starttime, ec_endtime)
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe,
    flightname,
    ec_under_time=ec_under_time,
    figsize=(28, 20),
    savefigparams=[savefig_format, savename, dpi],
)

# %% produce radar-only ec_under single quicklook
savefig_format = "png"
savename = path_saveplts / f"hamp_radar_ec_under_{flightname}.png"
dpi = 54
timeframe = slice(ec_starttime, ec_endtime)
plotql.radar_quicklook(
    hampdata,
    timeframe,
    flightname,
    ec_under_time=ec_under_time,
    figsize=(15, 7),
    is_latllonaxes=True,
    savefigparams=[savefig_format, savename, dpi],
)

# %%
