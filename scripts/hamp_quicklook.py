# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src import plot_quicklooks as plotql
from src import load_data_functions as loadfuncs
from src.helper_functions import find_crossing_time
from orcestra import sat
import yaml

# %%
### ----------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML ----------- ###
# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from the configuration
flight = config["flight"]
date = config["date"]
flightletter = config["flightletter"]
is_planet = config["is_planet"]

# Format paths using the extracted parameters
path_bahamas = config["paths"]["bahamas"].format(date=date, flightletter=flightletter)
path_radiometer = config["paths"]["radiometer"].format(
    date=date, flightletter=flightletter
)
path_radar = config["paths"]["radar"].format(flight=flight)
savedir = config["savedir"].format(flight=flight)

### ---------------------------------------------------- ###

# %% create HAMP post-processed data
hampdata = loadfuncs.do_post_processing(
    path_bahamas, path_radar, path_radiometer, date[2:], is_planet=is_planet
)

# %% get earthcare track forecasts
day = pd.Timestamp(config["date"])
track = sat.SattrackLoader(
    "EARTHCARE", (day - pd.Timedelta("1d")).strftime(format="%Y-%m-%d"), kind="PRE"
).get_track_for_day(day.strftime(format="%Y-%m-%d"))

track = track.where(track.time > (day + pd.Timedelta("1h")), drop=True)

# %% find time when earthcare crosses halo
crossing_time = find_crossing_time(track, hampdata.flightdata)
windowsize = pd.Timedelta("30m")

# %% produce HAMP single quicklook between startime and endtime
starttime, endtime = hampdata["183"].time[0].values, hampdata["183"].time[-1].values
is_savefig = True
savename = f"{savedir}/hamp_timesliceql_{flight}.png"
dpi = 500
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    ec_under_time=crossing_time,
    figsize=(18, 18),
    savefigparams=[is_savefig, savename, dpi],
)

# %% produce ec_under single quicklook
windowsize = pd.Timedelta("30m")
is_savefig = True
savename = f"{savedir}/ec_under_{flight}.png"
dpi = 500
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(crossing_time - windowsize / 2, crossing_time + windowsize / 2),
    flight=flight,
    ec_under_time=crossing_time,
    figsize=(18, 18),
    savefigparams=[is_savefig, savename, dpi],
)

# %% produce radar-only single quicklook between startime and endtime
starttime, endtime = hampdata.radar.time[0].values, hampdata.radar.time[-1].values
is_savefig = True
savename = f"{savedir}/radar_timesliceql_{flight}.png"
dpi = 500
plotql.radar_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    figsize=(9, 5),
    savefigparams=[is_savefig, savename, dpi],
)

# %% produce radiometer-only single quicklook between startime and endtime
starttime, endtime = hampdata["183"].time[0].values, hampdata["183"].time[-1].values
is_savefig = True
savename = f"{savedir}/radiometers_timesliceql_{flight}.png"
dpi = 500
plotql.radiometer_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    figsize=(10, 14),
    savefigparams=[is_savefig, savename, dpi],
)

# %% produce hourly HAMP quicklooks
start_hour = pd.Timestamp(hampdata["183"].time[0].values).floor(
    "h"
)  # Round start time to full hour
end_hour = pd.Timestamp(hampdata["183"].time[-1].values).ceil(
    "h"
)  # Round end time to full hour
is_savepdf = False
plotql.hamp_hourly_quicklooks(
    hampdata, flight, start_hour, end_hour, saveparams=[is_savepdf, savedir]
)

# %%
