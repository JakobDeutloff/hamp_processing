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
### ----------- USER PARAMETERS YOU MUST SET ----------- ###
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
# %%
is_savefig = False
savename = f"{savedir}/ec_under_{flight}.png"
dpi = 500
fig, axes = plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(crossing_time - windowsize / 2, crossing_time + windowsize / 2),
    flight=flight,
    figsize=(18, 18),
    savefigparams=[is_savefig, savename, dpi],
)
for ax in axes:
    ax.axvline(crossing_time, color="r", linestyle="-")
fig.savefig(savename, dpi=dpi)


# %%
