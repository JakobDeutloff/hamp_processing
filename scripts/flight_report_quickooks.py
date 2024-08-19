# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src import plot_quicklooks as plotql
from src import load_data_functions as loadfuncs
from src import helper_functions as helpfuncs
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

# %% find time when earthcare crosses halo
ec_track = helpfuncs.get_earthcare_track(config["date"])
ec_under_time = helpfuncs.find_ec_under_time(ec_track, hampdata.flightdata)

# %% produce HAMP single quicklook between startime and endtime
starttime, endtime = hampdata["183"].time[0].values, hampdata["183"].time[-1].values
is_savefig = "png"
savename = f"{savedir}/hamp_timesliceql_{flight}.png"
dpi = 72
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(24, 20),
    savefigparams=[is_savefig, savename, dpi],
)

plot_duration = pd.Timedelta("30m")
ec_starttime, ec_endtime = (
    ec_under_time - plot_duration / 2,
    ec_under_time + plot_duration / 2,
)

# %% produce ec_under single quicklook
is_savefig = "png"
savename = f"{savedir}/hamp_ec_under_{flight}.png"
dpi = 72
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(ec_starttime, ec_endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(24, 20),
    savefigparams=[is_savefig, savename, dpi],
)

# %% produce radar-only ec_under single quicklook
is_savefig = "png"
savename = f"{savedir}/hamp_radar_ec_under_{flight}.png"
dpi = 72
plotql.radar_quicklook(
    hampdata,
    timeframe=slice(ec_starttime, ec_endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(15, 6),
    savefigparams=[is_savefig, savename, dpi],
)
