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

# %% get earthcare track forecasts
ec_track = helpfuncs.get_earthcare_track(config["date"])

# %% find time when earthcare crosses halo
ec_under_time = helpfuncs.find_ec_under_time(ec_track, hampdata.flightdata)
plot_duration = pd.Timedelta("30m")
starttime, endtime = (
    ec_under_time - plot_duration / 2,
    ec_under_time + plot_duration / 2,
)

# %% produce ec_under single quicklook
is_savefig = "png"
savename = f"{savedir}/ec_under_{flight}.png"
dpi = 500
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(18, 18),
    savefigparams=[is_savefig, savename, dpi],
)

# %% produce radar-only ec_under single quicklook
is_savefig = "png"
savename = f"{savedir}/ec_under_{flight}_radar.png"
dpi = 500
plotql.radar_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    ec_under_time=ec_under_time,
    figsize=(15, 6),
    savefigparams=[is_savefig, savename, dpi],
)
