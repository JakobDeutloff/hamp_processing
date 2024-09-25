# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src import plot_quicklooks as plotql
from src import load_data_functions as loadfuncs
from src import readwrite_functions as rwfuncs
from src import earthcare_functions as ecfuncs
import yaml

# %%


def plot_radar(date):
    # change the date in config.yaml to date
    configfile = "config.yaml"
    with open(configfile, "r") as file:
        config_yaml = yaml.safe_load(file)
    config_yaml["date"] = date
    with open(configfile, "w") as file:
        yaml.dump(config_yaml, file)

    # read config
    cfg = rwfuncs.extract_config_params(configfile)
    path_saveplts = cfg["path_saveplots"]
    flightname = cfg["flightname"]

    # create HAMP post-processed data
    hampdata = loadfuncs.load_hamp_data(
        cfg["path_radar"], cfg["path_radiometers"], cfg["path_iwv"]
    )

    # find time when earthcare crosses halo
    ec_track = ecfuncs.get_earthcare_track(cfg["date"])
    ec_under_time = ecfuncs.find_ec_under_time(ec_track, hampdata.radar)

    plot_duration = pd.Timedelta("30m")
    ec_starttime, ec_endtime = (
        ec_under_time - plot_duration / 2,
        ec_under_time + plot_duration / 2,
    )

    savefig_format = "png"
    savename = path_saveplts + f"/hamp_radar_ec_under_{flightname}_highres.png"
    dpi = 256
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


# %% run the function
dates = [
    "20240811",
    "20240813",
    "20240816",
    "20240818",
    "20240821",
    "20240822",
    "20240825",
    "20240827",
    "20240829",
    "20240831",
    "20240903",
    "20240907",
    "20240909",
    "20240912",
    "20240914",
    "20240916",
    "20240919",
    "20240921",
]

for date in dates:
    plot_radar(date)


# %%
