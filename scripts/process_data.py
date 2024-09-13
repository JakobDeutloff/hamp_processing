# %% import modules
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import xarray as xr
from orcestra.postprocess.level1 import bahamas, radiometer, radar, iwv

# %% read paths from config file
date = "20240825"

config_file = "process_config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# configure paths with data
radar_path = config["root"] + config["radar"].format(date=date)
radiometer_path = config["root"] + config["radiometer"].format(date=date)
bahamas_path = config["root"] + config["bahamas"].format(date=date)


# %% load raw data
ds_radar_raw = xr.open_mfdataset(radar_path).load()
ds_bahamas = xr.open_dataset(bahamas_path)
ds_iwv_raw = xr.open_dataset(f"{radiometer_path}/KV/{date[2:]}.IWV.NC")
radiometers = ["183", "11990", "KV"]
ds_radiometers_raw = {}
for radio in radiometers:
    ds_radiometers_raw[radio] = xr.open_dataset(
        f"{radiometer_path}/{radio}/{date[2:]}.BRT.NC"
    )

# %% do level 1 processing
ds_bahamas_lev1 = bahamas(ds_bahamas)
ds_radar_lev1 = radar(ds_radar_raw, ds_bahamas_lev1)
ds_iwv_lev1 = iwv(ds_iwv_raw, ds_bahamas_lev1)
ds_radiometers_lev1 = {}
for radio in radiometers:
    ds_radiometers_lev1[radio] = radiometer(ds_radiometers_raw[radio], ds_bahamas_lev1)

# %% concatenate radiometers
ds_radiometers_lev1_concat = xr.concat(
    [ds_radiometers_lev1[radio] for radio in radiometers], dim="frequency"
)
