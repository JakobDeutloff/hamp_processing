# %%
import xarray as xr
import pandas as pd
from orcestra.postprocess.level0 import bahamas, radiometer
from orcestra.postprocess.level1 import (
    filter_radiometer,
)
from src.plot_functions import radiometer_ql
import numpy as np
import matplotlib.pyplot as plt

# %% read planet data
path_planet = "/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/planet_data/"
iwg = pd.read_csv(path_planet + "2024-08-16-TE03124060001-IWG1.csv")
iwg.set_index("issued", inplace=True)
iwg.index = pd.to_datetime(iwg.index)
iwg.index = iwg.index.tz_localize(None)
iwg.index.rename("time", inplace=True)
ds_iwg = xr.Dataset.from_dataframe(iwg)
ds_iwg = ds_iwg.assign_coords(time=iwg.index)

# %% read radiometer data all flights
path_radiometer = "/Volumes/Upload/HALO/Radiometer"
path_bahamas = (
    "/Volumes/ORCESTRA/HALO-20240811a/bahamas/QL_HALO-20240811a_BAHAMAS_V01.nc"
)
flights = ["RF01_20240811", "RF02_20240813", "RF03_20240816"]
dates = ["240811", "240813", "240816"]
radiometer_datasets = {}
height_datasets = {}
roll_datasets = {}
ds_bahamas_01 = xr.open_dataset(path_bahamas).pipe(bahamas)
ds_bahamas_02 = xr.open_dataset(path_bahamas.replace("20240811", "20240813")).pipe(
    bahamas
)
for flight, date in zip(flights, dates):
    radiometer_datasets[flight] = {
        "11990": xr.open_dataset(
            f"{path_radiometer}/{flight}/11990/{date}.BRT.NC"
        ).pipe(radiometer),
        "183": xr.open_dataset(f"{path_radiometer}/{flight}/183/{date}.BRT.NC").pipe(
            radiometer
        ),
        "KV": xr.open_dataset(f"{path_radiometer}/{flight}/KV/{date}.BRT.NC").pipe(
            radiometer
        ),
    }

# %% read radar data for first two flights 



# %% build height and roll datasets
height_datasets[flights[0]] = ds_bahamas_01["IRS_ALT"]
height_datasets[flights[1]] = ds_bahamas_02["IRS_ALT"]
height_datasets[flights[2]] = ds_iwg["data.gps_msl_alt"]
roll_datasets[flights[0]] = ds_bahamas_01["IRS_PHI"]
roll_datasets[flights[1]] = ds_bahamas_02["IRS_PHI"]
roll_datasets[flights[2]] = ds_iwg["data.roll"]

# %% filter radiometer data
for flight in flights:
    radiometer_datasets[flight]["11990"] = radiometer_datasets[flight][
        "11990"
    ].pipe(filter_radiometer, height_datasets[flight], roll_datasets[flight])
    radiometer_datasets[flight]["183"] = radiometer_datasets[flight][
        "183"
    ].pipe(filter_radiometer, height_datasets[flight], roll_datasets[flight])
    radiometer_datasets[flight]["KV"] = radiometer_datasets[flight][
        "KV"
    ].pipe(filter_radiometer, height_datasets[flight], roll_datasets[flight])


# %% plot all radiometer data
for flight in flights:
    fig, axes = radiometer_ql(
        radiometer_datasets[flight]["11990"],
        radiometer_datasets[flight]["183"],
        radiometer_datasets[flight]["KV"],
        timeframe=slice(
            radiometer_datasets[flight]["11990"].time[0].values,
            radiometer_datasets[flight]["11990"].time[-1].values,
        ),
        figsize=(14, 14)
    )
    fig.savefig("quicklooks/" + flight + "_radiometer_ql.png", dpi=300)

# %%
