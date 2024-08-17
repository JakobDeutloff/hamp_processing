# %%
import xarray as xr
import pandas as pd
from orcestra.postprocess.level0 import bahamas, radiometer, radar
from orcestra.postprocess.level1 import (
    filter_radiometer,
    filter_radar,
    correct_radar_height,
)
from src.plot_functions import produce_hourly_hamp_ql, hamp_ql
from src.helper_functions import read_planet
import numpy as np
import matplotlib.pyplot as plt

# %% define data directories
flight = "RF01_20240811"
date = "240811"
path_planet = "/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/planet_data/2024-08-16-TE03124060001-IWG1.csv"
path_radiometer = f"/Volumes/Upload/HALO/Radiometer/{flight}"
path_radar = f"/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/Radar_Data/RF01_20{date}/20{date}_1215.nc"
path_bahamas = (
    f"/Volumes/ORCESTRA/HALO-20{date}a/bahamas/QL_HALO-20{date}a_BAHAMAS_V01.nc"
)

# %% read  level 0 data
ds_planet = read_planet(path_planet)
ds_bahamas = xr.open_dataset(path_bahamas).pipe(bahamas)
ds_radar_lev0 = xr.open_dataset(path_radar).pipe(radar)
ds_183_lev0 = xr.open_dataset(f"{path_radiometer}/183/{date}.BRT.NC").pipe(radiometer)
ds_11990_lev0 = xr.open_dataset(f"{path_radiometer}/11990/{date}.BRT.NC").pipe(
    radiometer
)
ds_kv_lev0 = xr.open_dataset(f"{path_radiometer}/KV/{date}.BRT.NC").pipe(radiometer)
# %% produce level1 data
ds_radar_lev1 = ds_radar_lev0.pipe(filter_radar, ds_bahamas["IRS_PHI"]).pipe(
    correct_radar_height,
    ds_bahamas["IRS_PHI"],
    ds_bahamas["IRS_THE"],
    ds_bahamas["IRS_ALT"],
)
ds_183_lev1 = ds_183_lev0.pipe(
    filter_radiometer, ds_bahamas["IRS_ALT"], ds_bahamas["IRS_PHI"]
)
ds_11990_lev1 = ds_11990_lev0.pipe(
    filter_radiometer, ds_bahamas["IRS_ALT"], ds_bahamas["IRS_PHI"]
)
ds_kv_lev1 = ds_kv_lev0.pipe(
    filter_radiometer, ds_bahamas["IRS_ALT"], ds_bahamas["IRS_PHI"]
)

# %% produce hamp quicklook
fig, ax = hamp_ql(
    ds_radar_lev1,
    ds_11990_lev1,
    ds_183_lev1,
    ds_kv_lev1,
    timeframe=slice(ds_radar_lev1.time[0].values, ds_radar_lev1.time[-1].values),
    flight=flight,
)
fig.savefig(f"quicklooks/hamp/{flight}/hamp_ql_{flight}.png", dpi=500)

# %% produce hourly hamp quicklooks
produce_hourly_hamp_ql(ds_radar_lev1, ds_11990_lev1, ds_183_lev1, ds_kv_lev1, flight)

# %%
