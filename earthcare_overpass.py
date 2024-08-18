# %%
import xarray as xr
from orcestra.postprocess.level0 import bahamas, radiometer, radar
from orcestra.postprocess.level1 import (
    filter_radiometer,
    filter_radar,
    correct_radar_height,
)
from orcestra import sat
from src.plot_functions import produce_hourly_hamp_ql, hamp_ql
from src.helper_functions import read_planet
import pandas as pd

# %% define data directories
flight = "RF01_20240811"
date = "240811"

path_planet = "/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/planet_data/2024-08-16-TE03124060001-IWG1.csv"
path_radiometer = f"/Volumes/Upload/HALO/Radiometer/{flight}"
path_radar = f"/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/Radar_Data/{flight}/*.nc"
path_bahamas = (
    f"/Volumes/ORCESTRA/HALO-20{date}a/bahamas/QL_HALO-20{date}a_BAHAMAS_V01.nc"
)
# %% read data and do level0 processing
ds_planet = read_planet(path_planet)
ds_bahamas = xr.open_dataset(path_bahamas).pipe(bahamas)
ds_radar_lev0 = xr.open_mfdataset(f"{path_radar}").load().pipe(radar)
ds_183_lev0 = xr.open_dataset(f"{path_radiometer}/183/{date}.BRT.NC").pipe(radiometer)
ds_11990_lev0 = xr.open_dataset(f"{path_radiometer}/11990/{date}.BRT.NC").pipe(
    radiometer
)
ds_kv_lev0 = xr.open_dataset(f"{path_radiometer}/KV/{date}.BRT.NC").pipe(radiometer)
# %% level1 processing
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
# %% get earthcare track forecasts
track = sat.SattrackLoader("EARTHCARE", "2024-08-10", kind="PRE").get_track_for_day(
    "2024-08-11"
)
track_afternoon = track.where(
    track.time > pd.Timestamp("2024-08-11T06:00:00"), drop=True
)


# %% find time when earthcare crosses halo
def distance(lat_halo, lon_halo, lat_ec, lon_ec):
    return ((lat_halo - lat_ec) ** 2 + (lon_halo - lon_ec) ** 2) ** 0.5


def find_crossing_time(track, ds_bahamas):
    lat_halo = ds_bahamas["IRS_LAT"]
    lon_halo = ds_bahamas["IRS_LON"]
    lat_ec = track.lat
    lon_ec = track.lon
    dist = distance(lat_halo, lon_halo, lat_ec, lon_ec)
    return dist.idxmin()


crossing_time = find_crossing_time(track_afternoon, ds_bahamas)

# %%
windowsize = pd.Timedelta("30m")
hamp_ql(
    ds_radar_lev1,
    ds_11990_lev1,
    ds_183_lev1,
    ds_kv_lev1,
    timeframe=slice(crossing_time - windowsize / 2, crossing_time + windowsize / 2),
    flight=flight,
    figsize=(18, 18),
)

# %%
