# %%
import xarray as xr
import pandas as pd
from orcestra.postprocess.level0 import bahamas, radar_ql, radiometer
from orcestra.postprocess.level1 import (
    filter_radiometer,
    filter_radar,
    correct_radar_height,
    coarsen_radiometer,
)
from plot_functions import produce_hourly_hamp_ql, hamp_ql


# %% read raw data
ds_bahamas = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/BAHAMAS/EUREC4A_HALO_BAHAMAS-QL-10Hz_20200218a.nc"
)
ds_radar = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/Radar/20200218_1013_uncalib_prelim.nc"
)
ds_11990 = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/radiometer/11990/200218.BRT.NC"
)
ds_183 = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/radiometer/183/200218.BRT.NC"
)
ds_kv = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/radiometer/KV/200218.BRT.NC"
)

# %% produce level0 data
ds_bahamas_lev0 = ds_bahamas.pipe(bahamas)
ds_radar_lev0 = ds_radar.pipe(radar_ql)
ds_11990_lev0 = ds_11990.pipe(radiometer)
ds_183_lev0 = ds_183.pipe(radiometer)
ds_kv_lev0 = ds_kv.pipe(radiometer)

# %% produce level1 data
ds_11990_lev1 = ds_11990_lev0.pipe(filter_radiometer, ds_bahamas_lev0).pipe(
    coarsen_radiometer
)
ds_183_lev1 = ds_183_lev0.pipe(filter_radiometer, ds_bahamas_lev0).pipe(
    coarsen_radiometer
)
ds_kv_lev1 = ds_kv_lev0.pipe(filter_radiometer, ds_bahamas_lev0).pipe(
    coarsen_radiometer
)
ds_radar_lev1 = ds_radar_lev0.pipe(filter_radar, ds_bahamas_lev0).pipe(
    correct_radar_height, ds_bahamas_lev0
)

#  %% produce one png quicklook for whole flight
flightname = "20200218"
start_time = pd.Timestamp(ds_183_lev1.time[0].values).floor("H")
end_time = pd.Timestamp(ds_183_lev1.time[-1].values).ceil("H")
fig, ax = hamp_ql(
    ds_radar_lev1,
    ds_11990_lev1,
    ds_183_lev1,
    ds_kv_lev1,
    slice(start_time, end_time),
    figsize=(20, 14),
)
fig.savefig(
    f"quicklooks/{flightname}/ql_{start_time.strftime('%Y%m%d_%H%M')}_to_{end_time.strftime('%Y%m%d_%H%M')}.png",
    bbox_inches="tight",
    dpi=500,
)

# %% produce hourly pdf quicklooks
produce_hourly_hamp_ql(
    ds_radar_lev1, ds_11990_lev1, ds_183_lev1, ds_kv_lev1, "20200218"
)
