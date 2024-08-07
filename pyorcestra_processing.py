# %%
from orcestra.postprocess.level0 import bahamas, radar_ql, radiometer
from orcestra.postprocess.level1 import filter_radiometer, filter_radar, correct_radar_height, coarsen_radiometer
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


# %% read raw data
ds_bahamas = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/BAHAMAS/EUREC4A_HALO_BAHAMAS-QL-10Hz_20200218a.nc"
)
ds_radar = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/Radar/20200218_1013_uncalib_prelim.nc"
)
ds_radiometer = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/radiometer/11990/200218.BRT.NC"
)
ds_radiometer_met = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/radiometer/183/200218.MET.NC"
)


# %% produce level0 data 
ds_bahamas_lev0 = ds_bahamas.pipe(bahamas)
ds_radar_lev0 = ds_radar.pipe(radar_ql)
ds_radiometer_lev0 = ds_radiometer.pipe(radiometer)

# %% produce level1 data 
ds_radiometer_lev1 = ds_radiometer_lev0.pipe(filter_radiometer, ds_bahamas_lev0).pipe(coarsen_radiometer)
ds_radar_lev1 = ds_radar_lev0.pipe(filter_radar, ds_bahamas_lev0).pipe(correct_radar_height, ds_bahamas_lev0)
ds_radar_unfiltered = ds_radar_lev0.pipe(correct_radar_height, ds_bahamas_lev0)
# %% plot radar data 
fig, ax = plt.subplots(figsize=(10, 4))
ds_plot = ds_radar_unfiltered.sel(time=slice('2020-02-18 11:00', '2020-02-18 12:00'))
pcol = ax.pcolormesh(ds_plot.time, 
              ds_plot.height, 
              ds_plot.dBZg.where(ds_plot.dBZg > -25).T, 
              cmap="turbo",
              vmin=-25,
              vmax=25,)
cb = fig.colorbar(pcol, ax=ax)
ax.set_ylim(0, 12000)

# %% plot radiometer data 
fig, ax = plt.subplots(figsize=(10, 4))
ds_plot = ds_radiometer_lev1
ds_plot['TBs'].plot.line(ax=ax, x='time')

# %%
