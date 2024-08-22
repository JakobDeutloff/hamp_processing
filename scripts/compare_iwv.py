# %% import
from pylab import *
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import easygems.remap as egr
from orcestra.postprocess.level0 import bahamas
from orcestra.postprocess.level1 import filter_radiometer

# %% User-defined parameters

# Choose here the case to plot
mmdd = "0810"

# Choose flight
flight = "HALO-20240811a"

# %% Reading the dataset
datadir = (
    "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/orcestra_1250m_"
    + mmdd
    + "/"
)
weightsdir = "/work/bm1183/m301049/orcestra/aux/"
datafile = "orcestra_1250m_" + mmdd + "_atm_2d_ml_DOM01_2024" + mmdd + "T000000Z.nc"
meshname = "ORCESTRA_1250m_DOM01"
bahamasdir = f"/work/bm1183/m301049/orcestra/{flight}/bahamas/"
iwv_dir = f"/work/bm1183/m301049/orcestra/{flight}/radiometer/KV/"

ds = xr.open_dataset(datadir + datafile, chunks={"ncells": -1})
grid = xr.open_dataset(datadir + meshname + ".nc")
ds_bahamas = xr.open_mfdataset(f"{bahamasdir}/*.nc").load().pipe(bahamas)
ds_iwv = (
    xr.open_mfdataset(f"{iwv_dir}/*.IWV.NC")
    .load()
    .pipe(filter_radiometer, ds_bahamas["IRS_ALT"], ds_bahamas["IRS_PHI"])
)
ds_iwv_resampled = ds_iwv.resample(time="10min").mean()
ds_bahamas_resampled = (
    ds_bahamas.resample(time="10min").mean().sel(time=ds_iwv_resampled.time)
)

lat_max, lat_min = ds_bahamas["IRS_LAT"].max(), ds_bahamas["IRS_LAT"].min()
lon_max, lon_min = ds_bahamas["IRS_LON"].max(), ds_bahamas["IRS_LON"].min()
lat_vec = np.deg2rad(np.arange(lat_min, lat_max, 1 / (100 / 1.25)))
lon_vec = np.deg2rad(np.arange(lon_min, lon_max, 1 / (100 / 1.25)))
lon, lat = np.meshgrid(lon_vec, lat_vec)

# %% Compute (load) remapping weights
weightsname = f"weights_{meshname}_halo_{flight}_3d"
if os.path.exists(weightsdir + weightsname + ".nc"):
    print("Reading weights")
    weights = xr.open_dataset(weightsdir + weightsname + ".nc")
else:
    print("Calculating weights")
    weights = egr.compute_weights_delaunay(
        (grid.clon, grid.clat), (lon.ravel(), lat.ravel())
    )
    weights.to_netcdf(weightsdir + weightsname + ".nc", mode="w")

# %% Apply remapping function only to the variables we need
ds_remap = xr.apply_ufunc(
    egr.apply_weights,
    ds,
    kwargs=weights,
    input_core_dims=[["ncells"]],
    output_core_dims=[["xy"]],
    output_dtypes=["f4"],
    vectorize=True,
    dask="parallelized",
    dask_gufunc_kwargs={
        "output_sizes": {"xy": lon.size},
    },
)

# Assign the MultiIndex to the 'xy' dimension
ds_remap = ds_remap.assign_coords(
    xy=pd.MultiIndex.from_product(
        (np.degrees(lat[:, 0]), np.degrees(lon[0, :])),
        names=("lat", "lon"),
    )
).unstack("xy")

# %% select path along the flight track
ds_track = ds_remap.sel(
    lat=ds_bahamas_resampled["IRS_LAT"],
    lon=ds_bahamas_resampled["IRS_LON"],
    time=ds_bahamas_resampled.time,
    method="nearest",
    drop=True,
)
prw = ds_track["prw"].load()

# %% calculate correlation
p = np.corrcoef(ds_iwv_resampled.IWV, prw.values)

# %% plot prw path along the flight track
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ds_iwv.time, ds_iwv.IWV, label="HALO IWV", color="k")
ax.scatter(prw.time, prw.values, label="ICON PRW", marker="x", color="r")
ax.set_xlabel("Time")
ax.set_ylabel("Colum Water / kg m$^{-2}$")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.8, 0.05, f"Correlation: {p[0, 1]:.2f}", transform=ax.transAxes)
fig.savefig(f"quicklooks/icon_comparison/{flight}_prw.png", dpi=300)

# %%
