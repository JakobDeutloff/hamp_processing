# %% inport
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import eurec4a
from scipy import interpolate
from tqdm import tqdm

# %% define functions
def pp_bahamas(ds):
    """Pre-processing of BAHAMAS datasets."""
    # Fix time coordinate
    ds = ds.rename({"tid": "time", "TIME": "time"}).set_index(time="time")

    # Add length of line-of-sight in m
    ds["LOS"] = (
        ds.IRS_ALT / np.cos(np.radians(ds.IRS_THE)) / np.cos(np.radians(ds.IRS_PHI))
    )

    return ds


def pp_radar(ds):
    """Pre-processing of Radar datasets."""
    # Fix time coordinate
    datetime = np.datetime64("1970-01-01", "ns")
    datetime += ds.time.values * np.timedelta64(1, "s")
    datetime += ds.microsec.values * np.timedelta64(1, "us")

    ds["time"] = datetime
    ds = ds.drop_vars("microsec")

    # Add reflectivity in dB
    ds = ds.assign(dBZg=lambda dx: 10 * np.log10(dx.Zg))

    return ds


def pp_radiometer(ds):
    return (
        ds.rename(
            number_frequencies="frequency",
        )
        .set_index(
            frequency="frequencies",
        )
        .sortby("time")
        .resample(time="1s", origin="start")
        .mean("time")
    )


def correct_radar_height(ds_radar, radar_vars, z_grid):
    """Correct radar range gates with HALO flight altitude to height above ground."""

    ds_height = xr.Dataset(coords={'time': ds_radar.time, 'height': z_grid})
    for var in radar_vars:
        ds_height[var] = xr.DataArray(
            data=np.full((len(z_grid), len(ds_radar.time)), np.nan),
            dims=("height", "time"),
            coords={"height": z_grid, "time": ds_radar.time},
        )
    
    for i in tqdm(range(len(radar.time))):
        for var in radar_vars:
            ds_height[var][:, i] = interpolate.interp1d(
                ds_radar["height"].isel(time=i),
                ds_radar[var].isel(time=i),
                fill_value="extrapolate",
            )(z_grid)

    return ds_height


# %% read data
bahamas = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/BAHAMAS/EUREC4A_HALO_BAHAMAS-QL-10Hz_20200218a.nc"
).pipe(pp_bahamas)
# %%
radar = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/Radar/20200218_1013_uncalib_prelim.nc"
).pipe(pp_radar)
radiometer = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/radiometer/183/200218.BRT.NC"
).pipe(pp_radiometer)
radiometer_met = xr.open_dataset(
    "/work/mh0010/EUREC4A/HALO/20200218/radiometer/183/200218.MET.NC"
)

# %% load eureca data for comparison
cat = eurec4a.get_intake_catalog(
    use_ipfs="QmahMN2wgPauHYkkiTGoG2TpPBmj3p5FoYJAq9uE9iXT9N"
)
ds_eureca = cat.HALO.UNIFIED.HAMPradar["HALO-0205"].to_dask(chunks="auto")

# %% unigfy grid
z_grid = np.arange(0, bahamas.IRS_ALT.max() + 30, 30)
time_grid = pd.date_range(
    bahamas.time.min().values, bahamas.time.max().values, freq="1s"
)
gap_length_bahamas = 3000

# reindex bahamas dataset in case there are data gaps
bahamas_reidx = bahamas.reindex(time=time_grid)

#  reindex radiometer data
radiometer_reidx = radiometer.reindex(time=time_grid)

# reindex radar data 
radar_reidx = radar.interp(time=time_grid)

#%% assign height above ground to radar data
height = radar.range * np.cos(np.radians(bahamas_reidx.IRS_THE)) * np.cos(np.radians(bahamas_reidx.IRS_PHI))
radar_reidx["height"] = height

# %% interpolate radar data to height above ground
radar_vars = ["dBZg", "Zg"]
radar_height = correct_radar_height(radar_reidx, radar_vars, z_grid)

# %% filter radar data 
radar_height_filter = radar_height.height < bahamas_reidx.IRS_ALT
radar_height_filtered = radar_height.where(radar_height_filter)
radar_height_filtered['height'] = (radar_height_filtered['height'] - radar_height_filtered['height'].max()) * -1

# %% plot radar data
fig, ax = plt.subplots()
radar_plot = radar_height_filtered.sel(time=slice("2020-02-18 11:20", "2020-02-18 11:50"))
pmesh = ax.pcolormesh(radar_plot.time,
            radar_plot.height,
            radar_plot.dBZg,
            cmap="turbo",
            vmin=-25,
            vmax=25)
ax.set_ylim(0, 5000)
fig.colorbar(pmesh, ax=ax, label="dBZg")

# %% try xarray stuff
ds1 = xr.DataArray(
    data=np.array([1, 2, 3, np.nan, np.nan, 6, 7]),
    coords={"time": pd.date_range("2020-01-01", periods=7, freq="s")},
)
ds2 = xr.DataArray(
    data=np.array([100, 101, 102, 103, 104, 105, 106]),
    coords={"time": pd.date_range("2020-01-01 01:00", periods=7, freq="s")},
)
ds_merged = xr.combine_by_coords([ds1, ds2])
time_grid = pd.date_range(
    ds_merged.time.min().values, ds_merged.time.max().values, freq="1s"
)
time_grid_2 = pd.date_range(
    ds_merged.time.min().values, ds_merged.time.max().values, freq="2s"
)
ds_merged_interp = ds_merged.interp(time=time_grid)
ds_merged_reindex = ds_merged.reindex(time=time_grid)
ds_merged_reindex_2 = ds_merged.reindex(time=time_grid_2)
ds_reindex_interp = ds_merged_reindex.interp(time=time_grid)

# %%
