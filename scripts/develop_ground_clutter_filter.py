# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load data
ds_radar = xr.open_dataset(
    "ipns://latest.orcestra-campaign.org/products/HALO/radar/moments/HALO-20240926a.zarr",
    engine="zarr",
)

# %% plot histogram of dBZg > -30 against height
fig, ax = plt.subplots(figsize=(5, 5))
binary = ds_radar.dBZg > -30
ax.plot(binary.height / 1e3, binary.sum("time"))
ax.set_xlim(0, 0.5)


# %%
def filter_ground_signal(ds):
    """
    Filter radar data for ground reflections.

    If a cell has dBZg > -30 in any of the lowest three height levels and
    dBZg < -30 in the height level above, the cell is considered to be ground reflection.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered for ground reflections.
    """

    # create a binary mask
    binary = ds.dBZg > -30

    # filter all cells with dbZg > -30 in any of the lowest three height levels and dbZg < -30 in the height level above
    mask_height = np.zeros_like(binary)
    for level in range(3):
        mask_height[:, level] = (
            binary.isel(height=level).values > binary.isel(height=level + 1).values
        )
    mask_height[:, 0] = np.sum(mask_height[:, 0:3], axis=1)
    mask_height[:, 1] = np.sum(mask_height[:, 1:3], axis=1)

    # mask all cells below positive maks values
    mask_height = xr.DataArray(
        mask_height < 1,
        dims=["time", "height"],
        coords={"time": ds.time, "height": ds.height},
    )

    # filter all 2D variables
    ds = ds.assign(
        {
            var: ds[var].where(mask_height)
            for var in ds
            if ds[var].dims == ("time", "height")
        }
    )
    return ds


# %%
ds = filter_ground_signal(ds_radar)

# %% plot filtered vs unfiltered data
timeslice = slice("2024-09-26 12:42", "2024-09-26 12:45")
ds_radar_plot = ds_radar.sel(time=timeslice, height=slice(0, 3e3))
ds_radar_plot_ground = ds_radar.sel(time=timeslice, height=slice(0, 0.2e3))
ds_plot = ds.sel(time=timeslice, height=slice(0, 0.2e3))

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
pcol = axes[0].pcolormesh(
    ds_radar_plot.time,
    ds_radar_plot.height,
    ds_radar_plot.dBZg.T,
    cmap="YlGnBu",
    vmin=-30,
    vmax=30,
)

pcol = axes[1].pcolormesh(
    ds_radar_plot_ground.time,
    ds_radar_plot_ground.height,
    ds_radar_plot_ground.dBZg.T,
    cmap="YlGnBu",
    vmin=-30,
    vmax=30,
)

pcol = axes[2].pcolormesh(
    ds_plot.time,
    ds_plot.height,
    ds_plot.dBZg.T,
    cmap="YlGnBu",
    vmin=-30,
    vmax=30,
)

clab, extend, shrink = "Z /dBZg", "max", 0.8
fig.colorbar(pcol, ax=axes, label=clab, extend=extend, shrink=shrink)
axes[2].set_xlabel("Time")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Height / m")

fig.savefig("quicklooks/gound_filter.png", bbox_inches="tight", dpi=300)

# %%
