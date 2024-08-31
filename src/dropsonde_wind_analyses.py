import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

from . import plot_functions as plotfuncs


def get_dropsondes_within_heights(ds, height_min, height_max):
    """returns slice of dataset with gpsalt within ht1 <= gpsalt < ht2"""
    ht_min = ds.gpsalt.sel(gpsalt=height_min, method="nearest").values
    ht_max = ds.gpsalt.sel(gpsalt=height_max, method="nearest").values

    ds_heightslice = ds.where(ds.gpsalt < ht_max, drop=True)
    ds_heightslice = ds.where(ds_heightslice.gpsalt >= ht_min, drop=True)

    return ht_min, ht_max, ds_heightslice


def horizontal_wind_direction(northward, eastward, bearing=False):
    """returns horizontal wind direction in degrees in range
    -180 <= direction <=180 relative to westerly winds (i.e. the vector
    from (0,0) to (1,0)]). If bearing is true (False by default), direction
    is convered to a bearing before return, i.e. the direction relative
    to the vector from (0,0) to (0,1) is returned instead."""
    if bearing:
        direction = np.arctan2(eastward, northward) * 180 / np.pi
        direction = np.where(
            direction < 0, direction + 360, direction
        )  # [bearing from north]
    else:
        direction = np.arctan2(northward, eastward) * 180 / np.pi

    return direction


def plot_dropsonde_wind_vertical_profiles(
    ds_dropsonde, colorby, figsize=(21, 12), cmap="Blues", cbarlab=None
):
    fig, axes = plt.subplots(
        nrows=1, ncols=5, figsize=figsize, width_ratios=[1, 1, 1, 1, 1 / 27]
    )

    height = np.tile(ds_dropsonde.gpsalt / 1000, ds_dropsonde.sonde_id.size)  # [km]
    eastward = ds_dropsonde.u.values.flatten()
    northward = ds_dropsonde.v.values.flatten()
    direction = horizontal_wind_direction(eastward, northward)
    magnitude = np.sqrt(eastward * eastward + northward * northward)

    norm = mcolors.Normalize(vmin=colorby.min(), vmax=colorby.max())
    cmap = plt.get_cmap(cmap)
    color = cmap(norm(colorby.values.flatten()))

    axes[0].set_title("Eastward")
    axes[0].scatter(eastward, height, marker=".", s=1, color=color)
    axes[0].set_xlabel("u /m s$^{-1}$")

    axes[1].set_title("Northward")
    axes[1].scatter(northward, height, marker=".", s=1, color=color)
    axes[1].set_xlabel("v /m s$^{-1}$")

    axes[2].set_title("Direction")
    axes[2].scatter(direction.T, height, marker=".", s=1, color=color)
    axes[2].set_xlabel("$\u03C6$ /degrees")

    axes[3].set_title("Magnitude")
    axes[3].scatter(magnitude.T, height, marker=".", s=1, color=color)
    axes[3].set_xlabel("|V$_{xy}$| /m s$^{-1}$")

    cax = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=axes[4], label=cbarlab, shrink=0.8
    )

    for ax in axes[1:3]:
        ax.sharey(axes[0])
    axes[0].set_ylabel("height / km")
    plotfuncs.beautify_axes(axes)
    plotfuncs.beautify_colorbar_axes(cax)

    return fig, axes


def plot_wind_quiver_on_projection(
    ax,
    lon,
    lat,
    eastward,
    northward,
    axtitle=None,
    lonmin=-35,
    lonmax=-15,
    latmin=0,
    latmax=20,
):
    ax.coastlines()
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    ax.quiver(
        lon,
        lat,
        eastward,
        northward,
        angles="xy",
        scale_units="xy",
        scale=5,
        color="k",
        label="Arrows",
        headlength=2,
        headwidth=2,
        headaxislength=2,
    )

    if axtitle:
        ax.set_title(axtitle)

    xticks = np.linspace(lonmin, lonmax, 3)
    yticks = np.linspace(latmin, latmax, 3)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    return ax
