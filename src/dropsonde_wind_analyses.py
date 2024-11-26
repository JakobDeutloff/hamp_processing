import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

from . import plot_functions as plotfuncs


def get_dropsondes_within_heights(ds, height_min, height_max):
    """returns slice of dataset with alt within ht1 <= alt < ht2"""
    ht_min = ds.alt.sel(alt=height_min, method="nearest").values
    ht_max = ds.alt.sel(alt=height_max, method="nearest").values

    ds_heightslice = ds.where(ds.alt < ht_max, drop=True)
    ds_heightslice = ds.where(ds_heightslice.alt >= ht_min, drop=True)

    return ht_min, ht_max, ds_heightslice


def horizontal_wind_speed(northward, eastward):
    return np.sqrt(eastward * eastward + northward * northward)


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


def plot_coloured_dropsonde_vertical_profile(
    ax, verticaldata, height, vmin, vmax, colorby, cmap, axtitle=None, xlabel=None
):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    color = cmap(norm(colorby))

    ax.scatter(verticaldata, height, marker=".", s=1, color=color)

    ax.set_title(axtitle)
    ax.set_ylabel("height / km")
    ax.set_xlabel(xlabel)

    return ax, norm, cmap


def plot_dropsonde_wind_vertical_profiles(
    ds_dropsonde, colorby, figsize=(16, 9), cmap="Blues", cbarlab=None
):
    fig, axes = plt.subplots(
        nrows=1, ncols=5, figsize=figsize, width_ratios=[1, 1, 1, 1, 1 / 27]
    )

    height = np.tile(ds_dropsonde.alt / 1000, ds_dropsonde.sonde_id.size)  # [km]
    colorby = ds_dropsonde[colorby].values.flatten()
    vmin, vmax = np.nanmin(colorby), np.nanmax(colorby)

    eastward = ds_dropsonde.u.values
    ax, norm1, cmap1 = plot_coloured_dropsonde_vertical_profile(
        axes[0],
        eastward.flatten(),
        height,
        vmin,
        vmax,
        colorby,
        cmap,
        axtitle="Eastward",
        xlabel="u /m s$^{-1}$",
    )

    northward = ds_dropsonde.v.values
    ax, norm1, cmap1 = plot_coloured_dropsonde_vertical_profile(
        axes[1],
        northward.flatten(),
        height,
        vmin,
        vmax,
        colorby,
        cmap,
        axtitle="Northward",
        xlabel="v /m s$^{-1}$",
    )

    direction = horizontal_wind_direction(northward, eastward)
    ax, norm1, cmap1 = plot_coloured_dropsonde_vertical_profile(
        axes[2],
        direction.flatten(),
        height,
        vmin,
        vmax,
        colorby,
        cmap,
        axtitle="Direction from Westerlies",
        xlabel="$\u03c6$ /degrees",
    )

    magnitude = horizontal_wind_speed(northward, eastward)
    ax, norm1, cmap1 = plot_coloured_dropsonde_vertical_profile(
        axes[3],
        magnitude.flatten(),
        height,
        vmin,
        vmax,
        colorby,
        cmap,
        axtitle="Horizontal Wind Speed",
        xlabel="|V$_{xy}$| /m s$^{-1}$",
    )

    cax = fig.colorbar(
        ScalarMappable(norm=norm1, cmap=cmap1), cax=axes[4], label=cbarlab, shrink=0.8
    )

    for ax in axes[1:4]:
        ax.sharey(axes[0])
        ax.set_ylabel("")
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

    ax.set_title(axtitle)

    xticks = np.linspace(lonmin, lonmax, 3)
    yticks = np.linspace(latmin, latmax, 3)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    return ax


def plot_mean_wind_quiver_on_projection(
    ax,
    ds_dropsonde,
    height_min,
    height_max,
    lonmin=-35,
    lonmax=-15,
    latmin=0,
    latmax=20,
):
    """plot wind quivers on projection but for mean wind between height_min and height_max"""
    ht_min, ht_max, ds2mean = get_dropsondes_within_heights(
        ds_dropsonde, height_min, height_max
    )
    mean_lon = ds2mean.lon.mean(dim="alt")
    mean_lat = ds2mean.lat.mean(dim="alt")
    mean_eastward = ds2mean.u.mean(dim="alt")
    mean_northward = ds2mean.v.mean(dim="alt")

    axtitle = f"{ht_min/1000}km <= GPS Altitude < {ht_max/1000}km"
    plot_wind_quiver_on_projection(
        ax,
        mean_lon,
        mean_lat,
        mean_eastward,
        mean_northward,
        axtitle=axtitle,
        lonmin=lonmin,
        lonmax=lonmax,
        latmin=latmin,
        latmax=latmax,
    )

    return ax, mean_lon, mean_lat, mean_eastward, mean_northward
