# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
from src import readwrite_functions as rwfuncs
from src import load_data_functions as loadfuncs
from src import plot_functions as plotfuncs
from src.plot_quicklooks import save_figure


# %% function definitions
def horizontal_wind_direction(eastward, northward, bearing=False):
    """returns horizontal wind dircetion in degrees in range
    -180 <= direction <=180 relative to vector from (0,0) to (1,0)].
    If bearing is true (False by default), direction is convered to a
    bearing before return, i.e. the direction relative to the
    vector from (0,0) to (0,1) is returned instead."""
    direction = np.arctan2(eastward, northward) * 180 / np.pi
    if bearing:
        direction = np.where(
            direction < 0, direction + 360, direction
        )  # [bearing from north]
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
    magnitude = np.sqrt(eastward * eastward, northward * northward)

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


def get_dataset_within_heights(ds, height_min, height_max):
    """returns slice of dataset with gpsalt within ht1 <= gpsalt < ht2"""
    ht_min = ds.gpsalt.sel(gpsalt=height_min, method="nearest").values
    ht_max = ds.gpsalt.sel(gpsalt=height_max, method="nearest").values

    ds_heightslice = ds.where(ds.gpsalt < ht_max, drop=True)
    ds_heightslice = ds.where(ds_heightslice.gpsalt >= ht_min, drop=True)

    return ht_min, ht_max, ds_heightslice


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
    def plot_wind_quiver(ax, lon, lat, eastward, northward):
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

    ax.coastlines()
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    plot_wind_quiver(ax, lon, lat, eastward, northward)

    if axtitle:
        ax.set_title(axtitle)

    xticks = np.linspace(lonmin, lonmax, 3)
    yticks = np.linspace(latmin, latmax, 3)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    return ax


def plot_mean_wind_quiver_between_heights(
    ds_dropsonde, heights, figsize=(15, 9), lonmin=-35, lonmax=-15, latmin=0, latmax=20
):
    nrows = 2
    ncols = int(len(heights) // nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    axes = axes.flatten()
    for n in range(1, len(heights)):
        ht_min, ht_max, ds2mean = get_dataset_within_heights(
            ds_dropsonde, heights[n - 1], heights[n]
        )
        mean_lon = ds2mean.lon.mean(dim="gpsalt")
        mean_lat = ds2mean.lat.mean(dim="gpsalt")
        mean_eastward = ds2mean.u.mean(dim="gpsalt")
        mean_northward = ds2mean.v.mean(dim="gpsalt")

        axtitle = f"{ht_min/1000}km <= GPS Altitude < {ht_max/1000}km"
        plot_wind_quiver_on_projection(
            axes[n - 1],
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

    for ax in axes[len(heights) - 1 : len(axes)]:
        ax.remove()

    return fig, axes


def plot_mean_wind_seperate_east_north_quiver(
    ds_dropsonde,
    height_min,
    height_max,
    figsize=(16, 4),
    lonmin=-35,
    lonmax=-15,
    latmin=0,
    latmax=20,
):
    def plot_wind_component_scatter(ax, wind_component, latitude, xlab=None, fit=False):
        ax.scatter(wind_component, latitude, marker="x", color="k")
        ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlab)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
    fig.delaxes(axes[0])
    axes[0] = fig.add_subplot(131, projection=ccrs.PlateCarree())

    ht_min, ht_max, ds2mean = get_dataset_within_heights(
        ds_dropsonde, height_min, height_max
    )
    mean_lon = ds2mean.lon.mean(dim="gpsalt")
    mean_lat = ds2mean.lat.mean(dim="gpsalt")
    mean_eastward = ds2mean.u.mean(dim="gpsalt")
    mean_northward = ds2mean.v.mean(dim="gpsalt")

    plot_wind_quiver_on_projection(
        axes[0],
        mean_lon,
        mean_lat,
        mean_eastward,
        mean_northward,
        axtitle=None,
        lonmin=lonmin,
        lonmax=lonmax,
        latmin=latmin,
        latmax=latmax,
    )
    axes[0].set_ylabel("Latitude /$\u00B0$")
    axes[0].set_xlabel("Longitude /$\u00B0$")

    xlab = "Nothward wind, v / m s$^{-1}$"
    plot_wind_component_scatter(axes[1], mean_northward, mean_lat, xlab=xlab, fit=True)

    xlab = "Eastward wind, u / m s$^{-1}$"
    plot_wind_component_scatter(axes[2], mean_eastward, mean_lat, xlab=xlab, fit=False)

    yticks = np.linspace(latmin, latmax, 3)
    for ax in [axes[1], axes[2]]:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
    plotfuncs.beautify_axes(axes)

    title = f"{ht_min/1000}km <= GPS Altitude < {ht_max/1000}km"
    fig.suptitle(title, fontsize=15)

    return fig, axes


# %% load config and create HAMP post-processed data
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = sys.argv[1]
cfg = rwfuncs.extract_config_params(configfile)
path_saveplts = cfg["path_saveplts"]
flightname = cfg["flightname"]
### ------------------------------------------------------------- ###
ds = loadfuncs.load_dropsonde_data_for_date(cfg["path_dropsonde_level3"], cfg["date"])
# hampdata = loadfuncs.do_post_processing(
#     cfg["path_bahamas"],
#     cfg["path_radar"],
#     cfg["path_radiometer"],
#     cfg["radiometer_date"],
#     is_planet=cfg["is_planet"],
#     do_radar=True,
#     do_183=True,
#     do_11990=True,
#     do_kv=True,
#     do_cwv=True,
# )

# %% Plot Various Vertical Wind Profiles
latmax = 15
ds_dropsonde = ds.where(ds.lat < latmax, drop=True)

fig, axes = plot_dropsonde_wind_vertical_profiles(
    ds_dropsonde,
    ds_dropsonde.lat,
    figsize=(21, 12),
    cmap="plasma",
    cbarlab="latitude /$\u00B0$",
)
savefig_format = "png"
savename = (
    path_saveplts / f"dropsonde_vertical_wind_profiles_{flightname}_colorlatitutde.png"
)
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

fig, axes = plot_dropsonde_wind_vertical_profiles(
    ds_dropsonde,
    ds_dropsonde.rh,
    figsize=(21, 12),
    cmap="Blues",
    cbarlab="relative humidity Kg/Kg",
)
savefig_format = "png"
savename = (
    path_saveplts / f"dropsonde_vertical_wind_profiles_{flightname}_colorrelh.png"
)
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

# %% Plot Various Mean Winds Between Heights
heights = [0, 2000, 4000, 6000, 8000, 10000, 12000]
fig, axes = plot_mean_wind_quiver_between_heights(
    ds_dropsonde, heights, figsize=(15, 9)
)
savefig_format = "png"
savename = path_saveplts / f"dropsonde_wind_slices_{flightname}_colorrelh.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

for h in range(1, len(heights)):
    height_min, height_max = heights[h - 1], heights[h]
    fig, axes = plot_mean_wind_seperate_east_north_quiver(
        ds_dropsonde,
        height_min,
        height_max,
        figsize=(16, 4),
        lonmin=-35,
        lonmax=-15,
        latmin=0,
        latmax=20,
    )
plt.show()
