# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from src import readwrite_functions as rwfuncs
from src import load_data_functions as loadfuncs
from src import plot_functions as plotfuncs
from src import dropsonde_wind_analyses as dropfuncs
from src.plot_quicklooks import save_figure
from matplotlib.cm import ScalarMappable


# %% helper function definitions
def mean_wind_quivers_plot(
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
        dropfuncs.plot_mean_wind_quiver_on_projection(
            axes[n - 1],
            ds_dropsonde,
            heights[n - 1],
            heights[n],
            lonmin=lonmin,
            lonmax=lonmax,
            latmin=latmin,
            latmax=latmax,
        )

    for ax in axes[len(heights) - 1 : len(axes)]:
        ax.remove()

    return fig, axes


def mean_winds_seperate_east_north_scatter(
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

    (
        ax,
        mean_lon,
        mean_lat,
        mean_eastward,
        mean_northward,
    ) = dropfuncs.plot_mean_wind_quiver_on_projection(
        axes[0],
        ds_dropsonde,
        height_min,
        height_max,
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

    return fig, axes


def plot_allflights_wind_vertical_profile(
    ds_full,
    dates,
    calc_verticaldata,
    latmax,
    cmap,
    vmin,
    vmax,
    hmin,
    hmax,
    xlabel,
    figtitle,
):
    fig, axes = plt.subplots(
        nrows=1, ncols=7, figsize=(21, 9), width_ratios=[1] * 6 + [1 / 27]
    )

    for d, date in enumerate(dates):
        ds_dropsonde = loadfuncs.load_dropsonde_data_for_date(ds_full, date)
        ds_dropsonde = ds_dropsonde.where(ds_dropsonde.lat < latmax, drop=True)
        height = np.tile(ds_dropsonde.gpsalt / 1000, ds_dropsonde.sonde_id.size)  # [km]
        colorby = ds_dropsonde["lat"].values.flatten()

        ax, norm1, cmap1 = dropfuncs.plot_coloured_dropsonde_vertical_profile(
            axes[d],
            calc_verticaldata(ds_dropsonde),
            height,
            vmin,
            vmax,
            colorby,
            cmap,
            axtitle=None,
            xlabel=xlabel,
        )
        ax.set_title(f"HALO-{date}a", fontsize=15)
        ax.set_ylim(hmin, hmax)

    for ax in axes[1:6]:
        ax.sharey(axes[0])
        ax.sharex(axes[0])
        ax.set_ylabel("")

    fig.suptitle(figtitle, fontsize=15)
    cax = fig.colorbar(
        ScalarMappable(norm=norm1, cmap=cmap1),
        cax=axes[6],
        label="latitude /$\u00B0$",
        shrink=0.8,
    )
    plotfuncs.beautify_axes(axes)
    plotfuncs.beautify_colorbar_axes(cax)

    return fig, axes


# %% load config and create HAMP post-processed data
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = sys.argv[1]
cfg = rwfuncs.extract_config_params(configfile)
path_saveplts = cfg["path_saveplts"]
flightname = cfg["flightname"]
### ------------------------------------------------------------- ###
# ds = loadfuncs.load_dropsonde_data_for_date(cfg["path_dropsonde_level3"], cfg["date"])
# # hampdata = loadfuncs.do_post_processing(
# #     cfg["path_bahamas"],
# #     cfg["path_radar"],
# #     cfg["path_radiometer"],
# #     cfg["radiometer_date"],
# #     is_planet=cfg["is_planet"],
# #     do_radar=True,
# #     do_183=True,
# #     do_11990=True,
# #     do_kv=True,
# #     do_cwv=True,
# # )

# # %% Plot Vertical Wind Profiles Coloured by Latitude
# latmax = 15
# ds_dropsonde = ds.where(ds.lat < latmax, drop=True)

# fig, axes = dropfuncs.plot_dropsonde_wind_vertical_profiles(
#     ds_dropsonde,
#     "lat",
#     figsize=(21, 12),
#     cmap="plasma",
#     cbarlab="latitude /$\u00B0$",
# )
# savefig_format = "png"
# savename = (
#     path_saveplts / f"dropsonde_vertical_wind_profiles_{flightname}_colorlatitutde.png"
# )
# dpi = 64
# save_figure(fig, savefigparams=[savefig_format, savename, dpi])

# # %% Plot Vertical Wind Profiles Coloured by RelH
# fig, axes = dropfuncs.plot_dropsonde_wind_vertical_profiles(
#     ds_dropsonde,
#     "rh",
#     figsize=(21, 12),
#     cmap="Blues",
#     cbarlab="relative humidity Kg/Kg",
# )
# savefig_format = "png"
# savename = (
#     path_saveplts / f"dropsonde_vertical_wind_profiles_{flightname}_colorrelh.png"
# )
# dpi = 64
# save_figure(fig, savefigparams=[savefig_format, savename, dpi])

# # %% Plot Mean Winds Between Heights
# heights = [0, 2000, 4000, 6000, 8000, 10000, 12000]
# fig, axes = mean_wind_quivers_plot(ds_dropsonde, heights, figsize=(15, 9))
# savefig_format = "png"
# savename = path_saveplts / f"dropsonde_wind_slices_{flightname}_colorrelh.png"
# dpi = 64
# save_figure(fig, savefigparams=[savefig_format, savename, dpi])

# # %% Plot Mean Winds Between Height with scatter for components
# for h in range(1, len(heights)):
#     height_min, height_max = heights[h - 1], heights[h]
#     fig, axes = mean_winds_seperate_east_north_scatter(
#         ds_dropsonde,
#         height_min,
#         height_max,
#         figsize=(16, 4),
#         lonmin=-35,
#         lonmax=-15,
#         latmin=0,
#         latmax=20,
#     )
# plt.show()

# %% Plot Vertical Wind Profile for ALl Flights Coloured by Latitude
latmax = 15
dates = ["20240811", "20240813", "20240816", "20240818", "20240821", "20240822"]
vmin, vmax = 5, 12
hmin, hmax = -0.5, 15

ds_full = loadfuncs.load_dropsonde_data(cfg["path_dropsonde_level3"])
figtitle = "Eastward Wind Component"
xlabel = "u /m s$^{-1}$"
cmap = "plasma"


def calc_verticaldata(ds_dropsonde):
    return ds_dropsonde.u.values.flatten()


fig, axes = plot_allflights_wind_vertical_profile(
    ds_full,
    dates,
    calc_verticaldata,
    latmax,
    cmap,
    vmin,
    vmax,
    hmin,
    hmax,
    xlabel,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertical_eastward_profile_colorlatitutde.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])
