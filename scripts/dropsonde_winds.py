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
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


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
# latmax = 20
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


# %% Define more functions for plotting all flights
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
    xlims,
    figtitle,
):
    nd = len(dates)
    fig, axes = plt.subplots(
        nrows=1, ncols=nd + 1, figsize=(21, 9), width_ratios=[1] * nd + [1 / 27]
    )

    for d, date in enumerate(dates):
        ds_dropsonde = loadfuncs.load_dropsonde_data_for_date(ds_full, date)
        ds_dropsonde = ds_dropsonde.where(ds_dropsonde.lat < latmax, drop=True)
        height = np.tile(ds_dropsonde.alt / 1000, ds_dropsonde.sonde_id.size)  # [km]
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
        ax.set_xlim(xlims)
        xticks = np.linspace(xlims[0], xlims[1], 5)
        yticks = np.linspace(0, 15, 5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

    for ax in axes[1:nd]:
        ax.sharey(axes[0])
        ax.sharex(axes[0])
        ax.set_ylabel("")
        ax.set_yticklabels("")
    axes[0].set_yticklabels(yticks)

    fig.suptitle(figtitle, fontsize=21)
    cax = fig.colorbar(
        ScalarMappable(norm=norm1, cmap=cmap1),
        cax=axes[nd],
        label="latitude /$\u00B0$",
        shrink=0.8,
        extend="both",
    )
    plotfuncs.beautify_axes(axes)
    plotfuncs.beautify_colorbar_axes(cax)

    return fig, axes


# %% Plot Vertical Wind Profile for ALl Flights Coloured by Latitude
latmax = 20
dates = ["20240811", "20240813", "20240816", "20240818", "20240821", "20240822"]
vmin, vmax = 5, 12
hmin, hmax = -0.5, 15
cmap = "plasma"
ds_full = loadfuncs.load_dropsonde_data(cfg["path_dropsonde_level3"])


# %% Vertical line profile plots
figtitle = "Eastward Wind Component"
xlabel = "u /m s$^{-1}$"
xlims = [-20, 20]


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
    xlims,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertical_eastward_profile_colorlatitutde.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

figtitle = "Northward Wind Component"
xlabel = "v /m s$^{-1}$"
xlims = [-10, 10]


def calc_verticaldata(ds_dropsonde):
    return ds_dropsonde.v.values.flatten()


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
    xlims,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertical_northward_profile_colorlatitutde.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

figtitle = "Direction Relative to Westerlies (->)"
xlabel = "$\u03C6$ /degrees"
xlims = [-180, 180]


def calc_verticaldata(ds_dropsonde):
    eastward = ds_dropsonde.u.values.flatten()
    northward = ds_dropsonde.v.values.flatten()
    return dropfuncs.horizontal_wind_direction(northward, eastward)


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
    xlims,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertical_direction_profile_colorlatitutde.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

figtitle = "Horizontal Wind Speed"
xlabel = "|V$_{xy}$| /m s$^{-1}$"
xlims = [0, 30]


def calc_verticaldata(ds_dropsonde):
    eastward = ds_dropsonde.u.values.flatten()
    northward = ds_dropsonde.v.values.flatten()
    return dropfuncs.horizontal_wind_speed(northward, eastward)


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
    xlims,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertical_magnitude_profile_colorlatitutde.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])


# %% height-latitude plots
def plot_allflights_vertical_vs_latitude_wind(
    ds_full,
    dates,
    colorby_data,
    latmin,
    latmax,
    hmin,
    hmax,
    cmap,
    cbarlabel,
    extend,
    figtitle,
    figsize=(16, 9),
):
    def plot_wind_component_scatter(ax, ds_dropsonde, colorby_data, norm, cmap):
        height = np.tile(ds_dropsonde.alt / 1000, ds_dropsonde.sonde_id.size)  # [km]
        latitude = ds_dropsonde.lat.values.flatten()
        color = cmap(norm(colorby_data(ds_dropsonde)))

        ax.scatter(latitude, height, marker=",", color=color)
        ax.set_xlabel("Latitude /$\u00B0$")
        ax.set_ylabel("Height /km")

        return norm, cmap

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=2, ncols=4, width_ratios=[1, 1, 1, 1 / 27])
    axes = []
    for r in range(2):
        for c in range(3):
            axes.append(fig.add_subplot(gs[r, c]))
    cax = fig.add_subplot(gs[:, 3])

    cmap = plt.get_cmap(cmap)
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    for d, date in enumerate(dates):
        ds_dropsonde = loadfuncs.load_dropsonde_data_for_date(ds_full, date)
        plot_wind_component_scatter(axes[d], ds_dropsonde, colorby_data, norm, cmap)

        axes[d].set_title(f"HALO-{date}a", fontsize=15)
        axes[d].set_xlabel("")
        axes[d].set_ylabel("")
        xticks = np.linspace(latmin, latmax, 5)
        yticks = np.linspace(0, 15, 5)
        axes[d].set_xticks(xticks)
        axes[d].set_yticks(yticks)
        axes[d].set_yticklabels("")
        axes[d].set_xticklabels("")

        axes[d].set_xlim(latmin, latmax)
        axes[d].set_ylim(hmin, hmax)
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel("Height /km")
        ax.set_yticklabels(yticks)
    for ax in axes[3:]:
        ax.set_xlabel("Latitude /$\u00B0$")
        ax.set_xticklabels(xticks)

    cax = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        label=cbarlabel,
        shrink=0.8,
        extend=extend,
    )
    plotfuncs.beautify_axes(axes)
    plotfuncs.beautify_colorbar_axes(cax)

    fig.suptitle(figtitle, fontsize=18)

    return fig, axes


dates = ["20240811", "20240813", "20240816", "20240818", "20240821", "20240822"]


def colorby_data(ds_dropsonde):
    return ds_dropsonde.v.values.flatten()  # northward


latmin, latmax = 0, 20
hmin, hmax = -0.5, 15
cmap = "coolwarm"
figtitle = "Northward Wind Component"
cbarlabel = "Northward, v /m s$^{-1}$"
levels = [-25, -5, -2.5, 2.5, 5, 25]
extend = "both"
fig, axes = plot_allflights_vertical_vs_latitude_wind(
    ds_full,
    dates,
    colorby_data,
    latmin,
    latmax,
    hmin,
    hmax,
    cmap,
    cbarlabel,
    extend,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertvslat_northward.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])


def colorby_data(ds_dropsonde):
    return ds_dropsonde.u.values.flatten()  # eastward


latmin, latmax = 0, 20
hmin, hmax = -0.5, 15
cmap = "coolwarm"
figtitle = "Eastward Wind Component"
cbarlabel = "Eastward, v /m s$^{-1}$"
levels = [-25, -5, -2.5, 2.5, 5, 25]
extend = "both"
fig, axes = plot_allflights_vertical_vs_latitude_wind(
    ds_full,
    dates,
    colorby_data,
    latmin,
    latmax,
    hmin,
    hmax,
    cmap,
    cbarlabel,
    extend,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertvslat_eastward.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])


def colorby_data(ds_dropsonde):
    eastward = ds_dropsonde.u.values.flatten()
    northward = ds_dropsonde.v.values.flatten()
    return dropfuncs.horizontal_wind_direction(northward, eastward)


latmin, latmax = 0, 20
hmin, hmax = -0.5, 15
cmap = "twilight"
figtitle = "Direction Relative to Westerlies (->)"
cbarlabel = "$\u03C6$ /degrees"
levels = [-180, -90, -60, -30, 30, 60, 90, 180]
extend = None
fig, axes = plot_allflights_vertical_vs_latitude_wind(
    ds_full,
    dates,
    colorby_data,
    latmin,
    latmax,
    hmin,
    hmax,
    cmap,
    cbarlabel,
    extend,
    figtitle,
)
savefig_format = "png"
savename = path_saveplts / "allflights_vertvslat_direction.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])
