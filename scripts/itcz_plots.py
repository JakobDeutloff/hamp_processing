# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from src import readwrite_functions as rwfuncs
from src import plot_functions as plotfuncs
from src import itcz_functions as itczfuncs
from src.plot_quicklooks import save_figure


def plot_itcz_masked_radar_timeseries_and_composite(
    hampdata,
    itcz_mask_signal,
    figsize=(12, 9),
    savefigparams=[],
):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=figsize,
        width_ratios=[18, 7],
        sharey="row",
        sharex="col",
    )

    time_radar = hampdata.radar.time
    height_km = hampdata.radar.height / 1e3  # [km]
    signal = plotfuncs.filter_radar_signal(hampdata.radar.dBZg, threshold=-30)  # [dBZ]

    nrep = len(height_km)
    itcz_mask_signal = np.repeat(itcz_mask_signal, nrep)
    itcz_mask_signal = np.reshape(itcz_mask_signal.values, [len(time_radar), nrep])

    mask_values = {0: "Outside", 1: "Transition", 2: "Inside"}

    for a in mask_values.keys():
        ax0, ax1 = axs[a, 0], axs[a, 1]

        signal_plt = np.where(itcz_mask_signal == a, signal, np.nan)

        ax1.set_title(f"{mask_values[a]}", loc="left")
        cax = plotfuncs.plot_radardata_timeseries(
            time_radar, height_km, signal_plt.T, fig, ax0
        )[1]
        ax0.set_xlabel("")

        signal_range = [-30, 30]
        height_range = [0, np.nanmax(height_km)]
        signal_bins = 60
        height_bins = 100
        cmap = plotfuncs.get_greys_histogram_colourmap(signal_bins)
        plotfuncs.plot_radardata_histogram(
            time_radar,
            height_km,
            signal_plt,
            ax1,
            signal_range,
            height_range,
            height_bins,
            signal_bins,
            cmap,
        )
        axs[a, 1].set_ylabel("")

    timeframe = slice(hampdata.radar.time[0], hampdata.radar.time[-1])
    plotfuncs.add_lat_lon_axes(hampdata, timeframe, axs[2, 0])
    axs[2, 0].set_xlabel("UTC")
    axs[2, 1].set_xlabel("Z /dBZe")
    plotfuncs.beautify_axes(axs.flatten())
    plotfuncs.beautify_colorbar_axes(cax)

    fig.tight_layout()

    if savefigparams != []:
        save_figure(fig, savefigparams=[savefig_format, savename, dpi])


# %% load config and create HAMP post-processed data
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = sys.argv[1]
cfg = rwfuncs.extract_config_params(configfile)
path_saveplts = cfg["path_saveplts"]
flightname = cfg["flightname"]
### ------------------------------------------------------------- ###
# from src import load_data_functions as loadfuncs
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
hampdata = rwfuncs.load_timeslice_all_level1hampdata(
    cfg["path_hampdata"], cfg["is_planet"]
)


# %% Plot CWV and radar with ITCZ mask
savefig_format = "png"
savename = path_saveplts / f"itcz_mask_radar_column_water_path_{flightname}.png"
dpi = 64
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(12, 6), width_ratios=[18, 7], sharey="row"
)
fig, axes = plotfuncs.plot_radar_cwv_timeseries(fig, axes, hampdata)
axes[1, 0].legend(loc="upper right", frameon=False)
itcz_mask_1 = itczfuncs.identify_itcz_crossings(hampdata["CWV"]["IWV"])
itczfuncs.add_itcz_mask(fig, axes[1, 0], hampdata["CWV"].time, itcz_mask_1)
itcz_mask_2 = itczfuncs.interpolate_radiometer_mask_to_radar_mask(itcz_mask_1, hampdata)
itczfuncs.add_itcz_mask(fig, axes[0, 0], hampdata.radar.time, itcz_mask_2, cbar=False)
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

# %% Plot radar timeseries and histogram for each masked area
savefig_format = "png"
savename = path_saveplts / f"itcz_mask_radar_composite_{flightname}.png"
dpi = 64
plot_itcz_masked_radar_timeseries_and_composite(
    hampdata, itcz_mask_2, savefigparams=[savefig_format, savename, dpi]
)
