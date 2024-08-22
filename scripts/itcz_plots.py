# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.colors as mcolors
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from src import helper_functions as helpfuncs
import yaml
import matplotlib.pyplot as plt
from src import plot_functions as plotfuncs
from src.plot_quicklooks import save_figure

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configyaml = sys.argv[1]
with open(configyaml, "r") as file:
    print(f"Reading config YAML: '{configyaml}'")
    cfg = yaml.safe_load(file)
path_hampdata = cfg["paths"]["hampdata"]
hampdata = helpfuncs.load_timeslice_all_level1hampdata(path_hampdata, cfg["is_planet"])
### ------------------------------------------------------------- ###


# %% fuunctions for plotting HAMP post-processed data slice
def plot_radar_cwv_timeseries(
    hampdata,
    figsize=(9, 5),
    savefigparams=[],
):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=figsize, width_ratios=[18, 7], sharey="row"
    )

    cax = plotfuncs.plot_radar_timeseries(hampdata.radar, fig, axs[0, 0])[1]
    axs[0, 0].set_title("  Timeseries", fontsize=18, loc="left")

    plotfuncs.plot_beautified_radar_histogram(hampdata.radar, axs[0, 1])
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_title("Histogram", fontsize=18)

    plotfuncs.plot_column_water_vapour_timeseries(
        hampdata["CWV"]["IWV"], axs[1, 0], target_cwv=48
    )

    plotfuncs.beautify_axes(axs.flatten())
    plotfuncs.beautify_colorbar_axes(cax)
    axs[1, 1].remove()

    fig.tight_layout()

    if savefigparams != []:
        save_figure(fig, savefigparams)

    return fig, axs


def identify_itcz_crossings(cwv, thresholds=[45, 50]):
    """
    returns 'boolean-ish' type for whether CWV is inside/outside ITCZ:
    CWV < outside_itcz_threshold, value = 0;
    outside_itcz_threshold < CWV <= inside_itcz_threshold = 1;
    CWV > inside_itcz_threshold = or 2.
    """
    # create mask
    itcz_mask = np.where(cwv < thresholds[0], 0, 1)
    itcz_mask = np.where(cwv > thresholds[1], 2, itcz_mask)
    itcz_mask = np.where(np.isnan(cwv), np.nan, itcz_mask)

    # return in dataset with time dimension
    ds_mask1 = xr.Dataset(
        {
            "itcz_mask": xr.DataArray(
                itcz_mask, dims=hampdata["CWV"].dims, coords=hampdata["CWV"].coords
            )
        }
    )

    return ds_mask1


def add_itcz_mask(ax, xtime, itcz_mask, cbar=True):
    colors = ["red", "gold", "green"]  # Red, Green, Blue
    cmap = LinearSegmentedColormap.from_list("three_color_cmap", colors)
    levels = [-0.5, 0.5, 1.5, 2.5]

    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 2)
    xx, yy = np.meshgrid(xtime, y)
    z = np.array([itcz_mask, itcz_mask])

    cont = ax.contourf(
        xx,
        yy,
        z,
        levels=levels,
        cmap=cmap,
        alpha=0.2,
    )
    clab = "ITCZ Mask"
    if cbar:
        cbar = fig.colorbar(cont, ax=ax, label=clab, shrink=0.8)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Outside", "Transition", "Inside"])


# %% plot with mask on CWV and radar
fig, axes = plot_radar_cwv_timeseries(hampdata, figsize=(12, 6), savefigparams=[])
axes[1, 0].legend(loc="upper right", frameon=False)
ds_mask1 = identify_itcz_crossings(hampdata["CWV"]["IWV"])
add_itcz_mask(axes[1, 0], hampdata["CWV"].time, ds_mask1.itcz_mask)
ds_mask2 = ds_mask1.interp(time=hampdata.radar.time)
add_itcz_mask(axes[0, 0], hampdata.radar.time, ds_mask2.itcz_mask, cbar=False)
savefig_format = "png"
savename = Path(cfg["paths"]["saveplts"]) / "radar_column_water_path.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

# %%
fig, axs = plt.subplots(
    nrows=3, ncols=2, figsize=(12, 6), width_ratios=[18, 7], sharey="row", sharex="col"
)
signal = plotfuncs.filter_radar_signal(hampdata.radar.dBZg, threshold=-30)  # [dBZ]
x = hampdata.radar.time
y = hampdata.radar.height / 1e3  # [km]
nrep = len(hampdata.radar.height)
itcz_mask_signal = np.repeat(ds_mask2.itcz_mask, nrep)
itcz_mask_signal = np.reshape(itcz_mask_signal.values, [len(hampdata.radar.time), nrep])
mask_names = ["Outside", "Transition", "Inside"]
for a in [0, 1, 2]:
    axs[a, 0].set_title(f"{mask_names[a]}", loc="left")
    signal_plt = np.where(itcz_mask_signal == a, signal, np.nan)
    pcol = axs[a, 0].pcolormesh(
        x,
        y,
        signal_plt.T,
        cmap="YlGnBu",
        vmin=-30,
        vmax=30,
    )
    clab, extend, shrink = "Z /dBZe", "max", 0.8
    cax = fig.colorbar(pcol, ax=axs[a, 0], label=clab, extend=extend, shrink=shrink)
    axs[a, 0].set_ylabel("Height / km")

    yy = np.meshgrid(y, x)[0]
    signal_range = [-30, 30]
    signal_bins = 60
    height_bins = 100
    h_cmap = plt.get_cmap("Greys")
    h_cmap = mcolors.LinearSegmentedColormap.from_list(
        "Sampled_Greys", h_cmap(np.linspace(0.15, 1.0, signal_bins))
    )
    h_cmap.set_under("white")
    hist, xbins, ybins, im = axs[a, 1].hist2d(
        signal_plt.flatten(),
        yy.flatten(),
        range=[signal_range, [0, np.nanmax(y)]],
        bins=[signal_bins, height_bins],
        cmap=h_cmap,
        vmin=signal_plt[signal_plt > 0].min(),
    )
    axs[a, 1].set_ylabel("")

for ax in axs.flatten():
    ax.spines[["top", "right"]].set_visible(False)

axs[2, 0].set_xlabel("UTC")
axs[2, 1].set_xlabel("Z /dBZe")

fig.tight_layout()
savefig_format = "png"
savename = Path(cfg["paths"]["saveplts"]) / "radar_selected.png"
dpi = 64
save_figure(fig, savefigparams=[savefig_format, savename, dpi])
