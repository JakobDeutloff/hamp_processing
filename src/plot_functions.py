import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr


def filter_radar_signal(dBZg, threshold=-30):
    return dBZg.where(dBZg >= threshold)  # [dBZ]


def plot_radiometer_timeseries(ds, ax, is_90=False):
    """
    Plot radiometer data for all frequencies in dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """

    if is_90:
        ds.plot.line(ax=ax, x="time", color="k")
        ax.legend(
            handles=ax.lines,
            labels=["90 GHz"],
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            frameon=False,
        )
    else:
        frequencies = ds.frequency.values
        norm = mcolors.Normalize(vmin=frequencies.min(), vmax=frequencies.max())
        cmap = plt.colormaps.get_cmap("viridis")

        for freq in frequencies:
            color = cmap(norm(freq))
            ds.sel(frequency=freq).plot.line(
                ax=ax, x="time", color=color, label=f"{freq:.2f} GHz"
            )
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)

    ax.set_ylabel("TB / K")


def plot_radar_timeseries(ds, fig, ax, cax=None, cmap="YlGnBu"):

    # check if radar data is available
    if ds.dBZg.size == 0:
        ax.text(
            0.5,
            0.5,
            "No radar data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        time, height = ds.time, ds.height / 1e3  # [UTC], [km]
        signal = filter_radar_signal(ds.dBZg, threshold=-30).T  # [dBZ]
        pcol = ax.pcolormesh(
            time,
            height,
            signal,
            cmap=cmap,
            vmin=-30,
            vmax=30,
        )
        clab, extend, shrink = "Z /dBZe", "max", 0.8
        if cax:
            cax = fig.colorbar(pcol, cax=cax, label=clab, extend=extend, shrink=shrink)
        else:
            cax = fig.colorbar(pcol, ax=ax, label=clab, extend=extend, shrink=shrink)

    # get nicely formatting xticklabels
    stride = len(time) // 4
    xticks = time[::stride]
    xticklabs = [f"{t.hour:02d}:{t.minute:02d}" for t in pd.to_datetime(xticks)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabs)

    ax.set_xlabel("UTC")
    ax.set_ylabel("Height / km")

    return ax, cax, pcol


def plot_beautified_radar_histogram(ds_radar_plot, ax):
    signal_range = [-30, 30]
    signal_bins = 60
    height_bins = 100
    cmap = plt.get_cmap("Greys")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Sampled_Greys", cmap(np.linspace(0.15, 1.0, signal_bins))
    )
    plot_radar_histogram(
        ds_radar_plot,
        ax,
        signal_range=signal_range,
        height_bins=height_bins,
        signal_bins=signal_bins,
        cmap=cmap,
    )


def plot_radar_histogram(
    ds,
    ax,
    signal_range=[],
    height_range=[],
    height_bins=50,
    signal_bins=60,
    cmap="Grays",
):
    # get data in correct format for 2D histogram
    height = np.meshgrid(ds.height, ds.time)[0].flatten() / 1e3  # [km]
    signal = filter_radar_signal(ds.dBZg, threshold=-30).values.flatten()  # [dBZ]

    # remove nan data
    height = height[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]

    # set histogram parameters
    bins = [signal_bins, height_bins]
    if signal_range == []:
        signal_range = [signal.min(), signal.max()]
    if height_range == []:
        height_range = [0.0, height.max()]

    # plot 2D histogram
    cmap = plt.cm.get_cmap(cmap)
    cmap.set_under("white")
    hist, xbins, ybins, im = ax.hist2d(
        signal,
        height,
        range=[signal_range, height_range],
        bins=bins,
        cmap=cmap,
        vmin=signal[signal > 0].min(),
    )

    ax.set_xlabel("Z /dBZe")
    ax.set_ylabel("Height / km")
    ax.set_xticks

    return ax, hist, xbins, ybins
