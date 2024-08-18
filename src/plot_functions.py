import os
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd


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
            bbox_to_anchor=(1, 0.5),
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
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set_ylabel("TB / K")


def plot_radar_timeseries(ds, fig, ax):
    """WIP 15:41 UTC"""

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
        pcol = ax.pcolormesh(
            ds.time,
            ds.height / 1e3,
            ds.dBZg.where(ds.dBZg > -25).T,
            cmap="turbo",
            vmin=-25,
            vmax=25,
        )
        cax = fig.add_axes([0.84, 0.63, 0.02, 0.25])
        cb = fig.colorbar(pcol, cax=cax, label="dBZe", extend="max")

    ax.set_ylabel("Height / km")
