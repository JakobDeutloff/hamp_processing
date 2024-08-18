import os
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd


def plot_radiometer(ds, ax):
    """
    Plot radiometer data for all frequencies in dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """

    frequencies = ds.frequency.values
    norm = mcolors.Normalize(vmin=frequencies.min(), vmax=frequencies.max())
    cmap = plt.colormaps.get_cmap("viridis")

    for freq in frequencies:
        color = cmap(norm(freq))
        ds.sel(frequency=freq).plot.line(
            ax=ax, x="time", color=color, label=f"{freq:.2f} GHz"
        )
    ax.set_ylabel("TB / K")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
