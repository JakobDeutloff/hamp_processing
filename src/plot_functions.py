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


def hamp_ql(ds_radar_lev1, ds_11990_lev1, ds_183_lev1, ds_kv_lev1, timeframe, figsize=(14, 18), flight=None):
    """
    Produces HAMP quicklook for given timeframe.

    Parameters
    ----------
    ds_radar_lev1 : xr.Dataset
        Level1 radar dataset.
    ds_11990_lev1 : xr.Dataset
        Level1 119/90 GHz radiometer dataset.
    ds_183_lev1 : xr.Dataset
        Level1 183 GHz radiometer dataset.
    ds_kv_lev1 : xr.Dataset
        Level1 K/V radiometer dataset.
    timeframe : slice
        Timeframe to plot.
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)

    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(
        6, 1, figsize=figsize, height_ratios=[3, 1, 1, 1, 1, 1], sharex="col"
    )

    # plot radar
    ds_radar_plot = ds_radar_lev1.sel(time=timeframe)

    # check if radar data is available
    if ds_radar_plot.dBZg.size == 0:
        axes[0].text(0.5, 0.5, "No radar data available", ha="center", va="center", transform=axes[0].transAxes)
    else:
        pcol = axes[0].pcolormesh(
            ds_radar_plot.time,
            ds_radar_plot.height / 1e3,
            ds_radar_plot.dBZg.where(ds_radar_plot.dBZg > -25).T,
            cmap="turbo",
            vmin=-25,
            vmax=25,
        )
        cax = fig.add_axes([0.84, 0.63, 0.02, 0.25])
        cb = fig.colorbar(pcol, cax=cax, label="dBZe", extend="max")

    axes[0].set_ylabel("Height / km")
    fig.subplots_adjust(right=0.8)


    # plot K-Band radiometer
    plot_radiometer(
        ds_kv_lev1["TBs"].sel(time=timeframe, frequency=slice(22.24, 31.4)), axes[1]
    )

    # plot V-Band radiometer
    plot_radiometer(
        ds_kv_lev1["TBs"].sel(time=timeframe, frequency=slice(50.3, 58)), axes[2]
    )

    # plot 90 GHz radiometer
    ds_11990_lev1["TBs"].sel(time=timeframe, frequency=90).plot.line(
        ax=axes[3], x="time", color="k"
    )
    axes[3].legend(
        handles=axes[3].lines,
        labels=["90 GHz"],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    axes[3].set_ylabel("TB / K")


    # plot 119 GHz radiometer
    plot_radiometer(
        ds_11990_lev1["TBs"].sel(time=timeframe, frequency=slice(120.15, 127.25)), axes[4]
    )

    # plot 183 GHz radiometer
    plot_radiometer(ds_183_lev1["TBs"].sel(time=timeframe), axes[5])

    for ax in axes:
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"HAMP {flight}", y=0.92)
    return fig, axes


def radiometer_ql(ds_11990_lev1, ds_183_lev1, ds_kv_lev1, timeframe, figsize=(10, 14)):
    """
    Produces HAMP quicklook for given timeframe.

    Parameters
    ----------
    ds_radar_lev1 : xr.Dataset
        Level1 radar dataset.
    ds_11990_lev1 : xr.Dataset
        Level1 119/90 GHz radiometer dataset.
    ds_183_lev1 : xr.Dataset
        Level1 183 GHz radiometer dataset.
    ds_kv_lev1 : xr.Dataset
        Level1 K/V radiometer dataset.
    timeframe : slice
        Timeframe to plot.
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)

    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(
        5, 1, figsize=figsize, sharex="col"
    )

    # plot K-Band radiometer
    plot_radiometer(
        ds_kv_lev1["TBs"].sel(time=timeframe, frequency=slice(22.24, 31.4)), axes[0]
    )

    # plot V-Band radiometer
    plot_radiometer(
        ds_kv_lev1["TBs"].sel(time=timeframe, frequency=slice(50.3, 58)), axes[1]
    )

    # plot 90 GHz radiometer
    ds_11990_lev1["TBs"].sel(time=timeframe, frequency=90).plot.line(
        ax=axes[2], x="time", color="k"
    )
    axes[2].legend(
        handles=axes[2].lines,
        labels=["90 GHz"],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    axes[2].set_ylabel("TB / K")


    # plot 119 GHz radiometer
    plot_radiometer(
        ds_11990_lev1["TBs"].sel(time=timeframe, frequency=slice(120.15, 127.25)), axes[3]
    )

    # plot 183 GHz radiometer
    plot_radiometer(ds_183_lev1["TBs"].sel(time=timeframe), axes[4])

    for ax in axes:
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"HAMP {timeframe.start} - {timeframe.stop}", y=0.92)
    return fig, axes


def produce_hourly_hamp_ql(ds_radar_lev1, ds_11990_lev1, ds_183_lev1, ds_kv_lev1, flightname):
    """
    Produces hourly HAMP PDF quicklooks for given flight and stores them under the flightname in quicklooks folder.

    Parameters
    ----------
    ds_radar_lev1 : xr.Dataset
        Level1 radar dataset.
    ds_11990_lev1 : xr.Dataset
        Level1 119/90 GHz radiometer dataset.
    ds_183_lev1 : xr.Dataset
        Level1 183 GHz radiometer dataset.
    ds_kv_lev1 : xr.Dataset
        Level1 K/V radiometer dataset.
    flightname : str
        Name of the flight.
    """

    # make new directory for this flight
    os.makedirs(f"quicklooks/{flightname}", exist_ok=True)

    # Round start and end times to full hour
    start_time = pd.Timestamp(ds_183_lev1.time[0].values).floor('H')
    end_time = pd.Timestamp(ds_183_lev1.time[-1].values).ceil('H')

    # Generate hourly time slices
    timeslices = pd.date_range(start=start_time, end=end_time, freq='H')

    # produce ql plot for each full hour time
    for i in range(0, len(timeslices)-1):
        fig, _ = hamp_ql(ds_radar_lev1, ds_11990_lev1, ds_183_lev1, ds_kv_lev1, slice(timeslices[i], timeslices[i+1]))
        fig.savefig(f"quicklooks/hamp/{flightname}/ql_{timeslices[i].strftime('%Y%m%d_%H%M')}.pdf", bbox_inches='tight')
