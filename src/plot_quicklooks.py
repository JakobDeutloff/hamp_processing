import os
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

from .plot_functions import (
    plot_radiometer_timeseries,
    plot_radar_timeseries,
    plot_radar_histogram,
)
from .post_processed_hamp_data import PostProcessedHAMPData


def save_png_figure(fig, savename, dpi):
    fig.savefig(savename, dpi=dpi, bbox_inches="tight", facecolor="w", format="png")
    print("figure saved as .png in: " + savename)


def save_pdf_figure(fig, savename):
    fig.savefig(savename, bbox_inches="tight", format="pdf")
    print("figure saved as .pdf in: " + savename)


def hamp_timeslice_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    flight=None,
    figsize=(14, 18),
    savefigparams=[],
):
    """
    Produces HAMP quicklook for given timeframe and saves as .png if requested.

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    timeframe : slice
        Timeframe to plot.
    flight : str, optional
        name of flight, e.g. "RF01_20240811"
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [boolean, string, int] for
        [save figure if True, name to save figure, dpi of figure]

    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(
        6, 1, figsize=figsize, height_ratios=[3, 1, 1, 1, 1, 1], sharex="col"
    )

    # plot radar
    ds_radar_plot = hampdata.radar.sel(time=timeframe)
    cax = fig.add_axes([0.84, 0.63, 0.02, 0.25])
    plot_radar_timeseries(ds_radar_plot, fig, axes[0], cax)
    fig.subplots_adjust(right=0.8)

    # plot K-Band radiometer
    plot_radiometer_timeseries(
        hampdata.radiokv["TBs"].sel(time=timeframe, frequency=slice(22.24, 31.4)),
        axes[1],
    )

    # plot V-Band radiometer
    plot_radiometer_timeseries(
        hampdata.radiokv["TBs"].sel(time=timeframe, frequency=slice(50.3, 58)), axes[2]
    )

    # plot 90 GHz radiometer
    plot_radiometer_timeseries(
        hampdata.radio11990["TBs"].sel(time=timeframe, frequency=90),
        axes[3],
        is_90=True,
    )

    # plot 119 GHz radiometer
    plot_radiometer_timeseries(
        hampdata.radio11990["TBs"].sel(time=timeframe, frequency=slice(120.15, 127.25)),
        axes[4],
    )

    # plot 183 GHz radiometer
    plot_radiometer_timeseries(hampdata.radio183["TBs"].sel(time=timeframe), axes[5])

    for ax in axes:
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"HAMP {flight}", y=0.92)

    if savefigparams[0]:
        savename, dpi = savefigparams[1], savefigparams[2]
        save_png_figure(fig, savename, dpi)

    return fig, axes


def hamp_hourly_quicklooks(
    hampdata: PostProcessedHAMPData, flight, start_hour, end_hour, savepdfparams=[]
):
    """
    Produces hourly HAMP PDF quicklooks for given flight and saves them as pdfs if requested.

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    flight : str
        Name of the flight.
    start_hour :  pandas.Timestamp
        start hour of quicklooks
    end_hour :  pandas.Timestamp
        final hour of quicklooks
    savefigparams : tuple, optional
        tuple for parameters to save figures as .pdfs.
        Parameters are: [boolean, string, int] for
        [save pdf figures if True, directory to save .pdfs in]
    """

    # Generate hourly time slices
    timeslices = pd.date_range(start=start_hour, end=end_hour, freq="h")

    # produce quicklook plot for each full hour (excludes last timeslice)
    for i in range(0, len(timeslices) - 1):
        fig, _ = hamp_timeslice_quicklook(
            hampdata,
            timeframe=slice(timeslices[i], timeslices[i + 1]),
            flight=flight,
            figsize=(18, 18),
            savefigparams=[False],
        )

        if savepdfparams[0]:
            savename = f"{savepdfparams[1]}/hamp_hourql_{timeslices[i].strftime('%Y%m%d_%H%M')}.pdf"
            save_pdf_figure(fig, savename)


def radiometer_quicklook(
    hampdata: PostProcessedHAMPData, timeframe, figsize=(10, 14), savefigparams=[]
):
    """
    Produces HAMP quicklook for given timeframe.

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    timeframe : slice
        Timeframe to plot.
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [boolean, string, int] for
        [save figure if True, name to save figure, dpi of figure]
    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex="col")

    # plot K-Band radiometer
    plot_radiometer_timeseries(
        hampdata["kv"]["TBs"].sel(time=timeframe, frequency=slice(22.24, 31.4)), axes[0]
    )

    # plot V-Band radiometer
    plot_radiometer_timeseries(
        hampdata["kv"]["TBs"].sel(time=timeframe, frequency=slice(50.3, 58)), axes[1]
    )

    # plot 90 GHz radiometer
    plot_radiometer_timeseries(
        hampdata["11990"]["TBs"].sel(time=timeframe, frequency=90), axes[2], is_90=True
    )

    # plot 119 GHz radiometer
    plot_radiometer_timeseries(
        hampdata["11990"]["TBs"].sel(time=timeframe, frequency=slice(120.15, 127.25)),
        axes[3],
    )

    # plot 183 GHz radiometer
    plot_radiometer_timeseries(hampdata["183"]["TBs"].sel(time=timeframe), axes[4])

    for ax in axes:
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"HAMP {timeframe.start} - {timeframe.stop}", y=0.92)

    if savefigparams[0]:
        savename, dpi = savefigparams[1], savefigparams[2]
        save_png_figure(fig, savename, dpi)

    return fig, axes


def radar_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    flight=None,
    figsize=(9, 5),
    savefigparams=[],
):
    """
    Produces radar quicklook for given timeframe and saves as .png if requested.

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    timeframe : slice
        Timeframe to plot.
    flight : str, optional
        name of flight, e.g. "RF01_20240811"
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [boolean, string, int] for
        [save figure if True, name to save figure, dpi of figure]

    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=figsize, width_ratios=[4, 1], sharey=True
    )
    # plot radar
    ds_radar_plot = hampdata.radar.sel(time=timeframe)

    plot_radar_timeseries(ds_radar_plot, fig, axes[0], None)
    axes[0].set_xlabel("")
    axes[0].set_title("Timeseries")

    signal_range = [-40, 40]
    plot_radar_histogram(ds_radar_plot, axes[1], signal_range=signal_range)
    axes[1].set_ylabel("")
    axes[1].set_title("Histogram")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Radar During Flight {flight}")

    fig.tight_layout()

    if savefigparams[0]:
        savename, dpi = savefigparams[1], savefigparams[2]
        save_png_figure(fig, savename, dpi)

    return fig, axes
