import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

from .plot_functions import (
    plot_radiometer_timeseries,
    plot_radar_timeseries,
    plot_beautified_radar_histogram,
)
from .post_processed_hamp_data import PostProcessedHAMPData


def save_figure(fig, format, savename, dpi):
    def save_png_figure(fig, savename, dpi):
        fig.savefig(
            savename, dpi=dpi, bbox_inches="tight", facecolor="w", format=format
        )
        print("figure saved as .png in: " + savename)

    def save_pdf_figure(fig, savename):
        fig.savefig(savename, bbox_inches="tight", format=format)
        print("figure saved as .pdf in: " + savename)

    if format == "png":
        save_png_figure(fig, savename, dpi)
    elif format == "pdf":
        save_pdf_figure(fig, savename)
    else:
        raise ValueError("Figure format unknown, please choose 'pdf' or 'png'")


def add_earthcare_underpass(ax, ec_under_time):
    color = "r"
    ax.axvline(ec_under_time, color=color, linestyle="--", linewidth=1.0)

    x, y = ec_under_time, ax.get_ylim()[1]
    ax.annotate("EarthCARE", xy=(x, y), xytext=(x, y), fontsize=10, color=color)


def setup_hamp_timeslice_axes(fig):
    # Create a gridspec with 6 rows and 2 columns
    gs = gridspec.GridSpec(nrows=6, ncols=3, width_ratios=[24, 0.3, 4])

    # Top row for radar has two seperate plots
    ax0a = fig.add_subplot(gs[0, 0])
    cax0a = fig.add_subplot(gs[0, 1])
    ax0b = fig.add_subplot(gs[0, 2])

    # Remaining rows for radiometers have 1 plot
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[4, 0])
    ax5 = fig.add_subplot(gs[5, 0])

    return [[ax0a, cax0a, ax0b], ax1, ax2, ax3, ax4, ax5]


def hamp_timeslice_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    flight=None,
    ec_under_time=None,
    figsize=(14, 18),
    savefigparams=[None, None, None],
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
    ec_under_time: Time, Optional
        time of earthcare underpass
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

    fig = plt.figure(figsize=figsize)
    axes = setup_hamp_timeslice_axes(fig)

    # plot radar
    ds_radar_plot = hampdata.radar.sel(time=timeframe)
    plot_radar_timeseries(ds_radar_plot, fig, axes[0][0], cax=axes[0][1])
    plot_beautified_radar_histogram(ds_radar_plot, axes[0][2])

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

    axes_timeseries_plots = [axes[0][0]] + axes[1:]
    for ax in axes_timeseries_plots:
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines[["top", "right"]].set_visible(False)
        if ec_under_time:
            add_earthcare_underpass(ax, ec_under_time)

    axes[0][2].sharey(axes[0][0])
    for ax in axes[1:]:
        ax.sharex(axes[0][0])
    axes[0][2].spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Hour:Min UTC")

    fig.suptitle(f"HAMP {flight}", y=0.92)

    if savefigparams[0]:
        format, savename, dpi = savefigparams[0:3]
        save_figure(fig, format, savename, dpi)

    return fig, axes


def hamp_hourly_quicklooks(
    hampdata: PostProcessedHAMPData,
    flight,
    start_hour,
    end_hour,
    savefigparams=[None, None, None],
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

        if savefigparams[0]:
            format, savedir, dpi = savefigparams[0], savefigparams[1], savefigparams[2]
            savename = (
                savedir
                / f"hamp_hourql_{timeslices[i].strftime('%Y%m%d_%H%M')}.{format}"
            )
            save_figure(fig, format, savename, dpi)


def radiometer_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    figsize=(10, 14),
    savefigparams=[None, None, None],
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
        format, savename, dpi = savefigparams[0:3]
        save_figure(fig, format, savename, dpi)

    return fig, axes


def radar_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    flight=None,
    ec_under_time=None,
    figsize=(9, 5),
    savefigparams=[None, None, None],
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
    ec_under_time: Time, Optional
        time of earthcare underpass
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
        nrows=1, ncols=2, figsize=figsize, width_ratios=[18, 7], sharey=True
    )
    # plot radar
    ds_radar_plot = hampdata.radar.sel(time=timeframe)

    cax = plot_radar_timeseries(ds_radar_plot, fig, axes[0])[1]
    if ec_under_time:
        add_earthcare_underpass(axes[0], ec_under_time)
    axes[0].set_title("Timeseries", fontsize=18)

    plot_beautified_radar_histogram(ds_radar_plot, axes[1])
    axes[1].set_ylabel("")
    axes[1].set_title("Histogram", fontsize=18)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    cax.ax.tick_params(axis="y", which="major", labelsize=12)
    cax.ax.yaxis.label.set_size(15)
    cax.ax.tick_params(labelsize=15)

    fig.suptitle(f"Radar During {flight}", fontsize=20)

    fig.tight_layout()

    if savefigparams[0]:
        format, savename, dpi = savefigparams[0:3]
        save_figure(fig, format, savename, dpi)

    return fig, axes
