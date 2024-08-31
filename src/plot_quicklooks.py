import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

from . import plot_functions as plotfuncs
from . import earthcare_functions as ecfuncs
from .post_processed_hamp_data import PostProcessedHAMPData


def save_figure(fig, savefigparams):
    """
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [str, str, int] for [format to save figure,
        name to save figure, dpi of figure, used if format=="png"]
    """

    def save_png_figure(fig, savename, dpi):
        fig.savefig(
            savename, dpi=dpi, bbox_inches="tight", facecolor="w", format=format
        )
        print(f"figure saved as .png in: {savename}")

    def save_pdf_figure(fig, savename):
        fig.savefig(savename, bbox_inches="tight", format=format)
        print(f"figure saved as .pdf in: {savename}")

    format = savefigparams[0]
    if format == "png":
        savename, dpi = savefigparams[1:3]
        save_png_figure(fig, savename, dpi)
    elif format == "pdf":
        savename = savefigparams[1]
        save_pdf_figure(fig, savename)
    else:
        raise ValueError("Figure format unknown, please choose 'pdf' or 'png'")


def setup_hamp_timeslice_axes(fig):
    # Create a gridspec with 6 rows and 2 columns
    gs = gridspec.GridSpec(nrows=7, ncols=3, width_ratios=[24, 0.3, 4])

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
    ax6 = fig.add_subplot(gs[6, 0])

    return [[ax0a, cax0a, ax0b], ax1, ax2, ax3, ax4, ax5, ax6]


def hamp_timeslice_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    flightname,
    ec_under_time=None,
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
    flightname : str
        name of flight, e.g. "HALO-20240811a"
    ec_under_time: Time, Optional
        time of earthcare underpass
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [str, str, int] for [format to save figure,
        name to save figure, dpi of figure, used if format=="png"]

    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig = plt.figure(figsize=figsize)
    axes = setup_hamp_timeslice_axes(fig)

    # plot radar
    ds_radar_plot = hampdata.radar.sel(time=timeframe)
    plotfuncs.plot_radar_timeseries(ds_radar_plot, fig, axes[0][0], cax=axes[0][1])
    plotfuncs.plot_radar_histogram(ds_radar_plot, axes[0][2])

    # plot K-Band radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata.radiokv["TBs"].sel(time=timeframe, frequency=slice(22.24, 31.4)),
        axes[1],
    )

    # plot V-Band radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata.radiokv["TBs"].sel(time=timeframe, frequency=slice(50.3, 58)), axes[2]
    )

    # plot 90 GHz radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata.radio11990["TBs"].sel(time=timeframe, frequency=90),
        axes[3],
        is_90=True,
    )

    # plot 119 GHz radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata.radio11990["TBs"].sel(time=timeframe, frequency=slice(120.15, 127.25)),
        axes[4],
    )

    # plot 183 GHz radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata.radio183["TBs"].sel(time=timeframe), axes[5]
    )

    # plot CWV retrieval
    target_cwv = 48  # [mm]
    plotfuncs.plot_column_water_vapour_timeseries(
        hampdata["CWV"]["IWV"].sel(time=timeframe), axes[6], target_cwv=target_cwv
    )

    axes_timeseries_plots = [axes[0][0]] + axes[1:]
    for ax in axes_timeseries_plots:
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines[["top", "right"]].set_visible(False)
        if ec_under_time:
            ecfuncs.add_earthcare_underpass(ax, ec_under_time, annotate=False)
            ecfuncs.add_earthcare_underpass(axes[0][0], ec_under_time, annotate=True)

    axes[0][2].sharey(axes[0][0])
    for ax in axes[1:]:
        ax.sharex(axes[0][0])
    axes[0][2].spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Hour:Min UTC")

    fig.suptitle(f"HAMP {flightname}")

    fig.tight_layout()

    if savefigparams != []:
        save_figure(fig, savefigparams)

    return fig, axes


def hamp_hourly_quicklooks(
    hampdata: PostProcessedHAMPData,
    flightname,
    start_hour,
    end_hour,
    savefigparams=[],
):
    """
    Produces hourly HAMP PDF quicklooks for given flight and saves them as pdfs if requested.

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    flightname : str
        Name of flight, e.g. "HALO-20240811a"
    start_hour :  pandas.Timestamp
        start hour of quicklooks
    end_hour :  pandas.Timestamp
        final hour of quicklooks
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [str, str, int] for [format to save figure,
        directory to save figures in, dpi of figure, used if format=="png"]
    """

    # Generate hourly time slices
    timeslices = pd.date_range(start=start_hour, end=end_hour, freq="h")

    # produce quicklook plot for each full hour (excludes last timeslice)
    for i in range(0, len(timeslices) - 1):
        timeframe = slice(timeslices[i], timeslices[i + 1])
        fig, _ = hamp_timeslice_quicklook(
            hampdata,
            timeframe,
            flightname,
            figsize=(18, 18),
        )

        format, path_saveplts, dpi = savefigparams[0:3]
        savename = (
            path_saveplts
            / f"hamp_hourql_{timeslices[i].strftime('%Y%m%d_%H%M')}.{format}"
        )
        save_figure(fig, [format, savename, dpi])


def radiometer_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    figsize=(10, 14),
    savefigparams=[],
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
        Parameters are: [str, str, int] for [format to save figure,
        name to save figure, dpi of figure, used if format=="png"]
    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex="col")

    # plot K-Band radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata["kv"]["TBs"].sel(time=timeframe, frequency=slice(22.24, 31.4)), axes[0]
    )

    # plot V-Band radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata["kv"]["TBs"].sel(time=timeframe, frequency=slice(50.3, 58)), axes[1]
    )

    # plot 90 GHz radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata["11990"]["TBs"].sel(time=timeframe, frequency=90), axes[2], is_90=True
    )

    # plot 119 GHz radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata["11990"]["TBs"].sel(time=timeframe, frequency=slice(120.15, 127.25)),
        axes[3],
    )

    # plot 183 GHz radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata["183"]["TBs"].sel(time=timeframe), axes[4]
    )

    # plot CWV retrieval
    target_cwv = 48  # [mm]
    plotfuncs.plot_column_water_vapour_timeseries(
        hampdata["CWV"]["IWV"].sel(time=timeframe), axes[5], target_cwv=target_cwv
    )

    for ax in axes:
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"HAMP {timeframe.start} - {timeframe.stop}")

    fig.tight_layout()

    if savefigparams != []:
        save_figure(fig, savefigparams)

    return fig, axes


def radar_quicklook(
    hampdata: PostProcessedHAMPData,
    timeframe,
    flightname,
    ec_under_time=None,
    figsize=(9, 5),
    savefigparams=[],
    ds_radar=False,
    is_latllonaxes=True,
):
    """
    Produces radar quicklook for given timeframe and saves as .png if requested.

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    timeframe : slice
        Timeframe to plot.
    flightname : str
        name of flight, e.g. "HALO-20240811a"
    ec_under_time: Time, Optional
        time of earthcare underpass
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [str, str, int] for [format to save figure,
        name to save figure, dpi of figure, used if format=="png"]
    ds_radar : Any, optional
        If specified, use ds_radar for plotting rather than hampdata.radar
    is_latlonaxes : Bool, optional
        If True, add latitude and longitude axes to plot
        (requires hampdata with fligthdata as well as radar)
    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=figsize, width_ratios=[18, 7], sharey=True
    )
    # plot radar
    if ds_radar:
        ds_radar_plot = ds_radar.sel(time=timeframe)
    else:
        ds_radar_plot = hampdata.radar.sel(time=timeframe)

    ax, cax = plotfuncs.plot_radar_timeseries(ds_radar_plot, fig, axes[0])
    if ec_under_time:
        ecfuncs.add_earthcare_underpass(axes[0], ec_under_time, annotate=False)
        x, y = ec_under_time, axes[0].get_ylim()[1] * 0.92
        axes[0].annotate(" EarthCARE", xy=(x, y), xytext=(x, y), fontsize=15, color="r")

    if is_latllonaxes:
        plotfuncs.add_lat_lon_axes(hampdata, timeframe, ax, set_time_ticks=True)

    axes[0].set_title("  Timeseries", fontsize=18, loc="left")

    plotfuncs.plot_radar_histogram(ds_radar_plot, axes[1])
    axes[1].set_ylabel("")
    axes[1].set_title("Histogram", fontsize=18)

    plotfuncs.beautify_axes(axes)
    plotfuncs.beautify_colorbar_axes(cax)

    fig.suptitle(f"Radar During {flightname}", fontsize=20)

    fig.tight_layout()

    if savefigparams != []:
        save_figure(fig, savefigparams)

    return fig, axes


def plot_kvband_column_water_vapour_retrieval(
    hampdata: PostProcessedHAMPData,
    timeframe,
    flightname,
    ec_under_time=None,
    figsize=(5, 9),
    savefigparams=[],
):
    """
    Produces HAMP quicklook for column water vapour retireval
    and saves as .png if requested.

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    timeframe : slice
        Timeframe to plot.
    flightname : str, optional
        name of flight, e.g. "HALO-20240811a"
    ec_under_time: Time, Optional
        time of earthcare underpass
    figsize : tuple, optional
        Figure size in inches, by default (10, 14)
    savefigparams : tuple, optional
        tuple for parameters to save figure as .png.
        Parameters are: [str, str, int] for [format to save figure,
        name to save figure, dpi of figure, used if format=="png"]

    Returns
    -------
    fig, axes
        Figure and axes of the plot.
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize)

    # plot K-Band radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata.radiokv["TBs"].sel(time=timeframe, frequency=slice(22.24, 31.4)),
        axes[0],
    )
    axes[0].set_title("K-Band")

    # plot V-Band radiometer
    plotfuncs.plot_radiometer_timeseries(
        hampdata.radiokv["TBs"].sel(time=timeframe, frequency=slice(50.3, 58)), axes[1]
    )
    axes[1].set_title("V-Band")

    plotfuncs.plot_column_water_vapour_timeseries(
        hampdata["CWV"]["IWV"].sel(time=timeframe), axes[2], target_cwv=48
    )
    axes[2].set_title("KV-Bands Column Water Vapour Retrieval")

    if ec_under_time:
        for ax in axes:
            ecfuncs.add_earthcare_underpass(ax, ec_under_time, annotate=False)
        ecfuncs.add_earthcare_underpass(axes[0], ec_under_time, annotate=True)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Hour:Min UTC")
    fig.suptitle(f"Radiometers during {flightname}")

    fig.tight_layout()

    if savefigparams != []:
        save_figure(fig, savefigparams)

    return fig, axes
