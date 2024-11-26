import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates


def filter_radar_signal(dBZg, threshold=-30):
    return dBZg.where(dBZg >= threshold)  # [dBZ]


def beautify_axes(axes):
    def beautify_ax(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    for ax in axes:
        beautify_ax(ax)


def beautify_colorbar_axes(cax, xaxis=False):
    if xaxis:
        cax.ax.tick_params(axis="x", which="major", labelsize=12)
        cax.ax.xaxis.label.set_size(15)
    else:
        cax.ax.tick_params(axis="y", which="major", labelsize=12)
        cax.ax.yaxis.label.set_size(15)
    cax.ax.tick_params(labelsize=15)


def add_lat_lon_axes(hampdata, timeframe, ax_time, set_time_ticks=False):
    """add latitude and longitude axes beneath a time axis
    with nticks between time[0] and time[-1]"""

    def label_axis(ax, pos, ticklabs, lab):
        ax.spines["bottom"].set_position(("outward", pos))
        ax.set_xticklabels(ticklabs)
        ax.set_xlabel(lab)

    ax_lat, ax_lon = ax_time.twiny(), ax_time.twiny()
    nticks = 5
    time_slice = hampdata.radar.time.sel(time=timeframe)
    idxs = np.linspace(0, len(time_slice.time) - 1, nticks, dtype=int)
    xticks0 = time_slice.isel(time=idxs)
    xticks = hampdata.radar.time.sel(time=xticks0.values, method="nearest")

    for ax in [ax_lat, ax_lon]:
        ax.set_xlim(ax_time.get_xlim())
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xticks(xticks)
        ax.spines[["top", "right", "left"]].set_visible(False)

    lat_labels = hampdata.radar.lat.sel(time=xticks.values).round(1).values
    label_axis(ax_lat, 45, lat_labels, "Latitude /$\u00B0$")

    lon_labels = hampdata.radar.lat.sel(time=xticks.values).round(1).values
    label_axis(ax_lon, 85, lon_labels, "Longitude /$\u00B0$")

    if set_time_ticks:
        ax_time.set_xticks(xticks0)
        ax_time.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))


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
    # ds = ds.where((ds < 300) & (ds > 100))

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

    return ax


def plot_column_water_vapour_timeseries(ds, ax, target_cwv=None):
    """
    Plot column water vapour retrieval data (from KV band radiometer)

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset for column water vapour retrieval
    ax : matplotlib.axes.Axes
        Axes to plot on.
    target_cwv : float
        value to CWV [mm] to mark on plot, e.g. 48mm contour
    """

    # ds = ds.where((ds > 20) & (ds < 80))
    ds.plot.line(ax=ax, x="time", color="k")

    if target_cwv:
        # Plot where CWV equals the target value (within 0.01%)
        target_indices = np.where(
            np.isclose(ds.values, target_cwv, atol=1e-4 * target_cwv)
        )
        if ds[target_indices].any():
            ds[target_indices].plot.scatter(
                ax=ax, x="time", color="dodgerblue", marker="x"
            )

        ax.axhline(
            target_cwv,
            color="dodgerblue",
            linestyle="--",
            linewidth=1.0,
            label=f"CWV={target_cwv}mm",
        )

    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
    ax.set_ylabel("CWV / mm")

    return ax


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
        ax, cax = plot_radardata_timeseries(
            time, height, signal, fig, ax, cax=cax, cmap=cmap
        )

    return ax, cax


def plot_radardata_timeseries(time, height, signal, fig, ax, cax=None, cmap="YlGnBu"):
    """you may want to filter_radar_signal before calling this function"""
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
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))

    ax.set_xlabel("UTC")
    ax.set_ylabel("Height / km")

    return ax, cax


def get_greys_histogram_colourmap(nbins):
    cmap = plt.get_cmap("Greys")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Sampled_Greys", cmap(np.linspace(0.15, 1.0, nbins))
    )
    cmap.set_under("white")

    return cmap


def plot_radar_histogram(
    ds_radar,
    ax,
    signal_range=[-30, 30],
    height_range=[],
    signal_bins=60,
    height_bins=100,
    cmap=None,
):
    time = ds_radar.time
    height = ds_radar.height / 1e3  # [km]
    signal = filter_radar_signal(ds_radar.dBZg, threshold=-30).values  # [dBZ]

    if height_range == []:
        height_range = [0.0, np.nanmax(height)]

    if not cmap:
        if isinstance(signal_bins, int):
            cmap = get_greys_histogram_colourmap(signal_bins)
        else:
            cmap = get_greys_histogram_colourmap(len(signal_bins))
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    ax = plot_radardata_histogram(
        time,
        height,
        signal,
        ax,
        signal_range,
        height_range,
        height_bins,
        signal_bins,
        cmap,
    )

    return ax


def plot_radardata_histogram(
    time,
    height,
    signal,
    ax,
    signal_range,
    height_range,
    height_bins,
    signal_bins,
    cmap,
):
    """you may want to filter_radar_signal before calling this function"""

    # get data in correct format for 2D histogram and remove nan data
    signal = signal.flatten()
    height = np.meshgrid(height, time)[0].flatten()
    height = height[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]

    # set histogram parameters
    bins = [signal_bins, height_bins]
    cmap.set_under("white")

    # plot 2D histogram
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

    return ax


# %% fuunctions for plotting HAMP post-processed data slice
def plot_radar_cwv_timeseries(
    fig,
    axs,
    hampdata,
):
    cax = plot_radar_timeseries(hampdata.radar, fig, axs[0, 0])[1]
    axs[0, 0].set_title("  Timeseries", fontsize=18, loc="left")

    plot_radar_histogram(hampdata.radar, axs[0, 1])
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_title("Histogram", fontsize=18)

    plot_column_water_vapour_timeseries(
        hampdata["CWV"]["IWV"], axs[1, 0], target_cwv=48
    )

    beautify_axes(axs.flatten())
    beautify_colorbar_axes(cax)
    axs[1, 1].remove()

    fig.tight_layout()

    return fig, axs


def plot_dropsonde_iwv_comparison(ds_iwv, ds_dropsonde, bias, date, ax):
    """
    Plot IWV from radiometer and dropsondes for a given date.

    Parameters
    ----------
    ds_iwv : xr.Dataset
        Level1 radiometer dataset for column water vapour retrieval
    ds_dropsonde : xr.Dataset
        Level1 dropsonde dataset
    date : str
        Date in format YYMMDD
    ax : matplotlib.axes.Axes
        Axes to plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with plot.
    """

    ax.plot(ds_iwv.time, ds_iwv.IWV, label="KV Radiometer", color="dodgerblue")
    ax.scatter(
        ds_dropsonde.launch_time,
        ds_dropsonde.iwv,
        label="Dropsondes",
        marker="*",
        c="#FF5349",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(f"20{date[:2]}-{date[2:4]}-{date[4:]}")
    ax.annotate(
        f"Bias: {bias:.3f} kg m$^{-2}$",
        (0.8, 0.05),
        xycoords="axes fraction",
        ha="center",
        va="center",
    )
    return ax


def get_hamp_TBs(hampdata):
    TB_hamp = []
    freqs_hamp = []
    for radio in ["radiokv", "radio11990", "radio183"]:
        TB_hamp = TB_hamp + list(hampdata[radio]["TBs"].values)
        freqs_hamp = freqs_hamp + list(hampdata[radio]["frequency"].values)
    freqs_hamp = np.array(freqs_hamp).astype(float).round(2)
    return freqs_hamp, TB_hamp


def plot_arts_flux(TB_hamp, TB_arts, dropsonde_id, time):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(TB_arts.index, TB_arts.values, label="ARTS", marker="o")
    ax.scatter(TB_hamp.index, TB_hamp.values, label="HAMP", color="red", marker="x")
    ax.set_xlabel("Frequency / GHz")
    ax.set_ylabel("Brightness Temperature / K")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    # ax.set_title(f"Dropsonde {dropsonde_id} at {time.strftime('%Y-%m-%d %H:%M')}")
    fig.tight_layout()
    return fig, ax
