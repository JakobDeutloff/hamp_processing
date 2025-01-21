# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr
import matplotlib.pyplot as plt
from src import earthcare_functions as ecfuncs
import pandas as pd
import matplotlib.dates as mdates


# %%


def segmet_track(ec_track, ds_radar):
    # Calculate the time differences between consecutive time points
    time_diffs = ec_track.time.diff(dim="time")

    # Find the indices where the time difference is greater than 20 minutes
    gap_indices = time_diffs > pd.Timedelta("20min")

    # Extract the times where the gaps occur
    ec_sections = []
    gap_times = ec_track.time[:-1][gap_indices.values]
    ec_full = ec_track.copy()
    for gap_time in gap_times:
        ec_sections.append(ec_full.sel(time=slice(ec_full.time.min(), gap_time)))
        idx_gap = int((ec_full.time == gap_time).argmax())
        ec_full = ec_full.isel(time=slice(idx_gap + 1, None))

    # check overlap with radar data
    ec_inflight = []
    for ec_section in ec_sections:
        if (
            (ec_section.time[0].values > ds_radar.time[0].values)
            and (ec_section.time[0].values < ds_radar.time[-1].values)
        ) or (
            (ec_section.time[-1].values > ds_radar.time[0].values)
            and (ec_section.time[-1].values < ds_radar.time[-1].values)
        ):
            ec_inflight.append(ec_section)

    return ec_inflight


def read_radar(date, flightletter):
    ds_radar = xr.open_dataset(
        f"Data/Hamp_Processed/radar/HALO-{date}{flightletter}_radar.zarr", engine="zarr"
    )
    plot_duration = pd.Timedelta("20m")
    ec_track = ecfuncs.get_earthcare_track(date)
    if pd.Timestamp(date) > pd.Timestamp("2024-11-01"):
        ec_segments = segmet_track(ec_track, ds_radar)
    else:
        ec_segments = [ec_track]

    ec_under_times = []
    radar_segments = []
    for segment in ec_segments:
        ec_under_time = ecfuncs.find_ec_under_time(segment, ds_radar)
        ec_under_times.append(ec_under_time)
        ec_starttime, ec_endtime = (
            ec_under_time - plot_duration / 2,
            ec_under_time + plot_duration / 2,
        )
        timeframe = slice(ec_starttime, ec_endtime)
        radar = ds_radar.sel(time=timeframe)
        if radar.time.size > 0:
            radar_segments.append(radar)
        else:
            radar_segments.append(None)

    return radar_segments, ec_under_times


def plot_cloudfraction(ax, ds_radar, ec_under_time):
    # plot
    pmesh = (
        ds_radar["dBZg"]
        .where(ds_radar["dBZg"] > -30)
        .plot.pcolormesh(
            ax=ax,
            x="time",
            y="height",
            add_colorbar=False,
            vmax=30,
            vmin=-30,
            cmap="YlGnBu",
            extend="max",
        )
    )
    # print underpass time at the top of the axis

    ax.text(
        1,
        0.95,
        f"{pd.to_datetime(ec_under_time).strftime('%d.%m.')}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="k",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)

    # Set the x-axis formatter to show only the time
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    return pmesh


# %% load data
dates = [
    "20240811",
    "20240813",
    "20240816",
    "20240818",
    "20240821",
    "20240822",
    "20240825",
    "20240827",
    "20240829",
    "20240831",
    "20240903",
    "20240907",
    "20240909",
    "20240912",
    "20240914",
    "20240916",
    "20240919",
    "20240921",
    "20240923",
    "20240924",
    "20240926",
    "20240928",
    "20241105",
    "20241107",
    "20241110",
    "20241112",
    "20241114",
    "20241116",
    "20241119",
]
flightletters = ["a"] * len(dates)
flightletters[25] = "b"
flightletters[26] = "b"

underpasses = []
radar_data = []
for date, flightletter in zip(dates, flightletters):
    (
        radar_segments,
        ec_under_times,
    ) = read_radar(date, flightletter)
    radar_data += radar_segments
    underpasses += ec_under_times

radar_data = [entry for entry in radar_data if entry is not None]
underpasses = [entry for entry in underpasses if entry is not None]


# %% call plotfunction
fig, axes = plt.subplots(11, 3, figsize=(23.4, 33.1), sharey=True)
for i in range(len(radar_data)):
    ax = axes.flat[i]
    pmesh = plot_cloudfraction(
        ax, radar_data[i].assign(height=radar_data[i].height / 1e3), underpasses[i]
    )

fig.colorbar(
    pmesh,
    ax=axes,
    label="Z / dBZg",
    aspect=40,
    extend="max",
    orientation="horizontal",
    pad=0.03,
)
for ax in axes[:, 0]:
    ax.set_ylabel("Height / km")
for ax in axes[-1, :]:
    ax.set_xlabel("Time / UTC")
fig.savefig("quicklooks/underpasses.png", dpi=200, bbox_inches="tight")
# %%
