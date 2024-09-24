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
def read_radar(date):
    ds_radar = xr.open_dataset(
        f"Data/Hamp_Processed/radar/HALO-{date}a_radar.zarr", engine="zarr"
    )
    #  find time when earthcare crosses halo
    ec_track = ecfuncs.get_earthcare_track(date)
    ec_under_time = ecfuncs.find_ec_under_time(ec_track, ds_radar)

    plot_duration = pd.Timedelta("30m")
    ec_starttime, ec_endtime = (
        ec_under_time - plot_duration / 2,
        ec_under_time + plot_duration / 2,
    )
    timeframe = slice(ec_starttime, ec_endtime)
    ds_radar = ds_radar.sel(time=timeframe)
    return ds_radar, ec_under_time


def plot_cloudfraction(date, ax, ds_radar, ec_under_time):
    # calculate cloud fractions
    cf = (ds_radar["dBZe"].sel(height=slice(120, 15000)).max("height") > -30).astype(
        int
    )
    cf_total = cf.mean("time")

    # plot
    ax.fill_between(
        cf.time,
        cf * int(ds_radar.height.max()),
        color="grey",
        alpha=0.5,
        edgecolor=None,
    )
    pmesh = (
        ds_radar["dBZe"]
        .where(ds_radar["dBZe"] > -30)
        .plot.pcolormesh(
            ax=ax,
            x="time",
            y="height",
            add_colorbar=False,
            vmax=30,
            vmin=-30,
            cmap="viridis",
            extend="max",
        )
    )
    ax.set_title(f"{date}, CF: {(cf_total.values * 100).round(2)}%")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)

    # Set the x-axis formatter to show only the time
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.axvline(ec_under_time, color="r", linestyle="--", linewidth=1.0)

    return pmesh, cf_total


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
]
ec_under_times = {}
radar_data = {}
for date in dates:
    radar_data[date], ec_under_times[date] = read_radar(date)

# %% call plotfunction
cf_total = {}
fig, axes = plt.subplots(4, 4, figsize=(35, 18), sharey=True)
for date, ax in zip(dates, axes.flatten()):
    pmesh, cf_total[date] = plot_cloudfraction(
        date, ax, radar_data[date], ec_under_times[date]
    )

fig.colorbar(
    pmesh,
    ax=axes,
    label="Z / dBZe",
    aspect=40,
    extend="max",
)
fig.savefig("quicklooks/cloudfractions/all_days.png", dpi=300, bbox_inches="tight")
# %%
