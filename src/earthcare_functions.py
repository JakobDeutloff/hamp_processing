import pandas as pd
from orcestra import sat
import pyproj


def get_earthcare_track(date):
    day = pd.Timestamp(date)
    track = sat.SattrackLoader(
        "EARTHCARE", (day - pd.Timedelta("1d")).strftime(format="%Y-%m-%d"), kind="PRE"
    ).get_track_for_day(day.strftime(format="%Y-%m-%d"))

    track = track.where(track.time > (day + pd.Timedelta("1h")), drop=True)

    return track


def find_ec_under_time(ec_track, ds):
    """
    Returns time at closet overpass of HALO (from ds_bahamas) and EarthCARE track.
    Assumes HALO ds_bahamas has higher rate of temporal data.
    """
    geod = pyproj.Geod(ellps="WGS84")

    ec_track = ec_track.interp(time=ds.time)

    lat_halo = ds["lat"].fillna(-90)
    lon_halo = ds["lon"].fillna(-90)
    lat_ec = ec_track["lat"]
    lon_ec = ec_track["lon"]

    dist = geod.inv(lon_ec, lat_ec, lon_halo, lat_halo)[2]

    return ds.time[dist.argmin()].values


def add_earthcare_underpass(ax, ec_under_time, annotate=False):
    color = "r"
    ax.axvline(ec_under_time, color=color, linestyle="--", linewidth=1.0)

    if annotate:
        x, y = ec_under_time, ax.get_ylim()[1] * 0.975
        ax.annotate(" EarthCARE", xy=(x, y), xytext=(x, y), fontsize=10, color=color)
