import pandas as pd
from orcestra import sat
import pyproj
import warnings


def get_earthcare_track(date):
    day = pd.Timestamp(date)
    if day < pd.Timestamp("2024-09-06"):
        roi = "CAPE_VERDE"
    elif day < pd.Timestamp("2024-11-04"):
        roi = "BARBADOS"
    else:
        roi = "EUR"
    try:
        track = sat.SattrackLoader(
            "EARTHCARE",
            day.strftime(format="%Y-%m-%d"),
            kind="PRE",
            roi=roi,
        ).get_track_for_day(day.strftime(format="%Y-%m-%d"))
    except Exception:
        warnings.warn(f"No EarthCARE track found for {day}, falling back to day before")
        track = sat.SattrackLoader(
            "EARTHCARE",
            (day - pd.Timedelta("1d")).strftime(format="%Y-%m-%d"),
            kind="PRE",
            roi=roi,
        ).get_track_for_day((day).strftime(format="%Y-%m-%d"))

    track = track.sel(time=slice(day + pd.Timedelta("8h"), None))
    return track


def find_ec_under_time(ec_track, ds):
    """
    Returns time at closet overpass of HALO (from ds_bahamas) and EarthCARE track.
    Assumes HALO ds_bahamas has higher rate of temporal data.
    """
    geod = pyproj.Geod(ellps="WGS84")

    ds = ds.sel(time=slice(ec_track.time.min(), ec_track.time.max()))
    ec_track = ec_track.interp(time=ds.time)

    lat_halo = ds["lat"].fillna(-90)
    lon_halo = ds["lon"].fillna(-90)
    lat_ec = ec_track["lat"]
    lon_ec = ec_track["lon"]

    dist = geod.inv(lon_ec, lat_ec, lon_halo, lat_halo)[2]
    if dist.min() < 5e4:
        return ds.time[dist.argmin()].values
    else:
        return None


def add_earthcare_underpass(ax, ec_under_time, annotate=False):
    color = "r"
    ax.axvline(ec_under_time, color=color, linestyle="--", linewidth=1.0)

    if annotate:
        x, y = ec_under_time, ax.get_ylim()[1] * 0.975
        ax.annotate(" EarthCARE", xy=(x, y), xytext=(x, y), fontsize=10, color=color)
