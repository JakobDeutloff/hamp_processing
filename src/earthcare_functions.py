import pandas as pd
from orcestra import sat


def get_earthcare_track(date):
    day = pd.Timestamp(date)
    track = sat.SattrackLoader(
        "EARTHCARE", (day - pd.Timedelta("1d")).strftime(format="%Y-%m-%d"), kind="PRE"
    ).get_track_for_day(day.strftime(format="%Y-%m-%d"))

    track = track.where(track.time > (day + pd.Timedelta("1h")), drop=True)

    return track


def find_ec_under_time(track, ds_bahamas):
    def distance(lat_1, lon_1, lat_2, lon_2):
        return ((lat_1 - lat_2) ** 2 + (lon_1 - lon_2) ** 2) ** 0.5

    lat_halo = ds_bahamas["IRS_LAT"]
    lon_halo = ds_bahamas["IRS_LON"]
    lat_ec = track.lat
    lon_ec = track.lon
    dist = distance(lat_halo, lon_halo, lat_ec, lon_ec)
    return dist.idxmin().values


def add_earthcare_underpass(ax, ec_under_time, annotate=False):
    color = "r"
    ax.axvline(ec_under_time, color=color, linestyle="--", linewidth=1.0)

    if annotate:
        x, y = ec_under_time, ax.get_ylim()[1] * 0.975
        ax.annotate(" EarthCARE", xy=(x, y), xytext=(x, y), fontsize=10, color=color)
