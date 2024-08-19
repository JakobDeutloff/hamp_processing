import xarray as xr
import pandas as pd
from orcestra import sat
import yaml


def extract_config_params(config_file):
    """Load configuration from YAML file,
    return dict with correctly formated configuration parameters"""

    config = {}
    with open(config_file, "r") as file:
        config_yaml = yaml.safe_load(file)

    # Extract parameters from the configuration
    config["flight"] = config_yaml["flight"]
    config["date"] = config_yaml["date"]  # YYYYMMDD
    config["radiometer_date"] = config["date"][2:]  # YYMMDD
    config["flightletter"] = config_yaml["flightletter"]
    config["is_planet"] = config_yaml["is_planet"]

    # Format paths using the extracted parameters
    config["path_bahamas"] = config["paths"]["bahamas"].format(
        date=config["date"], flightletter=config["flightletter"]
    )
    config["path_radiometer"] = config["paths"]["radiometer"].format(
        date=config["date"], flightletter=config["flightletter"]
    )
    config["path_radar"] = config["paths"]["radar"].format(flight=config["flight"])
    config["savedir"] = config["savedir"].format(flight=config["flight"])

    return config


def read_planet(path):
    """Read planet data from csv file.

    Parameters:
    -----------
    path : str
        Path to csv file.

    Returns:
    --------
    xr.Dataset
        Planet data as xarray dataset.
    """
    ds = pd.read_csv(path)
    ds.set_index("issued", inplace=True)
    ds.index = pd.to_datetime(ds.index)
    ds.index = ds.index.tz_localize(None)
    ds.index.rename("time", inplace=True)
    ds = xr.Dataset.from_dataframe(ds)
    return ds


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
