import xarray as xr
import pandas as pd
import yaml
from pathlib import Path

from .post_processed_hamp_data import PostProcessedHAMPData


def extract_config_params(config_file):
    """Load configuration from YAML file,
    return dict with correctly formated configuration parameters"""

    config = {}
    with open(config_file, "r") as file:
        print(f"Reading config YAML: '{config_file}'")
        config_yaml = yaml.safe_load(file)

    config = get_config_params(config, config_yaml)

    return config


def formatted_data_path(config_yaml, pathname):
    """return formatted data path from reading a string which follows the
    HALO naming convention where flight date and flight letter are replaced with the
    keys from config"""
    path_str = config_yaml["paths"][pathname].format(
        date=config_yaml["date"], flightletter=config_yaml["flightletter"]
    )
    return Path(path_str)


def get_config_params(config, config_yaml):
    def get_formatted_path(pathname):
        if pathname in config_yaml["paths"]:
            try:
                path = formatted_data_path(config_yaml, pathname)
            except KeyError:
                path = Path(config_yaml["paths"][pathname])
        else:
            print(f"No '{pathname}' path found in config.yaml, path set to None")
            path = None
        return path

    # Extract REQUIRED parameters from the configuration
    dt, flght = config_yaml["date"], config_yaml["flightletter"]
    config["flightname"] = f"HALO-{dt}{flght}"
    config["date"] = config_yaml["date"]  # YYYYMMDD
    config["radiometer_date"] = config["date"][2:]  # YYMMDD
    config["is_planet"] = config_yaml["is_planet"]

    config["path_bahamas"] = get_formatted_path("bahamas")
    config["path_radiometer"] = get_formatted_path("radiometer")
    config["path_radar"] = get_formatted_path("radar")

    config["path_saveplts"] = get_formatted_path("saveplts")
    config["path_writedata"] = get_formatted_path("writedata")
    config["path_dropsondes_level3"] = get_formatted_path("dropsondes_level3")
    config["path_hampdata"] = get_formatted_path("hampdata")

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


def write_level1data_timeslice(
    hampdata: PostProcessedHAMPData, var, timeframe, ncfilename
):
    """
    writes slice of hampdata variable 'var' from starttime to endtime to a .nc file 'ncfilename'

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    var : str
        name of variable to slice
    timeframe : slice(starttime, endtime)
        Timeframe to plot.
    ncfilename:
        name of .nc file to write sliced data to
    """
    sliced_level1 = hampdata[var].sel(time=timeframe)
    sliced_level1.to_netcdf(ncfilename)

    sizemb_og = int(hampdata[var].nbytes / 1024 / 1024)  # convert bytes to MB
    sizemb_slice = int(sliced_level1.nbytes / 1024 / 1024)  # convert bytes to MB
    print(
        f"Timeslice of {var} data saved to: {ncfilename}\nOriginal data size = {sizemb_og}MB\nSliced size = {sizemb_slice}MB"
    )


def timeslice_all_level1hampdata(
    hampdata: PostProcessedHAMPData, timeframe, path_writedata
):
    """
    writes slice of (Level 1 post-processed) hampdata variable 'var' from starttime
    to endtime to a .nc file 'ncfilename'

    Parameters
    ----------
    hampdata : PostProcessedHAMPData
        Level 1 post-processed HAMP dataset
    timeframe : slice
        Timeframe to plot.
    """
    for var in ["flightdata", "radar", "183", "11990", "kv", "cwv"]:
        ncfilename = path_writedata / f"{var}_slice.nc"
        write_level1data_timeslice(hampdata, var, timeframe, ncfilename)


def load_timeslice_all_level1hampdata(
    path_slicedata, is_planet
) -> PostProcessedHAMPData:
    level1data = PostProcessedHAMPData(
        None, None, None, None, None, None, is_planet=is_planet
    )

    for var in ["flightdata", "radar", "183", "11990", "kv", "cwv"]:
        ncfilename = Path(path_slicedata) / f"{var}_slice.nc"
        level1data[var] = xr.open_mfdataset(ncfilename)

    return level1data
