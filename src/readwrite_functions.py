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

    # Extract parameters from the configuration
    config["flight"] = config_yaml["flight"]
    config["date"] = config_yaml["date"]  # YYYYMMDD
    config["radiometer_date"] = config["date"][2:]  # YYMMDD
    config["flightletter"] = config_yaml["flightletter"]
    config["is_planet"] = config_yaml["is_planet"]

    # Format paths using the extracted parameters
    config["path_bahamas"] = Path(
        config_yaml["paths"]["bahamas"].format(
            date=config["date"], flightletter=config["flightletter"]
        )
    )
    config["path_radiometer"] = Path(
        config_yaml["paths"]["radiometer"].format(
            date=config["date"], flightletter=config["flightletter"]
        )
    )
    config["path_radar"] = Path(
        config_yaml["paths"]["radar"].format(flight=config["flight"])
    )

    try:
        config["path_saveplts"] = Path(
            config_yaml["paths"]["saveplts"].format(flight=config["flight"])
        )
    except KeyError as e:
        print(f"{e}\nNo 'saveplts' path found in config.yaml")

    try:
        config["path_writedata"] = Path(
            config_yaml["paths"]["writedata"].format(flight=config["flight"])
        )
    except KeyError as e:
        print(f"{e}\nNo 'writedata' path found in config.yaml")

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
