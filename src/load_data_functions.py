import pandas as pd
import xarray as xr
from pathlib import Path
from .post_processed_hamp_data import PostProcessedHAMPData


def load_hamp_data(path_radar, path_radiometer, path_iwv):
    hampdata = PostProcessedHAMPData(
        xr.open_dataset(path_radar),
        xr.open_dataset(path_radiometer),
        xr.open_dataset(path_iwv),
    )
    return hampdata


def load_dropsonde_data(path_dropsonde):
    print(f"Using dropsondes from: {path_dropsonde}")
    ds_dropsonde = xr.open_dataset(path_dropsonde)
    return ds_dropsonde


def load_dropsonde_data_for_date(dropsondes, date):
    """extract dropsondes from a particular fligth date. "dropsondes" is assumed to be
    complete (Level 3) dataset of all dropsondes, or a path to load a dropsonde dataset
    """

    date_start = pd.to_datetime(date)
    date_end = pd.to_datetime(date) + pd.Timedelta("24h")

    if isinstance(dropsondes, Path):
        ds = load_dropsonde_data(dropsondes)
    else:
        ds = dropsondes

    print(
        f"Extracting dropsondes with launchtime between {date_start} and {date_end} from dataset"
    )
    ds = ds.where(
        ds["launch_time_(UTC)"].astype("datetime64") >= date_start, drop=True
    ).where(ds["launch_time_(UTC)"].astype("datetime64") < date_end, drop=True)
    return ds
