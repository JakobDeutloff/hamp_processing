import xarray as xr
import pandas as pd


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
