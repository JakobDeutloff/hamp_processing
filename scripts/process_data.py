# %% import modules
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import xarray as xr
from orcestra.postprocess.level1 import fix_bahamas, fix_radiometer, fix_radar, fix_iwv
from orcestra.postprocess.level2 import (
    correct_radar_height,
    filter_radar,
    filter_radiometer,
)


# %% define function
def postprocess_hamp(date, version):
    """
    Postprocess raw data from HAMP.

    Uses the raw data from hamp and applies the following processing steps:
    - Level 1 processing
        - Fix bahamas data: Fixes time axis
        - Fix radar data: Fixes time axis, adds dBZ variables, adds georefernce from bahamas
        - Fix radiometer data: Fixes time axis, adds georefernce from bahamas
        - Fix iwv data: Fixes time axis, adds georefernce from bahamas
        - Concatenate radiometers: Concatenates radiometer data along frequency dimension
    - Level 2 processing
        - Correct radar height: Writes radar data on geometric height grid with 30m vertical grid spacing
        - Filter radar: Filters radar data
        - Filter radiometer: Filters radiometer data
    Saves the processed data in the save_dir folder and gives a version number.

    Parameters
    ----------
    date : str
        Date of the data in the format YYYYMMDD
    version : str
        Version number of the processed data

    Returns
    -------
    None
    """

    # load config file
    config_file = "process_config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # configure paths
    paths = {}
    paths["radar"] = config["root"] + config["radar"].format(date=date)
    paths["radiometer"] = config["root"] + config["radiometer"].format(date=date)
    paths["bahamas"] = config["root"] + config["bahamas"].format(date=date)
    paths["sea_land_mask"] = config["root"] + config["sea_land_mask"]
    paths["save_dir"] = config["root"] + config["save_dir"].format(date=date)

    # load raw data
    print(f"Loading rad data for {date}")
    ds_radar_raw = xr.open_mfdataset(paths["radar"]).load()
    ds_bahamas = xr.open_dataset(paths["bahamas"])
    ds_iwv_raw = xr.open_dataset(f"{paths['radiometer']}/KV/{date[2:]}.IWV.NC")
    radiometers = ["183", "11990", "KV"]
    ds_radiometers_raw = {}
    for radio in radiometers:
        ds_radiometers_raw[radio] = xr.open_dataset(
            f"{paths['radiometer']}/{radio}/{date[2:]}.BRT.NC"
        )

    # do level 1 processing
    print("Level 1 processing")
    ds_bahamas_lev1 = fix_bahamas(ds_bahamas)
    ds_radar_lev1 = fix_radar(ds_radar_raw, ds_bahamas_lev1)
    ds_iwv_lev1 = fix_iwv(ds_iwv_raw, ds_bahamas_lev1)
    ds_radiometers_lev1 = {}
    for radio in radiometers:
        ds_radiometers_lev1[radio] = fix_radiometer(
            ds_radiometers_raw[radio], ds_bahamas_lev1
        )

    # concatenate radiometers
    ds_radiometers_lev1_concat = xr.concat(
        [ds_radiometers_lev1[radio] for radio in radiometers], dim="frequency"
    )

    # do level 2 processing
    print("Level 2 processing")
    sea_land_mask = xr.open_dataarray(paths["sea_land_mask"])
    ds_radar_lev2 = correct_radar_height(ds_radar_lev1).pipe(filter_radar)
    ds_radiometer_lev2 = filter_radiometer(
        ds_radiometers_lev1_concat, sea_land_mask=sea_land_mask
    )
    ds_iwv_lev2 = filter_radiometer(ds_iwv_lev1, sea_land_mask=sea_land_mask)

    # save data
    print(f"Saving data for {date}")
    os.makedirs(paths["save_dir"], exist_ok=True)
    ds_radar_lev2.attrs["version"] = version
    ds_radiometer_lev2.attrs["version"] = version
    ds_iwv_lev2.attrs["version"] = version
    ds_radar_lev2.to_netcdf(f"{paths['save_dir']}/HALO-{date}a_radar_vs{version}.nc")
    ds_radiometer_lev2.to_netcdf(
        f"{paths['save_dir']}/HALO-{date}a_radiometer_vs{version}.nc"
    )
    ds_iwv_lev2.to_netcdf(f"{paths['save_dir']}/HALO-{date}a_iwv_vs{version}.nc")


# %% run postprocessing
date = "20240825"
version = "0.1"
postprocess_hamp(date, version)
