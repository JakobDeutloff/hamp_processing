# %% import modules
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import xarray as xr
import shutil
from orcestra.postprocess.level0 import (
    radiometer,
    radar,
    iwv,
    add_georeference,
)
from orcestra.postprocess.level1 import (
    correct_radar_height,
    filter_radar,
    filter_radiometer,
)

from src.ipfs_helpers import add_encoding


# %% define function


def postprocess_hamp(date, flightletter, version):
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
    paths["radar"] = config["root"] + config["radar"].format(
        date=date, flightletter=flightletter
    )
    paths["radiometer"] = config["root"] + config["radiometer"].format(
        date=date, flightletter=flightletter
    )
    paths["bahamas"] = config["bahamas"].format(date=date, flightletter=flightletter)
    paths["sea_land_mask"] = config["root"] + config["sea_land_mask"]
    paths["save_dir"] = config["root"] + config["save_dir"].format(
        date=date, flightletter=flightletter
    )

    # load raw data
    print(f"Loading raw data for {date}")
    ds_radar_raw = xr.open_mfdataset(paths["radar"]).load()
    ds_bahamas = (
        xr.open_dataset(paths["bahamas"], engine="zarr")
        .reset_coords(["lat", "lon", "alt"])
        .resample(time="0.25s")
        .mean()
    )
    ds_iwv_raw = xr.open_dataset(f"{paths['radiometer']}/KV/{date[2:]}.IWV.NC")
    radiometers = ["183", "11990", "KV"]
    ds_radiometers_raw = {}
    for radio in radiometers:
        ds_radiometers_raw[radio] = xr.open_dataset(
            f"{paths['radiometer']}/{radio}/{date[2:]}.BRT.NC"
        )

    # do level 1 processing
    print("Level 1 processing")
    ds_radar_lev1 = radar(ds_radar_raw).pipe(
        add_georeference,
        lat=ds_bahamas["lat"],
        lon=ds_bahamas["lon"],
        plane_pitch=ds_bahamas["pitch"],
        plane_roll=ds_bahamas["roll"],
        plane_altitude=ds_bahamas["alt"],
        source=ds_bahamas.attrs["source"],
    )
    ds_iwv_lev1 = iwv(ds_iwv_raw).pipe(
        add_georeference,
        lat=ds_bahamas["lat"],
        lon=ds_bahamas["lon"],
        plane_pitch=ds_bahamas["pitch"],
        plane_roll=ds_bahamas["roll"],
        plane_altitude=ds_bahamas["alt"],
        source=ds_bahamas.attrs["source"],
    )
    ds_radiometers_lev1 = {}
    for radio in radiometers:
        ds_radiometers_lev1[radio] = radiometer(ds_radiometers_raw[radio])

    # concatenate radiometers and add georeference
    ds_radiometers_lev1_concat = xr.concat(
        [ds_radiometers_lev1[radio] for radio in radiometers], dim="frequency"
    ).sortby("frequency")
    ds_radiometers_lev1_concat = ds_radiometers_lev1_concat.assign(
        TBs=ds_radiometers_lev1_concat["TBs"].T
    ).pipe(
        add_georeference,
        lat=ds_bahamas["lat"],
        lon=ds_bahamas["lon"],
        plane_pitch=ds_bahamas["pitch"],
        plane_roll=ds_bahamas["roll"],
        plane_altitude=ds_bahamas["alt"],
        source=ds_bahamas.attrs["source"],
    )

    # do level 2 processing
    print("Level 2 processing")
    sea_land_mask = xr.open_dataarray(paths["sea_land_mask"])
    ds_radar_lev2 = correct_radar_height(ds_radar_lev1).pipe(filter_radar)
    ds_radiometer_lev2 = filter_radiometer(
        ds_radiometers_lev1_concat, sea_land_mask=sea_land_mask
    )
    ds_iwv_lev2 = filter_radiometer(ds_iwv_lev1, sea_land_mask=sea_land_mask)

    # save data - delete if exists to prevent overwriting which can cause issues with zarr
    print(f"Saving data for {date}")
    ds_radar_lev2.attrs["version"] = version
    ds_radiometer_lev2.attrs["version"] = version
    ds_iwv_lev2.attrs["version"] = version
    # radar
    path_radar = f"{paths['save_dir']}/radar/HALO-{date}a_radar.zarr"
    if os.path.exists(path_radar):
        shutil.rmtree(path_radar)
    ds_radar_lev2.chunk(time=4**9).pipe(add_encoding).to_zarr(path_radar, mode="w")
    # radiometer
    path_radiometer = f"{paths['save_dir']}/radiometer/HALO-{date}a_radio.zarr"
    if os.path.exists(path_radiometer):
        shutil.rmtree(path_radiometer)
    ds_radiometer_lev2.chunk(time=4**9, frequency=-1).pipe(add_encoding).to_zarr(
        path_radiometer, mode="w"
    )
    # iwv
    path_iwv = f"{paths['save_dir']}/iwv/HALO-{date}a_iwv.zarr"
    if os.path.exists(path_iwv):
        shutil.rmtree(path_iwv)
    ds_iwv_lev2.chunk(time=4**9).pipe(add_encoding).to_zarr(path_iwv, mode="w")


# %% run postprocessing
dates = [
    "20241112",
    "20241114",
    "20241116",
    "20241119",
]

flightletters = [
    "b",
    "b",
    "a",
    "a",
]

version = "0.3"
for date, flightletter in zip(dates, flightletters):
    postprocess_hamp(date, flightletter, version)

# %%
