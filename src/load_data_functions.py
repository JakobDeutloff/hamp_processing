# %
import xarray as xr
from orcestra.postprocess.level0 import bahamas, radiometer, radar
from orcestra.postprocess.level1 import (
    filter_radiometer,
    filter_radar,
    correct_radar_height,
)
from .helper_functions import read_planet
from .post_processed_hamp_data import PostProcessedHAMPData


def do_level0_processing(
    path_flightdata, path_radar, path_radiometer, radiometer_date, is_planet=False
):
    """read data and do level0 processing. arg radiometer_date usually excludes first YY in year,
    i.e. is format YYMMDD e.g. flight on YYYYMMDD 20240811 would have radiometer_date = 240811
    """
    if is_planet:
        ds_flightdata = read_planet(path_flightdata)
    else:  # bahamas
        ds_flightdata = xr.open_dataset(path_flightdata).pipe(bahamas)

    print(f"Using radar data from: {path_radar}")
    ds_radar_lev0 = xr.open_mfdataset(path_radar).load().pipe(radar)

    print(
        f"Using radiometer data from: {path_radiometer}/[XXX]/{radiometer_date}.BRT.NC"
    )
    ds_183_lev0 = xr.open_dataset(
        f"{path_radiometer}/183/{radiometer_date}.BRT.NC"
    ).pipe(radiometer)
    ds_11990_lev0 = xr.open_dataset(
        f"{path_radiometer}/11990/{radiometer_date}.BRT.NC"
    ).pipe(radiometer)
    ds_kv_lev0 = xr.open_dataset(f"{path_radiometer}/KV/{radiometer_date}.BRT.NC").pipe(
        radiometer
    )

    level0data = PostProcessedHAMPData(
        ds_flightdata, ds_radar_lev0, ds_183_lev0, ds_11990_lev0, ds_kv_lev0
    )
    return level0data


def do_level1_processing(
    level0data: PostProcessedHAMPData, is_planet=False
) -> PostProcessedHAMPData:
    """do level1 processing"""

    level1data = PostProcessedHAMPData(level0data.flightdata, None, None, None, None)

    if is_planet:
        phi = "data.roll"
        the = "data.pitch"
        alt = "data.gps_msl_alt"
    else:  # bahamas
        phi = "IRS_PHI"
        the = "IRS_THE"
        alt = "IRS_ALT"

    level1data["radar"] = (
        level0data["radar"]
        .pipe(filter_radar, level0data.flightdata[phi])
        .pipe(
            correct_radar_height,
            level0data.flightdata[phi],
            level0data.flightdata[the],
            level0data.flightdata[alt],
        )
    )

    for radio in ["183", "11990", "kv"]:
        level1data[radio] = level0data[radio].pipe(
            filter_radiometer, level0data.flightdata[alt], level0data.flightdata[phi]
        )

    return level1data


def do_post_processing(
    path_flightdata, path_radar, path_radiometer, radiometer_date, is_planet=False
):
    """do level0 and level1 processing"""

    level1data = do_level1_processing(
        do_level0_processing(
            path_flightdata,
            path_radar,
            path_radiometer,
            radiometer_date,
            is_planet=is_planet,
        ),
        is_planet=is_planet,
    )
    return level1data
