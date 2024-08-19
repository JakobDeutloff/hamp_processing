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


def load_flightdata(path_flightdata, is_planet):
    if is_planet:
        ds_flightdata = read_planet(path_flightdata)
    else:  # bahamas
        ds_flightdata = xr.open_dataset(path_flightdata).pipe(bahamas)

    return ds_flightdata


def do_level0_processing_radar(path_radar):
    print(f"Using radar data from: {path_radar}")
    ds_radar_lev0 = xr.open_mfdataset(path_radar).load().pipe(radar)
    return ds_radar_lev0


def do_level0_processing_radiometer(path_radiometer):
    print(f"Using radiometer data from: {path_radiometer}")
    ds_radiometer_lev0 = xr.open_dataset(path_radiometer).pipe(radiometer)
    return ds_radiometer_lev0


def do_level1_processing_radar(ds_radar_lev0, flightdata, phi, the, alt):
    ds_radar_lev1 = ds_radar_lev0.pipe(filter_radar, flightdata[phi]).pipe(
        correct_radar_height,
        flightdata[phi],
        flightdata[the],
        flightdata[alt],
    )
    return ds_radar_lev1


def do_level1_processing_radiometer(ds_radiometer_lev0, flightdata, phi, alt):
    ds_radiometer_lev1 = ds_radiometer_lev0.pipe(
        filter_radiometer, flightdata[alt], flightdata[phi]
    )
    return ds_radiometer_lev1


def do_level0_processing(
    path_flightdata,
    path_radar,
    path_radiometer,
    radiometer_date,
    is_planet=False,
    do_radar=True,
    do_183=True,
    do_11990=True,
    do_kv=True,
) -> PostProcessedHAMPData:
    """read data and do level0 processing. arg radiometer_date usually excludes first YY in year,
    i.e. is format YYMMDD e.g. flight on YYYYMMDD 20240811 would have radiometer_date = 240811
    """

    level0data = PostProcessedHAMPData(
        None, None, None, None, None, is_planet=is_planet
    )

    level0data.flightdata = load_flightdata(path_flightdata, level0data.is_planet)

    if do_radar:
        level0data.radar = do_level0_processing_radar(path_radar)

    print(f"Radiometer datasets from: {path_radiometer}/[XXX]/{radiometer_date}.BRT.NC")
    for radio, do_radio in zip(["183", "11990", "kv"], [do_183, do_11990, do_kv]):
        if do_radio:
            path_radiometer = f"{path_radiometer}/{radio}/{radiometer_date}.BRT.NC"
            level0data[radio] = do_level0_processing_radiometer(path_radiometer)

    return level0data


def do_level1_processing(level0data: PostProcessedHAMPData) -> PostProcessedHAMPData:
    """do level1 processing on level0 data"""

    level1data = PostProcessedHAMPData(
        level0data.flightdata, None, None, None, None, level0data.is_planet
    )

    if level1data.is_planet:
        phi = "data.roll"
        the = "data.pitch"
        alt = "data.gps_msl_alt"
    else:  # bahamas
        phi = "IRS_PHI"
        the = "IRS_THE"
        alt = "IRS_ALT"

    if level0data.radar:
        level1data["radar"] = do_level1_processing_radar(
            level0data.radar, level0data.flightdata, phi, the, alt
        )

    for radio in ["183", "11990", "kv"]:
        if level0data[radio]:
            level1data[radio] = do_level1_processing_radiometer(
                level0data[radio], level0data.flightdata, phi, alt
            )

    return level1data


def do_post_processing(
    path_flightdata,
    path_radar,
    path_radiometer,
    radiometer_date,
    is_planet=False,
    do_radar=True,
    do_183=True,
    do_11990=True,
    do_kv=True,
) -> PostProcessedHAMPData:
    """do level0 and level1 processing"""

    level1data = do_level1_processing(
        do_level0_processing(
            path_flightdata,
            path_radar,
            path_radiometer,
            radiometer_date,
            is_planet=is_planet,
            do_radar=do_radar,
            do_183=do_183,
            do_11990=do_11990,
            do_kv=do_kv,
        )
    )

    return level1data
