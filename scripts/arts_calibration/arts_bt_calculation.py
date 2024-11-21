# %%
import os
import sys
import xarray as xr
from src import load_data_functions as loadfuncs
from src.arts_functions import (
    run_arts,
    setup_workspace,
    extrapolate_dropsonde,
    get_profiles,
    average_double_bands,
    get_surface_temperature,
    get_surface_windspeed,
)
from src.plot_functions import plot_arts_flux
from src.ipfs_helpers import read_nc
from orcestra.postprocess.level0 import bahamas
from src import readwrite_functions as rwfuncs
import yaml
import pandas as pd
import numpy as np
import typhon
import pyarts
from tqdm import tqdm

# %% define frequencies
freq_k = [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]
freq_v = [50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]
freq_90 = [90.0]
center_freq_119 = 118.75
center_freq_183 = 183.31
width_119 = [1.4, 2.3, 4.2, 8.5]
width_183 = [0.6, 1.5, 2.5, 3.5, 5.0, 7.5]
freq_119 = [center_freq_119 - w for w in width_119] + [
    center_freq_119 + w for w in width_119
]
freq_183 = [center_freq_183 - w for w in width_183] + [
    center_freq_183 + w for w in width_183
]
all_freqs = freq_k + freq_v + freq_90 + freq_119 + freq_183
all_freqs = np.sort(all_freqs)

# %% setup workspace
print("Download ARTS data")
pyarts.cat.download.retrieve(verbose=True)
ws = setup_workspace()


# %% define function
def calc_arts_bts(date):
    """
    Calculates brightness temperatures for the radiometer frequencies with
    ARTS based on the dropsonde profiles for the flight on date.

    PARAMETERS
    ----------
    date: date on which flight took place (str)

    RETURN:
    ------
    None. Data is saved in arts_comparison folder.
    """

    print("Read Config")
    # change the date in config.yaml to date
    configfile = "config_ipns.yaml"
    with open(configfile, "r") as file:
        config_yml = yaml.safe_load(file)
    config_yml["date"] = date
    with open(configfile, "w") as file:
        yaml.dump(config_yml, file)
    # read config
    cfg = rwfuncs.extract_config_params(configfile)

    # load bahamas data from ipfs
    print("Load Bahamas Data")
    ds_bahamas = read_nc(
        f"ipns://latest.orcestra-campaign.org/raw/HALO/bahamas/{cfg['flightname']}/QL_{cfg['flightname']}_BAHAMAS_V01.nc"
    ).pipe(bahamas)

    # read dropsonde data
    print("Load Dropsonde Data")
    ds_dropsonde = xr.open_dataset(cfg["path_dropsondes"], engine="zarr")
    ds_dropsonde = ds_dropsonde.where(
        (ds_dropsonde["interp_time"] > pd.to_datetime(cfg["date"]))
        & (
            ds_dropsonde["interp_time"]
            < pd.to_datetime(cfg["date"]) + pd.DateOffset(hour=23)
        )
    ).dropna(dim="sonde_id", how="all")

    # read HAMP post-processed data
    print("Load HAMP Data")
    hampdata = loadfuncs.load_hamp_data(
        cfg["path_radar"], cfg["path_radiometers"], cfg["path_iwv"]
    )

    # cloud mask from radar
    print("Calculate Cloud Mask")
    ds_dropsonde = ds_dropsonde.assign(
        radar_cloud_flag=(
            hampdata.radar.sel(time=ds_dropsonde.launch_time, method="nearest").sel(
                height=slice(200, None)
            )["dBZe"]
            > -30
        ).max("height")
        * 1
    )
    cloud_free_idxs = (
        ds_dropsonde["sonde_id"]
        .where(ds_dropsonde["radar_cloud_flag"] == 0, drop=True)
        .values
    )

    # initialize result arrays
    freqs_hamp = hampdata.radiometers.frequency.values
    TBs_arts = pd.DataFrame(index=freqs_hamp, columns=cloud_free_idxs)
    TBs_hamp = TBs_arts.copy()
    dropsondes_extrap = []

    # setup folders
    print("Setup Folders")
    if not os.path.exists(f"Data/arts_calibration/{cfg['flightname']}"):
        os.makedirs(f"Data/arts_calibration/{cfg['flightname']}")
    if not os.path.exists(f"Data/arts_calibration/{cfg['flightname']}/plots"):
        os.makedirs(f"Data/arts_calibration/{cfg['flightname']}/plots")

    # loop over cloud free sondes
    print(f"Running {cloud_free_idxs.size} dropsondes for {cfg['flightname']}")
    for sonde_id in tqdm(cloud_free_idxs):
        # get profiles
        ds_dropsonde_loc, hampdata_loc, height, drop_time = get_profiles(
            sonde_id, ds_dropsonde, hampdata
        )

        # check if dropsonde is broken (contains only nan values)
        if ds_dropsonde_loc["ta"].isnull().mean().values == 1:
            print(f"Dropsonde {sonde_id} is broken, skipping")
            continue

        # get surface values
        surface_temp = get_surface_temperature(ds_dropsonde_loc)
        surface_ws = get_surface_windspeed(ds_dropsonde_loc)

        # extrapolate dropsonde profiles
        ds_dropsonde_extrap = extrapolate_dropsonde(
            ds_dropsonde_loc, height, ds_bahamas
        )
        dropsondes_extrap.append(ds_dropsonde_extrap)

        # run arts
        run_arts(
            pressure_profile=ds_dropsonde_extrap["p"].values,
            temperature_profile=ds_dropsonde_extrap["ta"].values,
            h2o_profile=typhon.physics.specific_humidity2vmr(
                ds_dropsonde_extrap["q"].values
            ),
            surface_ws=surface_ws,
            surface_temp=surface_temp,
            ws=ws,
            frequencies=all_freqs * 1e9,
            zenith_angle=180,
            height=height,
        )

        # get according hamp data
        TBs_hamp[sonde_id] = hampdata_loc.radiometers.TBs.values

        # average double bands
        TB_arts = pd.DataFrame(
            data=np.array(ws.y.value), index=np.float32(np.array(ws.f_grid.value) / 1e9)
        )
        TBs_arts[sonde_id] = average_double_bands(
            TB_arts,
            freqs_hamp,
        )

        # Plot to compare arts to hamp radiometers
        fig, ax = plot_arts_flux(
            TBs_hamp[sonde_id],
            TBs_arts[sonde_id],
            dropsonde_id=sonde_id,
            time=drop_time,
            ds_bahamas=ds_bahamas,
        )
        fig.savefig(f"Data/arts_calibration/{cfg['flightname']}/plots/{sonde_id}.png")

    # save results
    TBs_arts.to_csv(f"Data/arts_calibration/{cfg['flightname']}/TBs_arts.csv")
    TBs_hamp.to_csv(f"Data/arts_calibration/{cfg['flightname']}/TBs_hamp.csv")


# %% call function
date = str(sys.argv[1])
calc_arts_bts(date)

# %%
