# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import load_data_functions as loadfuncs
from src import readwrite_functions as rwfuncs
import numpy as np

import matplotlib.pyplot as plt
import xarray as xr
import yaml
from src.plot_functions import plot_radiometer_timeseries, plot_radar_timeseries
import src.plot_quicklooks as plotql
from datetime import datetime

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
path_saveplts = cfg["path_saveplots"]
flightname = cfg["flightname"]
### ------------------------------------------------------------- ###

# %% create HAMP post-processed data
hampdata = loadfuncs.load_hamp_data(
    cfg["path_radar"], cfg["path_radiometers"], cfg["path_iwv"]
)

# %% load raw radiometer data
ds_rad_raw = {}
for radio in ["11990", "KV", "183"]:
    ds_rad_raw[radio] = xr.open_dataset(
        f"Data/Radiometer_Data/HALO-{cfg['date']}a/{radio}/{cfg['date'][2:]}.LV0.NC"
    )

# %% plot radiometer data


def define_module(radio):
    if (radio == "K") | (radio == "V"):
        module = "KV"
        is_90 = False
    elif radio == "183":
        module = "183"
        is_90 = False
    elif radio == "90":
        module = "11990"
        is_90 = True
    elif radio == "119":
        module = "11990"
        is_90 = False
    else:
        raise ValueError("Invalid radiometer frequency")
    return module, is_90


freqs = {
    "K": slice(22, 32),
    "V": slice(50, 58),
    "183": slice(183, 191),
    "119": slice(120, 128),
    "90": 90,
}
radio = "119"
module, is_90 = define_module(radio)
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col", width_ratios=[10, 0.2])
plot_radar_timeseries(hampdata.radar, ax=axes[0, 0], fig=fig, cax=axes[0, 1])
plot_radiometer_timeseries(
    hampdata.radiometers.sel(frequency=freqs[radio])["TBs"], ax=axes[1, 0], is_90=is_90
)
ds_rad_raw[module].sel(frequency=freqs[radio])["gain"].plot.line(
    ax=axes[2, 0], x="time"
)
axes[1, 1].remove()
axes[2, 1].remove()
plt.show()

# %% filter radiometers

ds = hampdata
date = cfg["date"]
# read error file
with open(f"Data/error_files/HALO-{date}a.yaml") as file:
    error_file = yaml.safe_load(file)

# filter data
date_format = "%H:%M:%S"
variables_2d = [
    var for var in ds.radiometers.data_vars if ds.radiometers[var].ndim == 2
]
for radio in ["K", "V", "90", "119", "183"]:
    if radio in error_file.keys():
        for timeframe in error_file[radio]["times"]:
            start_time = datetime.strptime(timeframe[0], date_format).time()
            end_time = datetime.strptime(timeframe[1], date_format).time()
            mask = (ds.radiometers.time.dt.time < start_time) | (
                ds.radiometers.time.dt.time > end_time
            )
            for var in variables_2d:
                value = ds.radiometers[var]
                for f in ds.radiometers.sel(frequency=freqs[radio]).frequency.values:
                    value.loc[{"frequency": f}] = value.loc[{"frequency": f}].where(
                        mask
                    )
                ds.radiometers.assign({var: value})

# %% control plot
flight_starttime, flight_endtime = (
    ds.radiometers.time[0].values,
    ds.radiometers.time[-1].values,
)
timeframe = slice(flight_starttime, flight_endtime)
plotql.hamp_timeslice_quicklook(
    ds,
    timeframe,
    flightname,
    ec_under_time=None,
    figsize=(18, 12),
    savefigparams=[],
)
plt.show()
# %% calculate TB
f = 120.15  # frequency in GHz
ds = ds_rad_raw["11990"].sel(frequency=f)
TB = ds["DetVolts"] / ds["gain"] - ds["TRec"]
TB_meas = hampdata.radiometers["TBs"].sel(frequency=f)

# %% convert TB from radiance to brightness temperature
h = 6.62607015e-34  # Planck constant
c = 299792458  # speed of light
k = 1.380649e-23  # Boltzmann constant
lamb = c / (f * 1e9)  # wavelength in meters
TB = ((h * c) / (lamb * k)) * (1 / np.log((2 * h * c**2 / (lamb**5 * TB)) + 1))


# %%
