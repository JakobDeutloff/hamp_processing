# %%i
import xarray as xr
import numpy as np

# %% read data

dates = [
    "20240811",
    "20240813",
    "20240816",
    "20240818",
    "20240821",
    "20240822",
    "20240825",
    "20240827",
    "20240829",
    "20240831",
    "20240903",
    "20240906",
]

flightrns = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

radi0183 = {}
radiokv = {}
radiokv11990 = {}
iwv = {}

path = "/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/Radiometer_Data/"


def read_radio_data(path, flightnr, date):
    ds_kv = xr.open_dataset(f"{path}/RF{flightnr}_{date}/KV/{date[2:]}.BRT.NC")
    ds_11990 = xr.open_dataset(f"{path}/RF{flightnr}_{date}/11990/{date[2:]}.BRT.NC")
    ds_183 = xr.open_dataset(f"{path}/RF{flightnr}_{date}/183/{date[2:]}.BRT.NC")
    ds_iwv = xr.open_dataset(f"{path}/RF{flightnr}_{date}/KV/{date[2:]}.IWV.NC")
    return ds_kv, ds_11990, ds_183, ds_iwv


for date, flightrn in zip(dates, flightrns):
    radiokv[date] = read_radio_data(path, flightrn, date)[0]
    radiokv11990[date] = read_radio_data(path, flightrn, date)[1]
    radi0183[date] = read_radio_data(path, flightrn, date)[2]
    iwv[date] = read_radio_data(path, flightrn, date)[3]

# %% count number of repetioins of each timestep

occurences_kv = {}
occurences_11990 = {}
occurences_183 = {}
occurences_iwv = {}


def count_occurences(ds):
    max = ds.time.to_series().value_counts().max()
    min = ds.time.to_series().value_counts().min()
    return np.max([max - 4, 4 - min])


for date in dates:
    occurences_kv[date] = count_occurences(radiokv[date])
    occurences_11990[date] = count_occurences(radiokv11990[date])
    occurences_183[date] = count_occurences(radi0183[date])
    occurences_iwv[date] = count_occurences(iwv[date])

# %% print results

print("KV")
for key, value in occurences_kv.items():
    print(f"{key}: {value}")

print("11990")
for key, value in occurences_11990.items():
    print(f"{key}: {value}")

print("183")
for key, value in occurences_183.items():
    print(f"{key}: {value}")

print("IWV")
for key, value in occurences_iwv.items():
    print(f"{key}: {value}")

# %%
