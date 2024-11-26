# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from src import readwrite_functions as rwfuncs

# %% Read csv BTs
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
    "20240907",
    "20240909",
    "20240912",
    "20240914",
    "20240916",
    "20240919",
    "20240921",
    "20240923",
    "20240924",
    "20240926",
    "20240928",
]

TB_arts_list = []
TB_hamp_list = []
for date in dates:
    TB_arts_list.append(
        pd.read_csv(f"Data/arts_calibration/HALO-{date}a/TBs_arts.csv", index_col=0)
    )
    TB_hamp_list.append(
        pd.read_csv(f"Data/arts_calibration/HALO-{date}a/TBs_hamp.csv", index_col=0)
    )

# load dropsonde data
configfile = "config_ipns.yaml"
cfg = rwfuncs.extract_config_params(configfile)
ds_dropsonde = xr.open_dataset(cfg["path_dropsondes"], engine="zarr")

# %% restructure data
TB_arts = pd.concat(TB_arts_list, axis=1)
TB_hamp = pd.concat(TB_hamp_list, axis=1)
launch_time = ds_dropsonde.sel(sonde_id=TB_arts.columns).launch_time.values
TB_arts.columns = launch_time
TB_hamp.columns = launch_time
TB_arts = TB_arts.T
TB_hamp = TB_hamp.T

# %% calculate statistics for each flight
diffs = TB_arts - TB_hamp

# %% trow away outliers
diffs = diffs.where((diffs < 20) & (diffs > -20))

# %% define frequencies
freq_k = [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]
freq_v = [50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]
freq_90 = [90.0]
center_freq_119 = 118.75
center_freq_183 = 183.31
width_119 = [1.4, 2.3, 4.2, 8.5]
width_183 = [0.6, 1.5, 2.5, 3.5, 5.0, 7.5]
freq_119 = [center_freq_119 + w for w in width_119]
freq_183 = [center_freq_183 + w for w in width_183]


# %% plot differences
fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
mean_diff = diffs.groupby(diffs.index.date).mean()
std_diff = diffs.groupby(diffs.index.date).std()

for i, freq in enumerate([freq_k, freq_v, freq_90, freq_119, freq_183]):
    ax = axes.flatten()[i]
    for f in freq:
        ax.plot(mean_diff.index, mean_diff[f], label=f"{f} GHz")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_ylabel("TB ARTS - TB HAMP [K]")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    # format x-axis to show month and day
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
fig.tight_layout()
fig.savefig("Data/arts_calibration/diffs.png", dpi=300, bbox_inches="tight")


# %%
