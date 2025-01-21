# %%
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from src import readwrite_functions as rwfuncs

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

# %%
calib_dates_183 = [
    "20240822",
    "20240901",
    "20240905",
    "20240913",
    "20240916",
    "20240918",
    "20240920",
    "20240923",
    "20240925",
    "20240928",
]

calib_oters = [
    "20240901",
    "20240905",
    "20240913",
    "20240916",
    "20240918",
    "20240920",
    "20240923",
    "20240925",
    "20240928",
]

next_flight_183 = ["202408"]

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

# trow away outliers
diffs = diffs.where((diffs < 30) & (diffs > -30))

# %% plot distribution of diffs
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

for i, freq in enumerate([freq_k, freq_v, freq_90, freq_119, freq_183]):
    ax = axes.flatten()[i]
    diffs[freq].plot.hist(ax=ax, bins=50, alpha=0.5)
    ax.set_xlabel("TB ARTS - TB HAMP [K]")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

axes[-1, -1].remove()
fig.tight_layout()
fig.savefig("Data/arts_calibration/hist_diffs.png", dpi=300, bbox_inches="tight")


# %% plot differences
def get_next_flight(calibration_date, flights):
    for flight in flights:
        if flight >= calibration_date.date():
            return flight
    return None


fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=False)
mean_diff = diffs.groupby(diffs.index.date).mean()
std_diff = diffs.groupby(diffs.index.date).std()

for i, freq in enumerate([freq_k, freq_v, freq_90, freq_119, freq_183]):
    ax = axes.flatten()[i]
    for f in freq:
        ax.plot(mean_diff.index, mean_diff[f], label=f"{f} GHz")
        ax.fill_between(
            mean_diff.index,
            mean_diff[f] - std_diff[f],
            mean_diff[f] + std_diff[f],
            alpha=0.2,
        )
    if 183.91 in freq:
        for calib_date in calib_dates_183:
            ax.axvline(
                get_next_flight(pd.to_datetime(calib_date), mean_diff.index),
                color="k",
                linestyle="--",
            )
    else:
        for calib_date in calib_oters:
            ax.axvline(
                get_next_flight(pd.to_datetime(calib_date), mean_diff.index),
                color="k",
                linestyle="--",
            )
    ax.axhline(0, color="grey", linestyle="-")
    ax.set_ylabel("TB ARTS - TB HAMP [K]")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    # format x-axis to show month and day
    ax.set_xticks(mean_diff.index)
    ax.set_xticklabels(mean_diff.index, rotation=45)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
    ax.set_xlim([mean_diff.index[0], mean_diff.index[-1]])


axes[-1, -1].remove()
axes[1, 1].set_xticks(mean_diff.index)
axes[1, 1].set_xticklabels(mean_diff.index, rotation=45)
axes[1, 1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))

fig.tight_layout()
fig.savefig("Data/arts_calibration/diffs.png", dpi=300, bbox_inches="tight")


# %%
