# %%
import matplotlib.pyplot as plt
from src.load_data_functions import (
    do_level0_processing_column_water_vapour,
    do_level1_processing_column_water_vapour,
    load_dropsonde_data,
    load_flightdata,
)
from src.plot_functions import plot_dropsonde_iwv_comparison

# %% load data for all flights
dates = ["240811", "240813", "240816", "240818", "240821", "240822"]
bahamas = {}
iwv = {}
drops = {}
for date in dates:
    bahamas[date] = load_flightdata(
        f"/Volumes/ORCESTRA/HALO-20{date}a/bahamas/QL_HALO-20{date}a_BAHAMAS_V01.nc",
        is_planet=False,
    )
    iwv[date] = do_level0_processing_column_water_vapour(
        f"/Volumes/ORCESTRA/HALO-20{date}a/radiometer/KV/{date}.IWV.NC"
    ).pipe(
        do_level1_processing_column_water_vapour, bahamas[date], "IRS_PHI", "IRS_ALT"
    )
    drops[date] = load_dropsonde_data(
        f"/Volumes/Upload/HALO/Dropsonde/20{date}/quickgrid/quickgrid-dropsondes-20{date}.nc"
    )

# %% calculate bias
bias = {}
for date in dates:
    bias[date] = (iwv[date].IWV.mean() - drops[date].iwv.mean()).values
mean_bias = sum(bias.values()) / len(bias)

# %% plot IWV
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="all")
for date, ax in zip(dates, axes.flatten()):
    plot_dropsonde_iwv_comparison(iwv[date], drops[date], bias[date], date, ax)

for ax in axes[:, 0]:
    ax.set_ylabel("IWV / kg m$^{-2}$")
for ax in axes[1, :]:
    ax.set_xlabel("Time")

fig.subplots_adjust(bottom=0.13)
fig.legend(["KV Radiometer", "Dropsondes"], bbox_to_anchor=(0.6, 0.05), ncol=2)
fig.text(0.8, 0.03, f"Mean Bias: {mean_bias:.3f} kg m$^{-2}$", ha="center", va="center")
fig.savefig("quicklooks/iwv_dropsonde_comparison.png", dpi=300, bbox_inches="tight")

# %%
