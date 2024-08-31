# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src import plot_quicklooks as plotql
from src import plot_functions as plotfuncs
from src import load_data_functions as loadfuncs
from src import readwrite_functions as rwfuncs
import matplotlib.pyplot as plt

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
flightname = cfg["flightname"]
path_saveplts = cfg["path_saveplts"]
### ------------------------------------------------------------- ###

# %% create HAMP post-processed data
hampdata = loadfuncs.do_post_processing(
    cfg["path_bahamas"],
    cfg["path_radar"],
    cfg["path_radiometer"],
    cfg["radiometer_date"],
    is_planet=cfg["is_planet"],
    do_radar=True,
    do_183=True,
    do_11990=True,
    do_kv=True,
    do_cwv=True,
)
# %% custom timeframe
starttime, endtime = "2024-08-27 18:37", "2024-08-27 18:48"
timeframe = slice(starttime, endtime)
savefig_format = "png"
savename = path_saveplts / "southern_circle.png"
dpi = 200
fig, axes = plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe,
    flightname,
    ec_under_time=None,
    figsize=(18, 12),
    savefigparams=[savefig_format, savename, dpi],
)
fig.tight_layout()

# %% plot only radar and CVW
starttime, endtime = "2024-08-27 18:37", "2024-08-27 18:48"
timeframe = slice(starttime, endtime)
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex="col", width_ratios=[1, 0.03])
ax1, cax = plotfuncs.plot_radar_timeseries(
    hampdata.sel(timeframe, method=None).radar, ax=axes[0, 0], fig=fig, cax=axes[0, 1]
)
ax2 = plotfuncs.plot_column_water_vapour_timeseries(
    hampdata.sel(timeframe, method=None).column_water_vapour.IWV,
    ax=axes[1, 0],
    target_cwv=None,
)
ax1.set_ylim(0.1, 4)
axes[1, 1].remove()
for ax in axes.flatten():
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].invert_xaxis()
fig.tight_layout()
fig.savefig(path_saveplts / "radar_cwv.png", dpi=300)

# %% produce hourly HAMP quicklooks
start_hour = pd.Timestamp(hampdata["183"].time[0].values).floor(
    "h"
)  # Round start time to full hour
end_hour = pd.Timestamp(hampdata["183"].time[-1].values).ceil(
    "h"
)  # Round end time to full hour
savefig_format = "png"
dpi = 100
plotql.hamp_hourly_quicklooks(
    hampdata,
    flightname,
    start_hour,
    end_hour,
    savefigparams=[savefig_format, path_saveplts, dpi],
)

# %%
