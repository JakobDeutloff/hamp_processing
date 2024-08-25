# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src import load_data_functions as loadfuncs
from src import readwrite_functions as rwfuncs

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configfile = "config.yaml"
cfg = rwfuncs.extract_config_params(configfile)
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

# %% write slice of HAMP post-processed data
starttime = hampdata["radar"].time[0].values + pd.Timedelta("1H")
endtime = starttime + pd.Timedelta("4H")
rwfuncs.timeslice_all_level1hampdata(
    hampdata, slice(starttime, endtime), cfg["path_writedata"]
)

# %% load HAMP post-processed data slice
path_slicedata = cfg["path_writedata"]
hampdata_slice = rwfuncs.load_timeslice_all_level1hampdata(
    path_slicedata, cfg["is_planet"]
)
print(hampdata_slice)
