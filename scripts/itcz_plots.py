# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import helper_functions as helpfuncs
import yaml


# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configyaml = sys.argv[1]
with open(configyaml, "r") as file:
    print(f"Reading config YAML: '{configyaml}'")
    cfg = yaml.safe_load(file)
path_hampdata = cfg["paths"]["hampdata"]
hampdata = helpfuncs.load_timeslice_all_level1hampdata(path_hampdata, cfg["is_planet"])
### ------------------------------------------------------------- ###

# %% load HAMP post-processed data slice

print(hampdata.radar)
