# %%
import os
import numpy as np

path = os.getcwd()

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

for date in dates:
    print("-----node line -----")
    os.system(f"sbatch {path}/scripts/arts_calibration/call_bt_calculation.sh {date}")