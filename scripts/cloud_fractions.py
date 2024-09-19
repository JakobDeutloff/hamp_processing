import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr

# %% read data
date = "20240811"
ds_radar = xr.open_dataset(
    f"Data/Hamp_Processed/radar/HALO-{date}a_radar.zarr", engine="zarr"
)

# %% calculate cloud fractions
