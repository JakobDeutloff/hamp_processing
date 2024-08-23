# %%
import xarray as xr
import matplotlib.pyplot as plt
from orcestra.postprocess.level0 import bahamas
from orcestra.postprocess.level1 import filter_radiometer

# %% load radiometer and bahamas data
date = "240811"
ds_bahamas = (
    xr.open_mfdataset(f"/Volumes/ORCESTRA/HALO-20{date}a/bahamas/*.nc")
    .load()
    .pipe(bahamas)
)
ds_iwv = xr.open_dataset(
    f"/Volumes/ORCESTRA/HALO-20{date}a/radiometer/KV/{date}.IWV.NC"
).pipe(filter_radiometer, ds_bahamas["IRS_ALT"], ds_bahamas["IRS_PHI"])

# %% plot IWV
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ds_iwv.time, ds_iwv.IWV, label="KV Radiometer")
ax.set_xlabel("Time")
ax.set_ylabel("IWV / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)

# %%
