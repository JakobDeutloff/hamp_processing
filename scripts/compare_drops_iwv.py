# %%
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# %% load dropsonde data
ds_dropsonde = xr.open_dataset(
    f"/Volumes/Upload/HALO/Dropsonde/20{date}/quickgrid/quickgrid-dropsondes-20{date}.nc"
)

# %% plot IWV
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ds_iwv.time, ds_iwv.IWV, label="KV Radiometer", color="dodgerblue")
ax.scatter(
    ds_dropsonde.launch_time,
    ds_dropsonde.iwv,
    label="Dropsondes",
    marker="*",
    c="#FF5349",
)

ax.set_xlabel("Time / UTC")
ax.set_ylabel("IWV / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title(f"20{date[:2]}-{date[2:4]}-{date[4:]}")

plt.show()
# %%
