import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr


def identify_itcz_crossings(cwv, thresholds=[45, 50]):
    """
    returns 'boolean-ish' type for whether CWV is inside/outside ITCZ:
    CWV < outside_itcz_threshold, value = 0;
    outside_itcz_threshold < CWV <= inside_itcz_threshold = 1;
    CWV > inside_itcz_threshold = or 2.
    """
    itcz_mask = np.where(cwv < thresholds[0], 0, 1)
    itcz_mask = np.where(cwv > thresholds[1], 2, itcz_mask)
    itcz_mask = np.where(np.isnan(cwv), np.nan, itcz_mask)

    return itcz_mask


def add_itcz_mask(fig, ax, xtime, itcz_mask, alpha=0.2, cbar=True, cax=False):
    colors = ["red", "gold", "green"]  # Red, Green, Blue
    cmap = LinearSegmentedColormap.from_list("three_color_cmap", colors)
    levels = [-0.5, 0.5, 1.5, 2.5]

    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 2)
    xx, yy = np.meshgrid(xtime, y)
    z = np.array([itcz_mask, itcz_mask])

    cont = ax.contourf(
        xx,
        yy,
        z,
        levels=levels,
        cmap=cmap,
        alpha=alpha,
    )
    clab = "ITCZ Mask"
    if cbar:
        if cax:
            cbar = fig.colorbar(cont, cax=cax, label=clab, shrink=0.8)
        else:
            cbar = fig.colorbar(cont, ax=ax, label=clab, shrink=0.8)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Outside", "Transition", "Inside"])


def interpolate_radiometer_mask_to_radar_mask(itcz_mask, hampdata):
    """returns mask for radar time dimension interpolated
    from mask with radiometer (CWV) time dimension"""
    ds_mask1 = xr.Dataset(
        {
            "itcz_mask": xr.DataArray(
                itcz_mask, dims=hampdata["CWV"].dims, coords=hampdata["CWV"].coords
            )
        }
    )
    ds_mask2 = ds_mask1.interp(time=hampdata.radar.time)

    return ds_mask2.itcz_mask
