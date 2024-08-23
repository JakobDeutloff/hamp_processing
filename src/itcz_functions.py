import numpy as np


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
