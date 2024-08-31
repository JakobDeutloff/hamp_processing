import numpy as np


def get_all_clouds_flags_dropsondes(ds, coord="sonde_id", rh_name="rh"):
    """
    Function to derive cloud flags for dropsondes based on relative humidity.
    Developed by Nina Robbins

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the dropsonde data
    coord : str, optional
        Coordinate name for the dropsonde index, by default "sonde_id"
    rh_name : str, optional
        Name of the relative humidity variable in the dataset, by default "rh"

    Returns
    -------
    xarray.Dataset
        Dataset with cloud flags added as a new variable "cloud_flag"
    """

    min_rh_low = 0.92
    max_rh_low = 0.95
    max_rh_middle = 0.93

    min_moistlayer_thickness = 400
    min_moistlayer_base = 120

    min_cloudlayer_base = 280

    min_cloud_thickness_low = 30
    min_cloud_thickness_middle = 60

    cloud_flag_vec = np.full(len(ds[coord]), np.nan)

    for i, sonde in enumerate(ds[coord]):
        rh = ds[rh_name][i]

        rh_ml = rh.where(rh > min_rh_low)

        # Check if all values are nan, then there is no moist layer
        if rh_ml.isnull().all().item():
            cloud_flag = 0
            cloud_flag_vec[i] = cloud_flag
            continue

        else:
            # Keep only moist layer for this analysis
            rh_ml = rh_ml.where(~np.isnan(rh_ml), drop=True)

            base_ml = rh_ml.gpsalt[0]
            top_ml = rh_ml.gpsalt[-1]
            thickness_ml = top_ml - base_ml

            min_rh_low = 0.92
            max_rh_low = 0.95
            max_rh_middle = 0.93

            min_moistlayer_thickness = 400
            min_moistlayer_base = 120

            min_cloudlayer_base = 280

            min_cloud_thickness_low = 30
            min_cloud_thickness_middle = 60

            if (thickness_ml > min_moistlayer_thickness) and (
                base_ml > min_moistlayer_base
            ):
                # Check if max RH threshold for moist layer is exceeded --> cloud
                max_rh_ml = rh_ml.max()

                # Choose altitude range and max RH based on height of moist layer base
                if base_ml < 1300:
                    max_rh = max_rh_low
                    min_cloud_thickness = min_cloud_thickness_low
                    flag = 1

                if base_ml > 1300:
                    max_rh = max_rh_middle
                    min_cloud_thickness = min_cloud_thickness_middle
                    flag = 2

                # Check if the layer is cloudy
                if max_rh_ml > max_rh:
                    rh_cl = rh_ml.where(rh_ml > max_rh, drop=True)

                    # Check if cloud is thicker than threshold in total (includes separate cloud layers)
                    if len(rh_cl) > (min_cloud_thickness / 10):
                        # Check if cloud top is lower than threshold
                        if top_ml > min_cloudlayer_base:
                            cloud_flag = flag
                        else:
                            cloud_flag = 0
                    else:
                        cloud_flag = 0
                else:
                    cloud_flag = 0
            else:
                cloud_flag = 0

        # Add cloud flag to array
        cloud_flag_vec[i] = cloud_flag

    # Add cloud flags to dataset
    ds["cloud_flag"] = ((coord), cloud_flag_vec)
    ds["cloud_flag"] = ds.cloud_flag.assign_attrs(
        standard_name="cloud_flag",
        flag_meanings=["no_cloud", "low_cloud", "high_cloud"],
        flag_values=[0, 1, 2],
    )
    return ds
