import numpy as np
import pyarts
from scipy.optimize import curve_fit
import xarray as xr
import pandas as pd


def setup_workspace(verbosity=0):
    """Set up ARTS workspace.

    Returns:
        Workspace: ARTS workspace.
    """

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.iy_main_agendaSet(option="Emission")
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Absorption species
    ws.abs_speciesSet(
        species=[
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            "CO2, CO2-CKDMT252",
            "CH4",
            "O2,O2-CIAfunCKDMT100",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "O3",
        ]
    )

    # Read a line file and a matching small frequency grid
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # Read cross section data
    ws.ReadXsecData(basename="lines/")

    return ws


def run_arts(
    pressure_profile,
    temperature_profile,
    h2o_profile,
    ws: pyarts.workspace.Workspace,
    N2=0.78,
    O2=0.21,
    CO2=400e-6,
    CH4=1.8e-6,
    O3=1e-6,
    zenith_angle=180,
    height=1e4,
    surface_altitude=0.0,
    fmin=10e9,
    fmax=250e9,
    fnum=100,
    frequencies=None,
):
    """Perform a radiative transfer simulation.

    Parameters:
        species (list[str]): List of species tags.
        zenith_angle (float): Viewing angle [deg].
        height (float): Sensor height [m].
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.

    Returns:
        ndarray, ndarray, ndarray:
          Frequency grid [Hz], Brightness temperature [K], Optical depth [1]
    """

    # Set frequencies
    if frequencies is None:
        ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))
    else:
        ws.f_grid = np.array(frequencies)

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Non reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.1)
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")

    # Emissivity of sea surface
    ws.surface_emission = [[0]]

    # No sensor properties
    ws.sensorOff()

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    # Extract optical depth as auxiliary variables
    ws.ArrayOfStringSet(ws.iy_aux_vars, ["Optical depth"])

    # Atmosphere and surface
    ws.Touch(ws.lat_grid)
    ws.Touch(ws.lon_grid)
    ws.lat_true = np.array([0.0])
    ws.lon_true = np.array([0.0])

    ws.AtmosphereSet1D()
    ws.p_grid = pressure_profile
    ws.t_field = temperature_profile[:, np.newaxis, np.newaxis]

    vmr_field = np.zeros((6, len(pressure_profile), 1, 1))
    vmr_field[0, :, 0, 0] = h2o_profile
    vmr_field[1, :, 0, 0] = CO2
    vmr_field[2, :, 0, 0] = CH4
    vmr_field[3, :, 0, 0] = O2
    vmr_field[4, :, 0, 0] = N2
    vmr_field[5, :, 0, 0] = O3
    ws.vmr_field = vmr_field

    ws.z_surface = np.array([[surface_altitude]])
    ws.p_hse = 100000
    ws.z_hse_accuracy = 100.0
    ws.z_field = 16e3 * (5 - np.log10(pressure_profile[:, np.newaxis, np.newaxis]))
    ws.atmfields_checkedCalc()
    ws.z_fieldFromHSE()

    # Set surface temperature equal to the lowest atmosphere level
    ws.surface_skin_t = ws.t_field.value[0, 0, 0]
    ws.t_surface = [[ws.t_field.value[0, 0, 0]]]

    # Definition of sensor position and line of sight (LOS)
    ws.MatrixSet(ws.sensor_pos, np.array([[height]]))
    ws.MatrixSet(ws.sensor_los, np.array([[zenith_angle]]))

    # Perform RT calculations
    ws.propmat_clearsky_agendaAuto()  # Calculate the absorption coefficient matrix automatically
    ws.lbl_checkedCalc()  # checks if line-by-line parameters are ok
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.yCalc()

    return (
        ws.f_grid.value[:].copy(),
        ws.y.value[:].copy(),
        ws.y_aux.value[0][:].copy(),
    )


def exponential(x, a, b):
    return a * np.exp(b * x)


def linear(x, a, b):
    return a * x + b


def fit_exponential(x, y, p0):
    valid_idx = (~np.isnan(y)) & (~np.isnan(x))
    x = x[valid_idx]
    y = y[valid_idx]
    popt, pcov = curve_fit(exponential, x, y, p0=p0)
    return popt


def fit_linear(x, y):
    valid_idx = (~np.isnan(y)) & (~np.isnan(x))
    x = x[valid_idx]
    y = y[valid_idx]
    popt, pcov = curve_fit(linear, x, y)
    return popt


def fill_upper_levels(x, y, popt, func):
    offset = y.dropna("alt").values[-1]
    nan_vals = np.isnan(y)
    idx_nan = np.where(nan_vals)[0]
    nan_vals[idx_nan - 1] = True  # get overlap of one
    filled = np.zeros_like(y)
    filled[~nan_vals] = y[~nan_vals]
    new_vals = func(x[nan_vals], *popt)
    filled[nan_vals] = new_vals - new_vals[0] + offset
    return filled


def get_profiles(sonde_id, ds_dropsonde, hampdata):
    ds_dropsonde_loc = ds_dropsonde.sel(sonde_id=sonde_id)
    drop_time = ds_dropsonde_loc["interp_time"].dropna("alt").min().values
    hampdata_loc = hampdata.sel(timeslice=drop_time, method="nearest")
    height = float(hampdata_loc.flightdata["IRS_ALT"].values)
    return ds_dropsonde_loc, hampdata_loc, height, pd.to_datetime(drop_time)


def extrapolate_dropsonde(ds_dropsonde, height):
    ds_dropsonde = ds_dropsonde.where(ds_dropsonde["alt"] < height, drop=True)
    # drop nans at lower levels
    bool = (ds_dropsonde["p"].isnull()) & (ds_dropsonde["alt"] < 100)
    ds_dropsonde = ds_dropsonde.where(~bool, drop=True)
    popt_p = fit_exponential(
        ds_dropsonde["alt"].values, ds_dropsonde["p"].values, p0=[1e5, -0.0001]
    )
    popt_ta = fit_linear(ds_dropsonde["alt"].values, ds_dropsonde["ta"].values)

    p_extrap = fill_upper_levels(
        ds_dropsonde["alt"].values,
        ds_dropsonde["p"].interpolate_na("alt"),
        popt_p,
        exponential,
    )
    ta_extrap = fill_upper_levels(
        ds_dropsonde["alt"].values,
        ds_dropsonde["ta"].interpolate_na("alt"),
        popt_ta,
        linear,
    )
    q_extrap = ds_dropsonde["q"].interpolate_na("alt").values
    q_extrap[np.isnan(q_extrap)] = q_extrap[~np.isnan(q_extrap)][-1]

    return xr.Dataset(
        {
            "p": (("alt"), p_extrap),
            "ta": (("alt"), ta_extrap),
            "q": (("alt"), q_extrap),
        },
        coords={"alt": ds_dropsonde["alt"].values},
    )
