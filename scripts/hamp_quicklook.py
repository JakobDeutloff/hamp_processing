# %
import pandas as pd
from src import plot_quicklooks as plotql
from src import load_data_functions as loadfuncs

### ----------- USER PARAMETERS YOU MUST SET ----------- ###
# flight name, date and letter (for data paths below)
flight = "RF01_20240811"
date = "20240811"
flightletter="a"

# paths to flight data (choose either planet or bahamas and set `is_planet` boolean accordingly)
is_planet=False
path_bahamas = (
    f"/Volumes/ORCESTRA/HALO-{date}{flightletter}/bahamas/QL_HALO-{date}{flightletter}_BAHAMAS_V01.nc"
)
# path_planet = "/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/planet_data/2024-08-16-TE03124060001-IWG1.csv"
path_radiometer = f"/Volumes/ORCESTRA/HALO-{date}{flightletter}/radiometer"
path_radar = f"/Users/yoctoyotta1024/Documents/c1_springsummer2024/orcestra/radar_data/{flight}/*.nc"

# path to directory to save quicklook .png and/or .pdf figures inside
savedir = f"/Users/yoctoyotta1024/Documents/c1_springsummer2024/orcestra/hamp_processing/quicklooks/{flight}"
### ---------------------------------------------------- ###

# % create HAMP post-processed data
hampdata = loadfuncs.do_post_processing(path_bahamas, path_radar, path_radiometer, date[2:], is_planet=is_planet)

# % produce HAMP single quicklook between startime and endtime
starttime, endtime = hampdata["183"].time[0].values, hampdata["183"].time[-1].values
is_savefig = True
savename = f"{savedir}/hamp_timesliceql_{flight}.png"
dpi=500
plotql.hamp_timeslice_quicklook(
    hampdata,
    timeframe=slice(starttime, endtime),
    flight=flight,
    figsize=(18, 18),
    savefigparams=[is_savefig, savename, dpi]
)

# % produce radiometer-only single quicklook between startime and endtime
starttime, endtime = hampdata["183"].time[0].values, hampdata["183"].time[-1].values
is_savefig = True
savename = f"{savedir}/radiometers_timesliceql_{flight}.png"
dpi=500
plotql.radiometer_quicklook(hampdata,
                            timeframe=slice(starttime, endtime),
                            figsize=(10, 14),
                            savefigparams=[is_savefig, savename, dpi])

# % produce hourly HAMP quicklooks
start_hour = pd.Timestamp(hampdata["183"].time[0].values).floor('h') # Round start time to full hour
end_hour = pd.Timestamp(hampdata["183"].time[-1].values).ceil('h') # Round end time to full hour
is_savepdf = True
plotql.hamp_hourly_quicklooks(hampdata, flight, start_hour, end_hour,
                               savepdfparams=[is_savepdf, savedir])
