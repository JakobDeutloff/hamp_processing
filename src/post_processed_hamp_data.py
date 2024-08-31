import pandas as pd


class PostProcessedHAMPData:
    def __init__(
        self,
        flightdata,
        radar,
        radio183,
        radio11990,
        radiokv,
        column_water_vapour,
        is_planet=False,
    ):
        self.flightdata = flightdata
        self.radar = radar
        self.radio183 = radio183
        self.radio11990 = radio11990
        self.radiokv = radiokv
        self.column_water_vapour = column_water_vapour

        self.is_planet = (
            is_planet  # False = using bahamas, True = using planet for flightdata
        )

    def __setitem__(self, key: str, data):
        if key == "flightdata" or key == "bahamas" or key == "planet":
            self.flightdata = data
        elif key == "radar":
            self.radar = data
        elif key == "radio183" or key == "183":
            self.radio183 = data
        elif key == "radio11990" or key == "11990":
            self.radio11990 = data
        elif key == "radiokv" or key == "kv" or key == "KV":
            self.radiokv = data
        elif key == "column_water_vapour" or key == "cwv" or key == "CWV":
            self.column_water_vapour = data
        elif key == "is_planet":
            self.is_planet = data
        elif key == "is_bahamas":
            self.is_planet = not data
        else:
            raise KeyError(f"no known key provided for assignment '{key}'")

    def __getitem__(self, key: str):
        if key == "flightdata" or key == "bahamas" or key == "planet":
            return self.flightdata
        elif key == "radar":
            return self.radar
        elif key == "radio183" or key == "183":
            return self.radio183
        elif key == "radio11990" or key == "11990":
            return self.radio11990
        elif key == "radiokv" or key == "kv" or key == "KV":
            return self.radiokv
        elif key == "column_water_vapour" or key == "cwv" or key == "CWV":
            return self.column_water_vapour
        elif key == "is_planet":
            return self.is_planet
        elif key == "is_bahamas":
            return not self.is_planet
        else:
            raise KeyError(f"no known return provided for key '{key}'")

    def sel(self, timeslice: pd.timedelta_range, method="nearest"):
        cut_data = __class__(
            None, None, None, None, None, None, is_planet=self.is_planet
        )
        if self.flightdata is not None:
            cut_data.flightdata = self.flightdata.sel(time=timeslice, method=method)
        if self.radar is not None:
            cut_data["radar"] = self.radar.sel(time=timeslice, method=method)
        if self.radio183 is not None:
            cut_data["radio183"] = self.radio183.sel(time=timeslice, method=method)
        if self.radio11990 is not None:
            cut_data["radio11990"] = self.radio11990.sel(time=timeslice, method=method)
        if self.radiokv is not None:
            cut_data["radiokv"] = self.radiokv.sel(time=timeslice, method=method)
        if self.column_water_vapour is not None:
            cut_data["column_water_vapour"] = (
                self.column_water_vapour.sel(time=timeslice, method=method),
            )
        return cut_data
