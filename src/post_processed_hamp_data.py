class PostProcessedHAMPData:
    def __init__(
        self,
        radar,
        radiometers,
        column_water_vapour,
    ):
        self.radar = radar
        self.radiometers = radiometers
        self.column_water_vapour = column_water_vapour

    def __setitem__(self, key: str, data):
        if key == "radar":
            self.radar = data
        elif key == "radiometers":
            self.radiometers = data
        elif key == "column_water_vapour" or key == "cwv" or key == "CWV":
            self.column_water_vapour = data
        else:
            raise KeyError(f"no known key provided for assignment '{key}'")

    def __getitem__(self, key: str):
        if key == "radar":
            return self.radar
        elif key == "radiometers":
            return self.radiometers
        elif key == "column_water_vapour" or key == "cwv" or key == "CWV":
            return self.column_water_vapour
        else:
            raise KeyError(f"no known return provided for key '{key}'")

    def sel(self, timeslice, method="nearest"):
        cut_data = __class__(None, None, None)
        if self.radar is not None:
            cut_data["radar"] = self.radar.sel(time=timeslice, method=method)
        if self.radiometers is not None:
            cut_data["radiometers"] = self.radiometers.sel(
                time=timeslice, method=method
            )
        if self.column_water_vapour is not None:
            cut_data["column_water_vapour"] = self.column_water_vapour.sel(
                time=timeslice, method=method
            )

        return cut_data
