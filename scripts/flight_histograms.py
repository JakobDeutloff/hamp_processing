import xarray as xr

flights_sal = [
    "HALO-20240811a",
    "HALO-20240813a",
    "HALO-20240816a",
    "HALO-20240818a",
    "HALO-20240821a",
    "HALO-20240822a",
    "HALO-20240825a",
    "HALO-20240827a",
    "HALO-20240829a",
    "HALO-20240831a",
    "HALO-20240903a",
]
flights_tf = [
    "HALO-20240906a",
]
flights_bb = [
    "HALO-20240907a",
    "HALO-20240909a",
    "HALO-20240912a",
    "HALO-20240914a",
    "HALO-20240916a",
    "HALO-20240919a",
]
flights = flights_sal + flights_tf + flights_bb

flights = [
    "HALO-20240811a",
    "HALO-20240813a",
    "HALO-20240816a",
    "HALO-20240818a",
    "HALO-20240821a",
    "HALO-20240822a",
    "HALO-20240825a",
    "HALO-20240827a",
    "HALO-20240829a",
    "HALO-20240831a",
    "HALO-20240903a",
    "HALO-20240906a",
    "HALO-20240907a",
    "HALO-20240909a",
    "HALO-20240912a",
    "HALO-20240914a",
    "HALO-20240916a",
    "HALO-20240919a",
]


def radar_dataset(flight):
    return (
        f"ipns://latest.orcestra-campaign.org/products/HALO/radar/moments/{flight}.zarr"
    )


dataset = [
    xr.open_dataset(radar_dataset(f), engine="zarr").chunk("auto") for f in flights
]
ds = xr.concat(dataset, dim="time").chunk("auto")
ds.to_zarr("./level0", mode="w")
