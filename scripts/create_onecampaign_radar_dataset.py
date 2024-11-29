import xarray as xr
from pathlib import Path
from typing import List

all_flights = [
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


def onedataset_from_scratch(dataset_file: Path, flights: List[str]):
    ds = [
        xr.open_dataset(radar_dataset(f), engine="zarr").chunk("auto") for f in flights
    ]
    ds = xr.concat(ds, dim="time").chunk("auto")
    ds.to_zarr(dataset_file, mode="w")


def append_to_dataset_scratch(dataset_file: Path, flights: List[str]):
    ds = xr.open_dataset(dataset_file, engine="zarr").chunks("auto")
    add_ds = [
        xr.open_dataset(radar_dataset(f), engine="zarr").chunk("auto") for f in flights
    ]
    ds = xr.concat([ds, add_ds], dim="time").chunk("auto")
    ds.to_zarr(dataset_file, mode="w")


def main():
    """
    e.g.
    python scripts/create_onecampaign_radar_dataset.py ./level1.zarr True all
    or
    python scripts/create_onecampaign_radar_dataset.py ./level1.zarr False HALO-20240816a
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("is_new", type=bool)
    parser.add_argument("flights", nargs="+", type=str)
    args = parser.parse_args()

    if args.is_new:
        if args.flights[0] == "all":
            flights = all_flights
        else:
            flights = args.flights
        onedataset_from_scratch(args.dataset, flights)
    else:
        append_to_dataset_scratch(args.dataset, args.append_flights)


if __name__ == "__main__":
    exit(main())
