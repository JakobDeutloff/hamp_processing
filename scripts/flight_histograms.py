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


def get_ipfs_flight(flight):
    return f"ipns://latest.orcestra-campaign.org/products/HALO/radar/moments/{flight}"


flights = {f: get_ipfs_flight(f) for f in flights}
for k, v in flights.items():
    print(k, v)
