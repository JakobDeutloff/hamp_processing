import numcodecs
import xarray as xr
import fsspec
import io


def get_chunks(dimensions):
    if "frequency" in dimensions:
        chunks = {
            "time": 4**8,
            "frequency": 5,
        }
    elif "height" in dimensions:
        chunks = {
            "time": 4**5,
            "height": 4**4,
        }
    else:
        chunks = {
            "time": 4**9,
        }

    return tuple((chunks[d] for d in dimensions))


def add_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)
    compressor = numcodecs.Blosc("zstd")

    for var in dataset.variables:
        if var not in dataset.dims:
            dataset[var].encoding = {
                "compressor": compressor,
                "dtype": "float32",
                "chunks": get_chunks(dataset[var].dims),
            }

    return dataset


def read_nc(url):
    with fsspec.open(url, "rb", expand=True) as fp:
        bio = io.BytesIO(fp.read())
        return xr.open_dataset(bio, engine="scipy")


async def get_client(**kwargs):
    import aiohttp

    conn = aiohttp.TCPConnector(limit=1)
    return aiohttp.ClientSession(connector=conn, **kwargs)
