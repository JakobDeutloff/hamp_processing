# %%
import subprocess
import ruamel.yaml
from tqdm import tqdm


# %%
def path2ipfs(path):
    ret = subprocess.run(
        [
            "ipfs",
            "add",
            "--recursive",
            "--hidden",
            "--quieter",
            "--raw-leaves",
            path,
        ],
        capture_output=True,
    )
    return ret.stdout.decode().strip()


# %% write ipfs hashes to file
yaml = ruamel.yaml.YAML(typ="rt")
tree = yaml.load(
    open("/Users/jakobdeutloff/Programming/Orcestra/ipfs_tools/tree.yaml", "r")
)

dates = [
    "20240921",
    "20240923",
    "20240924",
    "20240926",
]

# Ensure the structure exists and initialize empty dicts if they don't exist
tree["products"]["HALO"]["radar"].setdefault("moments", {})
if tree["products"]["HALO"]["radar"]["moments"] is None:
    tree["products"]["HALO"]["radar"]["moments"] = {}
tree["products"]["HALO"].setdefault("radiometer", {})
if tree["products"]["HALO"]["radiometer"] is None:
    tree["products"]["HALO"]["radiometer"] = {}
tree["products"]["HALO"].setdefault("iwv", {})
if tree["products"]["HALO"]["iwv"] is None:
    tree["products"]["HALO"]["iwv"] = {}

# add the hashes to the tree
for date in tqdm(dates):
    radar = path2ipfs(f"Data/HAMP_Processed/radar/HALO-{date}a_radar.zarr")
    radiometer = path2ipfs(f"Data/HAMP_Processed/radiometer/HALO-{date}a_radio.zarr")
    iwv = path2ipfs(f"Data/HAMP_Processed/iwv/HALO-{date}a_iwv.zarr")
    tree["products"]["HALO"]["radar"]["moments"][f"HALO-{date}a.zarr"] = radar
    tree["products"]["HALO"]["radiometer"][f"HALO-{date}a.zarr"] = radiometer
    tree["products"]["HALO"]["iwv"][f"HALO-{date}a.zarr"] = iwv

# Save the updated tree back to the YAML file
with open(
    "/Users/jakobdeutloff/Programming/Orcestra/ipfs_tools/tree.yaml", "w"
) as file:
    yaml.dump(tree, file)

# %%
