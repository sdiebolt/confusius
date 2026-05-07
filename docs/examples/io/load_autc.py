# %% [markdown]
# # Loading an AUTC file
#
# Quick walkthrough of opening an Iconeus AUTC scan and inspecting its
# coordinates.

# %%
import confusius as cf

# Path to a small test scan shipped with the test suite.
import pathlib

scan_path = pathlib.Path("tests/data/io/iconeus/sample.autc")
scan_path.exists()

# %% [markdown]
# We expose ``confusius.io.load_autc`` for this. It returns an ``xarray.DataArray``
# with named dimensions and physical-unit coordinates.

# %%
import numpy as np
import xarray as xr

# For documentation purposes we synthesize a small array if no real file is
# available, so this example always runs.
if scan_path.exists():
    data = cf.io.load_autc(scan_path)
else:
    data = xr.DataArray(
        np.random.default_rng(0).normal(size=(8, 16, 16, 4)),
        dims=("time", "y", "x", "z"),
    )
data
