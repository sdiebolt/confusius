"""
Using Xarray with ConfUSIus
===========================

This example demonstrates the basic usage of ConfUSIus's Xarray accessor for
scaling operations on functional ultrasound imaging data.
"""

# %%
# Import necessary libraries
import numpy as np
import xarray as xr


# %%
# Create sample fUSI data
# -----------------------
# Let's create a simple DataArray representing power values from an ultrasound scan.

data = xr.DataArray(
    np.array([1, 10, 100, 1000, 10000]),
    dims=["sample"],
    coords={"sample": np.arange(5)},
    name="power",
    attrs={"units": "arbitrary", "description": "Sample ultrasound power data"},
)

data

# %%
# Apply decibel scaling
# ----------------------
# The most common transformation in ultrasound imaging is converting to decibel scale.
# By default, factor=10 is used for power quantities.

data_db = data.fusi.scale.db(factor=10)
data_db

# %%
# For amplitude quantities, use factor=20

data_db_20 = data.fusi.scale.db(factor=20)
data_db_20

# %%
# Apply logarithmic scaling
# --------------------------
# You can also apply natural logarithm scaling.

data_log = data.fusi.scale.log()
data_log

# %%
# Apply power scaling
# -------------------
# Power scaling is useful for visualization. The default exponent of 0.5 applies
# a square root transformation.

data_sqrt = data.fusi.scale.power(exponent=0.5)
data_sqrt

# %%
# Chain operations
# ----------------
# You can chain multiple xarray operations with the ConfUSIus accessor.

result = data.where(data > 50).fusi.scale.db()
result
