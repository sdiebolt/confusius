---
icon: lucide/brackets
---

# Using Xarray with ConfUSIus

## Overview

ConfUSIus provides a custom Xarray accessor that adds fUSI-specific functionality directly to Xarray DataArrays. This allows you to perform common fUSI operations using a convenient, chainable syntax.

The accessor is registered under the `.fusi` namespace and provides four main sub-accessors:

- **`.fusi.scale`**: Scaling transformations (decibel, log, power)
- **`.fusi.io`**: Input/output operations
- **`.fusi.plot`**: Visualization methods
- **`.fusi.register`**: Image registration

To use the accessor, simply import ConfUSIus:

```python
import xarray as xr
import confusius  # Registers the accessor automatically

# Now you can use data.fusi.* on any DataArray
data = xr.DataArray([1, 10, 100, 1000])
```

## Scaling Operations

The `.fusi.scale` accessor provides common scaling transformations used in ultrasound imaging analysis.

### Decibel scaling

The most common transformation in ultrasound imaging is converting to decibel scale. By default, `factor=10` is used for power quantities.

```python
import numpy as np
import xarray as xr
import confusius

# Create sample fUSI data
data = xr.DataArray(
    np.array([1, 10, 100, 1000, 10000]),
    dims=["sample"],
    coords={"sample": np.arange(5)},
    name="power",
    attrs={"units": "arbitrary", "description": "Sample ultrasound power data"},
)

# Apply decibel scaling (factor=10 for power quantities)
data_db = data.fusi.scale.db(factor=10)
# Result: [-40., -30., -20., -10., 0.] dB
```

For amplitude quantities, use `factor=20`:

```python
data_db_20 = data.fusi.scale.db(factor=20)
# Result: [-80., -60., -40., -20., 0.] dB
```

### Logarithmic scaling

Apply natural logarithm scaling:

```python
data_log = data.fusi.scale.log()
# Result: [0., 2.30, 4.61, 6.91, 9.21]
```

### Power scaling

Power scaling is useful for visualization. The default exponent of 0.5 applies a square root transformation:

```python
data_sqrt = data.fusi.scale.power(exponent=0.5)
# Result: [1., 3.16, 10., 31.62, 100.]
```

Use `exponent=2.0` for squaring, or any other power.

### Chaining operations

You can chain multiple xarray operations with the ConfUSIus accessor:

```python
result = data.where(data > 50).fusi.scale.db()
```

## IO Operations

The `.fusi.io` accessor provides convenient methods for saving fUSI data.

### Saving to NIfTI

Save DataArrays directly to NIfTI format with a JSON sidecar for metadata:

```python
import xarray as xr
import numpy as np
import confusius

# Create sample 4D fUSI data
data = xr.DataArray(
    np.random.rand(10, 32, 1, 64),
    dims=["time", "z", "y", "x"],
    coords={
        "time": np.arange(10),
        "z": np.arange(32) * 0.1,
        "y": [0.0],
        "x": np.arange(64) * 0.1 - 3.2,
    },
    attrs={"units": "mm/s"},
)

# Save via accessor (shorter syntax)
data.fusi.io.to_nifti("output.nii.gz")
```

This creates both `output.nii.gz` and `output.json` (sidecar with coordinates and metadata).

!!! tip "Module function alternative"
    You can also use the module-level function:
    
    ```python
    from confusius.io import save_nifti
    save_nifti(data, "output.nii.gz", save_sidecar=True)
    ```

## Plotting

The `.fusi.plot` accessor provides specialized plotting methods for fUSI data.

### Napari visualization

Display fUSI data in napari with automatic scaling:

```python
import xarray as xr
import confusius

# Load your data
data = xr.open_zarr("output.zarr")["iq"]

# Display with default decibel scaling
viewer, layer = data.fusi.plot.napari()

# Custom contrast limits
viewer, layer = data.fusi.plot.napari(contrast_limits=(-15, 0))

# Amplitude scaling (factor=20)
viewer, layer = data.fusi.plot.napari(
    scale_method="db", scale_kwargs={"factor": 20}
)

# No scaling
viewer, layer = data.fusi.plot.napari(scale_method=None)
```

### Carpet plots

Plot voxel intensities across time as a raster image (useful for quality control):

```python
import xarray as xr
import numpy as np
import confusius

data = xr.open_zarr("output.zarr")["iq"]

# Basic carpet plot
fig, ax = data.fusi.plot.carpet()

# Without detrending
fig, ax = data.fusi.plot.carpet(detrend=False)

# With mask (xarray)
mask = data.isel(time=0).pipe(np.abs) > threshold
fig, ax = data.fusi.plot.carpet(mask=mask)

# With mask (numpy array)
mask_array = np.ones(data.shape[1:], dtype=bool)  # (z, y, x)
mask_array[:10, :, :] = False  # Exclude first 10 z slices
fig, ax = data.fusi.plot.carpet(mask=mask_array)
```

## Registration

The `.fusi.register` accessor provides motion correction capabilities.

### Volumewise registration

Register all volumes to a reference time point:

```python
import xarray as xr
import confusius

data = xr.open_zarr("output.zarr")["pwd"]

# Register to first time point (translation + rotation)
registered = data.fusi.register.volumewise(reference_time=0)

# Translation only
registered = data.fusi.register.volumewise(
    reference_time=0, 
    allow_rotation=False
)

# With custom rotation penalty
registered = data.fusi.register.volumewise(
    reference_time=0, 
    allow_rotation=True,
    rotation_penalty=200.0  # Very strong constraint
)
```

!!! note "Parallel processing"
    Registration uses parallel processing by default (`n_jobs=-1`). Set `n_jobs=1` for serial processing if needed.

## Complete workflow example

Here's a complete example combining multiple accessor operations:

```python
import xarray as xr
import numpy as np
import confusius

# 1. Load data
ds = xr.open_zarr("recording.zarr")
iq_data = ds["iq"]

# 2. Compute power Doppler
power = np.abs(iq_data) ** 2

# 3. Apply decibel scaling for visualization
power_db = power.fusi.scale.db(factor=10)

# 4. Register for motion correction
registered = power_db.fusi.register.volumewise(reference_time=0)

# 5. Visualize in napari
viewer, layer = registered.fusi.plot.napari(
    contrast_limits=(-30, 0),
    colormap="viridis"
)

# 6. Save results to NIfTI
registered.fusi.io.to_nifti("registered_power.nii.gz")
```

## API Reference

For detailed documentation of all accessor methods, see the [Xarray Integration API Reference](../../api/xarray.md).
