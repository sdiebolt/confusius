---
icon: lucide/brackets
---

# Working with Xarray

## Why Xarray?

A typical fUSI recording is a 4D array indexed by time, elevation (`z`), depth (`y`),
and lateral position (`x`). Storing this as a plain NumPy array means losing all of that
structure: axes become anonymous integers, physical coordinates must be tracked
separately, and keeping them in sync with the data after slicing or averaging is
error-prone. A custom wrapper class would address the labeling and coordinate tracking,
but at the cost of needing a complex reimplementation of many array operations and
losing access to the broader scientific Python ecosystem.

[Xarray](https://xarray.dev/) solves this by wrapping arrays with named dimensions and
physical coordinates, while [maintaining compatibility](https://xarray.dev/#ecosystem)
with the Python scientific ecosystem. With Xarray, operations become self-documenting:

```python
# NumPy: what does axis 0 mean here?
mean_volume = pwd.mean(axis=0)

# Xarray: unambiguous.
mean_volume = pwd.mean("time")
```

Coordinates also carry metadata through transformations, so voxel sizes, timestamps, and
acquisition parameters travel with the data rather than being stored separately.

```python
# Select a depth range by physical coordinate (mm), not by index.
shallow = pwd.sel(y=slice(0, 2.5))

# Coordinates are updated automatically, no manual bookkeeping needed.
shallow.coords["y"]  # [0.0, 0.1, ..., 2.5] mm
```

ConfUSIus builds on this by storing all fUSI recordings as `DataArray` objects, and by
extending the Xarray API with fUSI-specific operations through a custom accessor.

## Datasets and DataArrays

Xarray has two core data structures: `Dataset` and `DataArray`.

A **`Dataset`** is a dictionary-like container of multiple named arrays that share
the same coordinate system. When you open a Zarr archive, you get a `Dataset`:

```python
import xarray as xr
import confusius

ds = xr.open_zarr("power_doppler.zarr")
ds
```

```
<xarray.Dataset> Size: 76MB
Dimensions:        (time: 860, z: 1, y: 128, x: 86)
Coordinates:
  * time           (time) float64 7kB 0.299 0.899 1.499 ... 514.5 515.1 515.7
  * z              (z) float64 8B 0.0
  * y              (y) float64 1kB 5.664 5.713 5.762 5.811 ... 11.77 11.82 11.87
  * x              (x) float64 688B -3.583 -3.492 -3.402 ... 3.946 4.037 4.127
Data variables:
    power_doppler  (time, z, y, x) float64 76MB dask.array<chunksize=(215, 1, 32, 22), meta=np.ndarray>
```

A **`DataArray`** is a single variable from that collection—one array with its own
dimensions, coordinates, and attributes. You would get a `DataArray` if you opened a
NIfTI file through [`load_nifti`][confusius.io.load_nifti], or if you extract a
variable from a `Dataset`:

```python
pwd = ds["power_doppler"]
pwd
```

```
<xarray.DataArray 'power_doppler' (time: 860, z: 1, y: 128, x: 86)> Size: 76MB
dask.array<open_dataset-power_doppler, shape=(860, 1, 128, 86), dtype=float64, chunksize=(215, 1, 32, 22), chunktype=numpy.ndarray>
Coordinates:
  * time     (time) float64 7kB 0.299 0.899 1.499 2.099 ... 514.5 515.1 515.7
  * z        (z) float64 8B 0.0
  * y        (y) float64 1kB 5.664 5.713 5.762 5.811 ... 11.72 11.77 11.82 11.87
  * x        (x) float64 688B -3.583 -3.492 -3.402 -3.311 ... 3.946 4.037 4.127
Attributes: (11/16)
    transmit_frequency:           15625000.0
    probe_n_elements:             128
    probe_pitch:                  8.999999999999999e-05
    sound_velocity:               1520.0
    plane_wave_angles:            [-10.0, -9.310344696044922, -8.620689392089...
    ...                           ...
    clutter_filter_method:        svd_indices
    clutter_window_width:         300
    clutter_window_stride:        300
    doppler_window_width:         300
    doppler_window_stride:        300
    clutter_low_cutoff:           40
```

Reading the output from top to bottom, a `DataArray` has four components:

- **Dimensions** `(time, z, y, x)`: named axes in the order they appear in the
  underlying array. `time` is the temporal axis, `z` is elevation, `y` is depth (axial),
  and `x` is the lateral position.
- **Data**: the underlying array. For Zarr-backed data this is a Dask array, meaning
  values are not loaded into memory until you explicitly request them (e.g., by
  calling `.compute()` or accessing `.values`).
- **Coordinates**: physical values for each dimension, typically timestamps in seconds,
  spatial positions in millimeters. They are what enable `.sel(y=slice(0, 2.5))` to work
in physical units rather than array indices.
- **Attributes**: acquisition metadata as a flat key-value dictionary. Attributes are
  preserved through most ConfUSIus operations, and some are required for certain
  functions (e.g., `transmit_frequency` is needed for velocity calculations).

ConfUSIus operates on `DataArray` objects. The `Dataset` is only used as an entry point
when loading data, or to store multiple related variables together (e.g.,
`power_doppler` for the power Doppler signals and `brain_mask` for a corresponding brain
mask).

## The `.fusi` Accessor

Importing ConfUSIus registers the `.fusi` accessor on every `DataArray`:

```python
import xarray as xr
import confusius as cf  # Registers the .fusi accessor automatically.
```

The accessor is organized into six focused sub-accessors:

| Accessor | Description |
|---|---|
| [`.fusi.iq`][confusius.xarray.FUSIIQAccessor] | Process beamformed IQ into power Doppler or axial velocity volumes. |
| [`.fusi.scale`][confusius.xarray.FUSIScaleAccessor] | Scaling transformations: decibel, log, and power scaling. |
| [`.fusi.register`][confusius.xarray.FUSIRegistrationAccessor] | Motion correction via volumewise image registration. |
| [`.fusi.extract`][confusius.xarray.FUSIExtractAccessor] | Extract and reconstruct signals using spatial masks. |
| [`.fusi.plot`][confusius.xarray.FUSIPlotAccessor] | Visualization with Napari and carpet plots. |
| [`.fusi.io`][confusius.xarray.FUSIIOAccessor] | Save data to NIfTI with a JSON sidecar. |

The sub-accessors offer the same functions as the module-level API, but with an
intuitive syntax that allows quick operations directly on `DataArray` objects. They are
designed to be used for easy exploration and quick analyses, while the module-level
functions are available for more complex workflows where you might prefer explicit
function calls for readability.

### IQ Processing ([`.fusi.iq`][confusius.xarray.FUSIIQAccessor])

The [`.fusi.iq`][confusius.xarray.FUSIIQAccessor] accessor lets you access the
[`process_iq_to_power_doppler`][confusius.iq.process_iq_to_power_doppler] and
[`process_iq_to_axial_velocity`][confusius.iq.process_iq_to_axial_velocity] functions
directly on a `DataArray` containing beamformed IQ data. Refer to the [Beamformed IQ
guide](beamformed-iq.md) for background IQ processing.

```python
import dask
import xarray as xr

import confusius  # Registers the .fusi accessor.

ds = xr.open_zarr("iq.zarr")
iq = ds["iq"]

# Power Doppler with SVD clutter filtering (default).
pwd = iq.fusi.iq.process_to_power_doppler(
    clutter_window_width=200,
    doppler_window_width=100,
    low_cutoff=40,
)

# Axial velocity (in m/s).
velocity = iq.fusi.iq.process_to_axial_velocity(
    clutter_window_width=200,
    velocity_window_width=100,
)

(pwd, velocity) = dask.compute(pwd, velocity)  # Compute both in a single pass.
```

### Scaling ([`.fusi.scale`][confusius.xarray.FUSIScaleAccessor])

The [`.fusi.scale`][confusius.xarray.FUSIScaleAccessor] accessor provides common
scaling transformations: decibel, natural log, and power scaling.

```python
import numpy as np

pwd_db = pwd.fusi.scale.db()  # Default factor=10 for power quantities.
iq_db = np.abs(iq).fusi.scale.db(factor=20)  # Use factor=20 for amplitude quantities.

pwd_log = pwd.fusi.scale.log()

pwd_sqrt = pwd.fusi.scale.power(exponent=0.5)
```

Because the accessor returns a `DataArray`, it chains naturally with standard Xarray
operations:

```python
pwd_db = pwd.where(pwd > 0).fusi.scale.db()
```

### Registration ([`.fusi.register`][confusius.xarray.FUSIRegistrationAccessor])

The [`.fusi.register`][confusius.xarray.FUSIRegistrationAccessor] accessor provides easy
access to the [`register_volumewise`][confusius.registration.register_volumewise]
function for motion correction.

```python
registered = pwd.fusi.register.volumewise(reference_time=0)
```

By default, both translation and rotation are allowed (with a penalty to keep
rotations small). Pass `allow_rotation=False` for translation-only correction, or
increase `rotation_penalty` to constrain rotations more strongly.

### Signal Extraction ([`.fusi.extract`][confusius.xarray.FUSIExtractAccessor])

The [`.fusi.extract`][confusius.xarray.FUSIExtractAccessor] accessor provides access to
the [`extract_with_mask`][confusius.extract.extract_with_mask] and
[`unmask`][confusius.extract.unmask] functions for extracting signals from masked voxels
and reconstructing spatial volumes from extracted signals. This makes it easy to pass
fUSI data to scikit-learn, pandas, or other tools that expect a 2D matrix of shape
``(samples, features)``.

```python
mask = xr.open_zarr("brain_mask.zarr")["mask"]

# signals has dims (time, voxels).
signals = registered.fusi.extract.with_mask(mask)
```

For a quick round-trip,
[`.unstack("voxels")`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.unstack.html)
reconstructs the spatial dimensions within the bounding box of the mask. To reconstruct
the full spatial volume, use
[`.fusi.extract.unmask()`][confusius.xarray.FUSIExtractAccessor.unmask] with the
original mask:

```python
# signals has dims (time, voxels).
reconstructed = signals.fusi.extract.unmask(mask)
# reconstructed has dims (time, z, y, x).
```

For processed signals that are NumPy arrays (e.g., obtained from scikit-learn), use the
functional API instead:

```python
from sklearn.decomposition import PCA
from confusius.extract import unmask

pca = PCA(n_components=5).fit(signals)
components = pca.components_  # (5, voxels)

spatial_components = unmask(components, mask, new_dims=["component"])
# spatial_components has dims (component, z, y, x).
```

### Visualization ([`.fusi.plot`][confusius.xarray.FUSIPlotAccessor])

The [`.fusi.plot`][confusius.xarray.FUSIPlotAccessor] accessor provides easy access to
visualization functions for quick data inspection and quality control.

Display data in [Napari](https://napari.org/) (decibel-scaled by default):

```python
viewer, layer = registered.fusi.plot.napari(contrast_limits=(-20, 0))
```

Carpet plots show voxel time-series as a raster image, useful for quality control:

```python
fig, ax = registered.fusi.plot.carpet(mask=mask)
```

### Saving to Files ([`.fusi.io`][confusius.xarray.FUSIIOAccessor])

Save to NIfTI with an accompanying JSON sidecar that stores coordinates and
attributes:

```python
registered.fusi.io.to_nifti("sub-01_task-awake_pwd.nii.gz")
# Creates: sub-01_task-awake_pwd.nii.gz and sub-01_task-awake_pwd.json
```

## Complete Workflow Example

The following example shows a typical fUSI analysis from raw IQ to saved results:

```python
import xarray as xr
import confusius
from sklearn.decomposition import PCA
from confusius.extract import unmask

# 1. Load beamformed IQ data and corresponding brain mask.
iq = xr.open_zarr("iq.zarr")["iq"]
brain_mask = xr.open_zarr("brain_mask.zarr")["mask"]

# 2. Process IQ into power Doppler.
pwd = iq.fusi.iq.process_to_power_doppler(
    clutter_window_width=200,
    doppler_window_width=100,
    clutter_mask=brain_mask,
    low_cutoff=40,
)

# 3. Inspect in Napari (decibel-scaled by default).
viewer, layer = pwd.fusi.plot.napari(contrast_limits=(-20, 0))

# 4. Motion correction.
registered = pwd.fusi.register.volumewise(reference_time=0)

# 5. Quick quality check with a carpet plot.
fig, ax = registered.fusi.plot.carpet(mask=brain_mask)

# 6. Save registered power Doppler to NIfTI with JSON sidecar.
registered.fusi.io.to_nifti("sub-01_task-awake_pwd.nii.gz")

# 7. Extract brain voxel time-series.
signals = registered.fusi.extract.with_mask(brain_mask)

# 8. Decompose brain signals with PCA and map components back to brain space.
pca = PCA(n_components=5).fit(signals)
spatial_components = unmask(pca.components_, brain_mask, new_dims=["component"])
```

## API Reference

For full parameter documentation, see the [Xarray Integration API
reference](../api/xarray.md).
