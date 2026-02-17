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
but at the cost of reimplementing the full array API and losing compatibility with the
broader scientific Python ecosystem.

[Xarray](https://xarray.dev/) solves this by wrapping arrays with named dimensions and
physical coordinates, while [maintaining compatibility](https://xarray.dev/#ecosystem)
with the broader Python scientific ecosystem. With Xarray, operations become
self-documenting:

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
Attributes: (12/17)
    voxdim:                       [0.4, 0.04883074509803921, 0.09070866141732...
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
| `.fusi.iq` | Process beamformed IQ into power Doppler or axial velocity volumes. |
| `.fusi.scale` | Scaling transformations: decibel, log, and power scaling. |
| `.fusi.register` | Motion correction via volumewise image registration. |
| `.fusi.extract` | Extract and reconstruct signals using spatial masks. |
| `.fusi.plot` | Visualization with Napari and carpet plots. |
| `.fusi.io` | Save data to NIfTI with a JSON sidecar. |

The sub-accessors offer the same functions as the module-level API, but with an
intuitive syntax that allows quick operations directly on `DataArray` objects. They are
designed to be used for easy exploration and quick analyses, while the module-level
functions are available for more complex workflows where you might prefer explicit
function calls for readability.

!!! info "Coming soon"
    This user guide is a work in progress. In the meantime, please refer to the [Xarray
    Integration API documentation](../../api/xarray.md) for a complete reference of all
    available functions and parameters.

## Complete Workflow Example

The following example shows a typical fUSI analysis from raw IQ to saved results:

```python
import xarray as xr
import confusius

# 1. Load beamformed IQ data.
iq = xr.open_zarr("iq.zarr")["iq"]

# 2. Process IQ into power Doppler.
pwd = iq.fusi.iq.process_to_power_doppler(
    clutter_window_width=200,
    doppler_window_width=100,
    low_cutoff=40,
)

# 3. Motion correction.
registered = pwd.fusi.register.volumewise(reference_time=0)

# 4. Quick quality check with a carpet plot.
fig, ax = registered.fusi.plot.carpet(mask=mask)

# 5. Inspect in Napari (decibel-scale by default).
viewer, layer = registered.fusi.plot.napari(
    contrast_limits=(-20, 0),
)

# 6. Extract brain signals with a mask for later analyses
mask = xr.open_zarr("brain_mask.zarr")["mask"]
signals = registered.fusi.extract.with_mask(mask)

# 7. Save registered power Doppler to NIfTI with JSON sidecar.
registered.fusi.io.to_nifti("sub-01_task-awake_pwd.nii.gz")
```

## API Reference

For full parameter documentation, see the [Xarray Integration API reference](../../api/xarray.md).
