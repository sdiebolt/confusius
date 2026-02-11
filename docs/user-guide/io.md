---
icon: lucide/file-input
---

# Input/Output

## Overview

ConfUSIus is designed to handle large-scale fUSI datasets efficiently. To achieve this,
it relies on [Xarray](https://docs.xarray.dev/) and
[Zarr](https://zarr.readthedocs.io/).

- **Xarray**: Used for labeled multi-dimensional arrays. This allows you to access data
  using meaningful axis names (e.g., ``time``, ``x``) and coordinates (e.g., time in
  seconds, depth in mm) rather than raw array indices.
- **Zarr**: A cloud-native, chunked, compressed, N-dimensional array format. It allows
  ConfUSIus to process datasets that are larger than memory (out-of-core processing) by
  loading only the necessary chunks of data.

ConfUSIus is designed to handle the full fUSI workflow. You may load beamformed IQ data
to perform clutter filtering and compute power Doppler signals, or work from
already-processed acquisition data (e.g., power Doppler or velocity acquisitions) and
jump straight to denoising and statistical analysis.

## Supported beamformed IQ formats

ConfUSIus requires beamformed IQ data to be in an Xarray-compatible format (e.g., Zarr) 
to leverage its full capabilities[^iq-dask]. Currently, ConfUSIus supports automatic
conversion of beamformed IQ data obtained from AUTC and EchoFrame systems to an
Xarray-compatible Zarr format. 

!!! question "Using beamformed IQ data from other sources?"
    You may also convert beamformed IQ data to any Xarray-compatible format (e.g.,
    netCDF, HDF5, memory-mapped NumPy arrays) and use them with ConfUSIus, but AUTC and
    EchoFrame formats are currently the only ones with built-in conversion utilities.
    Other formats may be supported in the future, and contributions are welcome. 

=== "AUTC DATs"

    This format consists of a series of binary `.dat` files (often split into parts),
    where each file contains multiple acquisition blocks.

    To convert a folder of AUTC DAT files to Zarr, use the
    [`convert_autc_dats_to_zarr`][confusius.io.convert_autc_dats_to_zarr] function.

    ```python
    from confusius.io import convert_autc_dats_to_zarr

    convert_autc_dats_to_zarr(
        dats_root="path/to/data_folder",
        output_path="sub-01_task-awake_iq.zarr",
        # Optional: specify block start times, transmit frequency, axis coordinates, and
        # other metadata via keyword arguments (see API for details).
        block_times=block_times,
        compound_sampling_frequency=500.0,
        transmit_frequency=15.625e6,
    )
    ```

    This will create a Zarr group containing:

    - `iq`: Beamformed IQ data with dimensions `(time, z, y, x)`.
    - `time`, `z`, `y`, `x`: Coordinate arrays.
    - Metadata attributes (e.g., `voxdim`, `transmit_frequency`, `plane_wave_angles`) as
      provided via keyword arguments.

=== "EchoFrame DAT"

    This format consists of a binary `.dat` file containing the beamformed data and a
    `.mat` file containing metadata (sequence parameters).

    To convert EchoFrame data to Zarr, use the
    [`convert_echoframe_dat_to_zarr`][confusius.io.convert_echoframe_dat_to_zarr]
    function.

    ```python
    from confusius.io import convert_echoframe_dat_to_zarr

    convert_echoframe_dat_to_zarr(
        dat_path="path/to/data.dat",
        meta_path="path/to/metadata.mat",
        output_path="sub-01_task-awake_iq.zarr"
        # Optional: specify block start times. Other metadata (e.g., transmit frequency,
        # axis coordinates) will be automatically extracted from the metadata file.
        block_times=block_times,
    )
    ```

    This will create a Zarr group containing:

    - `iq`: Beamformed IQ data with dimensions `(time, z, y, x)`.
    - `time`, `z`, `y`, `x`: Coordinate arrays.
    - Metadata attributes (e.g., `voxdim`, `transmit_frequency`, `plane_wave_angles`) as
      extracted from the metadata file.

## Loading beamformed IQ data

Once your beamformed IQ data is converted to Zarr, you can easily load it using Xarray.

```python
import xarray as xr

ds = xr.open_zarr("sub-01_task-awake_iq.zarr")
iq_data = ds["iq"]

print(iq_data)
```

This returns an `xarray.DataArray` lazily loaded from the Zarr store, ready for
processing:

```text
<xarray.DataArray 'iq' (time: 1168500, z: 1, y: 118, x: 52)> Size: 57GB
dask.array<open_dataset-iq, shape=(1168500, 1, 118, 52), dtype=complex64, chunksize=(300, 1, 118, 52), chunktype=numpy.ndarray>
Coordinates:
  * time     (time) float64 9MB 5.551 5.553 5.555 ... 2.355e+03 2.355e+03
  * z        (z) float64 8B 0.0
  * y        (y) float64 944B 4.656 4.705 4.753 4.802 ... 10.23 10.28 10.33
  * x        (x) float64 416B -2.671 -2.57 -2.469 -2.369 ... 2.268 2.369 2.469
Attributes:
    voxdim:                       [0.4, 0.04850949019607844, 0.10078740157480...
    transmit_frequency:           15625000.0
    probe_n_elements:             128
    probe_pitch:                  0.0001
    sound_velocity:               1510.0
    plane_wave_angles:            [-10.0, -9.310344696044922, -8.620689392089...
    compound_sampling_frequency:  500.0
    pulse_repetition_frequency:   15000.0
    beamforming_method:           Fourier
```

## Loading derived acquisitions (e.g., power Doppler, velocity)

ConfUSIus supports NIfTI, the standard format for fUSI-BIDS compatibility. NIfTI files
allow interoperability with neuroimaging analysis tools and preserve spatial metadata
(voxel dimensions, affine transformations).

### Loading NIfTI files

Use [`load_nifti`][confusius.io.load_nifti] to load NIfTI files as lazy Xarray DataArrays:

```python
from confusius.io import load_nifti

# Load with automatic BIDS sidecar metadata
da = load_nifti("sub-01_task-awake_pwd.nii.gz", load_sidecar=True)
print(da.dims)
# Output: ('time', 'z', 'y', 'x')
```

By default, ConfUSIus will automatically load a JSON sidecar file with same basename
(e.g., `sub-01_task-awake_pwd.json`) if present, following the BIDS specification for
fUSI metadata.

### Saving to NIfTI

You can save DataArrays to NIfTI using either the module function or the Xarray accessor:

**Using the module function:**

```python
from confusius.io import save_nifti

save_nifti(data_array, "output.nii.gz", save_sidecar=True)
```

**Using the Xarray accessor:**

```python
# Save via accessor (shorter syntax)
data.fusi.io.to_nifti("output.nii.gz")
```

Both methods create a JSON sidecar file containing additional metadata (coordinates, units, custom attributes) alongside the NIfTI file.

### Converting between formats

ConfUSIus provides utilities to convert between NIfTI and Zarr formats:

```python
from confusius.io import convert_nifti_to_zarr

# Convert NIfTI to Zarr for efficient chunked access
convert_nifti_to_zarr(
    "brain.nii.gz",
    "brain.zarr",
    chunks="auto"
)

# Load the converted Zarr with Xarray
import xarray as xr
ds = xr.open_zarr("brain.zarr")
```

[^iq-dask]:
    While ConfUSIus has been designed to use Xarray for handling beamformed IQ data,
    some functions (e.g., [`process_iq_blocks`][confusius.iq.process_iq_blocks])
    may be used using Dask arrays directly, allowing the use of any Dask-compatible
    array format (e.g., HDF5, memory-mapped NumPy arrays, or even in-memory arrays).
    However, using Xarray provides additional benefits such as automatic handling of
    coordinates, metadata, and integration with the rest of the ConfUSIus processing
    pipeline.
