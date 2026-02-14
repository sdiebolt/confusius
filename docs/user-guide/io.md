---
icon: lucide/file-input
---

# Input/Output

## Overview

ConfUSIus is designed to handle large-scale fUSI datasets efficiently. This guide
explains how to work with different data formats and convert them to work with
ConfUSIus.

## Working with Xarray

ConfUSIus uses [Xarray](https://docs.xarray.dev/) as its core data structure for
representing multi-dimensional fUSI data. Xarray provides several advantages over raw
NumPy arrays:

- **Named dimensions**: access data using meaningful names (e.g., `time`, `x`, `y`, `z`)
  instead of remembering axis indices.
- **Coordinates**: associate physical coordinates with each dimension (e.g., time
  in seconds, depth in millimeters).
- **Metadata storage**: keep acquisition parameters, units, and other metadata alongside
  your data
- **Unified API**: use the same operations regardless of the underlying storage format

### Xarray-Compatible Formats

Xarray can read and write data from multiple storage formats, including:

- **[Zarr](https://zarr.dev/)**: Chunked, compressed, cloud-native format (recommended for large datasets).
- **[HDF5](https://www.hdfgroup.org/solutions/hdf5/)**: Hierarchical format widely used in research.
- **[netCDF](https://www.unidata.ucar.edu/software/netcdf)**: Self-describing format
  common in scientific computing (e.g., basis of the
  [MINC](https://mcin.ca/technology/minc/) 1.0 format).

Additionally, ConfUSIus provides utilities to read and write
[**NIfTI**](https://nifti.nimh.nih.gov/) files (the standard neuroimaging format for
BIDS) as Xarray DataArrays, enabling seamless integration with BIDS-compliant fUSI
datasets.

### Recommended Formats for fUSI

ConfUSIus works with all Xarray-compatible formats, but two formats are particularly
well-suited for fUSI workflows:

=== "NIfTI for Sharing and Interoperability"

    [NIfTI](https://nifti.nimh.nih.gov/) is the standard format for:

    - **fUSI-BIDS compliance**: required for sharing datasets following the BIDS
      specification.
    - **Neuroimaging pipelines**: compatible with tools like
      [Nilearn](https://nilearn.github.io/),
      [ANTsPy](https://antspy.readthedocs.io/en/stable/), and
      [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/), which have been used in some fUSI
      studies.
    - **Derived acquisitions**: power Doppler, velocity, and other processed signals.

    Use NIfTI when you need to share data, ensure BIDS compliance, or integrate with
    existing neuroimaging analysis tools.

=== "Zarr for Large-Scale Processing"

    [Zarr](https://zarr.readthedocs.io/) excels at handling massive datasets through:

    - **Out-of-core processing**: Work with datasets larger than memory by loading only
      needed chunks.
    - **Compression**: Reduce storage footprint without sacrificing access speed.
    - **Parallel I/O**: Read and write data concurrently from multiple processes.
    - **Cloud compatibility**: Store and access data on remote object storage (S3, GCS,
      etc.).

    Use Zarr for beamformed IQ data, large-scale analyses, and cloud-based processing
    workflows. ConfUSIus converts beamformed IQ data to Zarr by default for optimal
    performance with large-scale datasets.

## fUSI Data Types

fUSI workflows involve two main categories of data:

=== "Beamformed IQ Data"

    Complex-valued signals resulting from ultrasound beamforming of RF signals. These
    signals are typically further processed to extract derived signals such as power Doppler
    or velocity. Beamformed IQ datasets are typically very large (10s to 100s of GB per
    acquisition session), and stored in proprietary formats depending on the acquisition
    system (e.g., Iconeus, AUTC, EchoFrame, etc.).

=== "Derived Acquisitions"

    Processed data products such as power Doppler, velocity, or other signals derived from
    beamformed IQ data. These datasets are generally much smaller than the original IQ data
    (often 10-100x smaller) and are frequently stored in standardized formats like NIfTI for 
    interoperability and BIDS compliance. 

    !!! tip 
        Large-scale derived acquisitions (e.g., long power Doppler recordings) can also
        benefit from storage in Zarr for efficient processing.

## Converting Beamformed IQ Data

Most fUSI acquisition systems output beamformed IQ data in proprietary binary formats.
ConfUSIus currently provides built-in conversion utilities for **AUTC** and
**EchoFrame** systems to transform their data into Zarr for efficient processing.

!!! question "Using beamformed IQ data from other sources?"
    You may convert beamformed IQ data to any Xarray-compatible format (e.g., netCDF,
    HDF5, or any Dask-compatible array) and use them with ConfUSIus. However, AUTC and
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
        output_path="sub-01_task-awake_iq.zarr",
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

## Converting Derived Acquisitions

Derived acquisitions (power Doppler, velocity, etc.) are often stored in NIfTI format
for compatibility with neuroimaging standards like BIDS. ConfUSIus provides utilities to
load NIfTI files into Xarray (see [Loading NIfTI Files](#loading-nifti-files)).

For large derived acquisitions or workflows requiring repeated access to subsets of
data, converting NIfTI to Zarr provides better performance:

```python
from confusius.io import convert_nifti_to_zarr

# Convert NIfTI to Zarr for efficient chunked access.
convert_nifti_to_zarr(
    input_path="sub-01_task-awake_pwd.nii.gz",
    output_path="sub-01_task-awake_pwd.zarr",
)
```

ConfUSIus automatically preserves BIDS sidecar metadata during conversion, ensuring
acquisition parameters remain accessible.

## Loading Data

### Loading Zarr Files

Once your data is in Zarr format, load it using Xarray's standard interface:

```python
import xarray as xr

# Load beamformed IQ data.
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

Notice that the data remains on disk (shown by `dask.array<...>`) until you explicitly
compute operations on it.

### Loading NIfTI Files

Use [`load_nifti`][confusius.io.load_nifti] to load NIfTI files as lazy Xarray
DataArrays:

```python
from confusius.io import load_nifti

# Load with automatic BIDS sidecar metadata.
da = load_nifti("sub-01_task-awake_pwd.nii.gz", load_sidecar=True)
print(da.dims)
# Output: ('time', 'z', 'y', 'x')
```

By default, ConfUSIus automatically loads a JSON sidecar file with the same basename
(e.g., `sub-01_task-awake_pwd.json`) if present, following the BIDS specification for
fUSI metadata.

## Saving Data

### Saving to NIfTI

You can save DataArrays to NIfTI using either the module function
[`save_nifti`][confusius.io.save_nifti] or the Xarray accessor:

=== "Module function"

    ```python
    from confusius.io import save_nifti

    save_nifti(data_array, "output.nii.gz", save_sidecar=True)
    ```

=== "Xarray accessor"

    ```python
    data_array.fusi.io.to_nifti("output.nii.gz")
    ```

Both methods create a JSON sidecar file containing additional metadata (coordinates,
units, custom attributes) alongside the NIfTI file, ensuring BIDS compatibility.

### Saving to Zarr

Use Xarray's built-in Zarr support:

```python
# Save DataArray to Zarr.
data_array.to_zarr("output.zarr")
```

## Format Conversion Reference

Quick reference for converting between formats:

| From | To | Function |
|------|-----|----------|
| AUTC DATs | Zarr | [`convert_autc_dats_to_zarr`][confusius.io.convert_autc_dats_to_zarr] |
| EchoFrame DAT | Zarr | [`convert_echoframe_dat_to_zarr`][confusius.io.convert_echoframe_dat_to_zarr] |
| NIfTI | Zarr | [`convert_nifti_to_zarr`][confusius.io.convert_nifti_to_zarr] |
| Xarray DataArray | NIfTI | [`save_nifti`][confusius.io.save_nifti] or `.fusi.io.to_nifti()` |

