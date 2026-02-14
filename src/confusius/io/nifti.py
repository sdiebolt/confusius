"""Utilities for loading and saving NIfTI files.

This module provides functions to load NIfTI neuroimaging files as lazy Xarray
DataArrays using nibabel's proxy arrays and Dask for out-of-core processing. Following
ConfUSIus conventions, data is stored with dimensions ``(time, z, y, x)``.
"""

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr
import zarr

from confusius.io.utils import check_path

if TYPE_CHECKING:
    import nibabel as nib

NiftiVersion: TypeAlias = Literal[1, 2]
"""Type alias for NIfTI file format version."""

# ConfUSIus and NIfTI dimension mapping.
_NIFTI_DIM_ORDER = ("x", "y", "z", "time", "dim4", "dim5", "dim6")


class _NiftiHeaderExtractor:
    """Extract relevant metadata from NIfTI header."""

    def __init__(
        self,
        header: "nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header",
    ) -> None:
        self.header = header

    def get_voxel_dimensions(self) -> tuple[float, ...]:
        """Get voxel dimensions (pixdim) in millimeters.

        Returns
        -------
        tuple[float, ...]
            Voxel dimensions for each spatial dimension.
        """
        zooms = self.header.get_zooms()
        space_units, _ = self.header.get_xyzt_units()

        unit_scaler = {"meter": 1000.0, "mm": 1.0, "micron": 0.001}.get(
            space_units, 1.0
        )
        return tuple(z * unit_scaler for z in zooms[:3])

    def get_repetition_time(self) -> float | None:
        """Get repetition time (TR) in seconds from header.

        Returns
        -------
        float or None
            Repetition time in seconds, or ``None`` if not available.
        """
        zooms = self.header.get_zooms()
        if len(zooms) < 4:
            return None

        time_zoom = zooms[3]
        _, time_units = self.header.get_xyzt_units()

        time_scaler = {"sec": 1.0, "msec": 0.001, "usec": 1e-6}.get(time_units, 1.0)
        return time_zoom * time_scaler

    def to_attrs(self) -> dict[str, Any]:
        """Convert header information to attributes dictionary."""
        header_module = type(self.header).__module__
        nifti_version = 2 if "nifti2" in header_module else 1

        # Get both qform and sform matrices (nibabel returns None if not set)
        sform_matrix, sform_code = self.header.get_sform(coded=True)
        qform_matrix, qform_code = self.header.get_qform(coded=True)

        attrs: dict[str, Any] = {
            "nifti_version": nifti_version,
            "voxdim": list(self.get_voxel_dimensions()),
        }

        # Code > 0 indicates a valid sform/qform; code 0 means unknown.
        if sform_code > 0 and sform_matrix is not None:
            attrs["sform"] = sform_matrix.tolist()
            attrs["sform_code"] = int(sform_code)
        if qform_code > 0 and qform_matrix is not None:
            attrs["qform"] = qform_matrix.tolist()
            attrs["qform_code"] = int(qform_code)

        tr = self.get_repetition_time()
        if tr is not None:
            attrs["repetition_time"] = tr

        return attrs


def _load_nifti_with_nibabel(
    path: Path,
) -> tuple[Union["nib.nifti1.Nifti1Image", "nib.nifti2.Nifti2Image"], dict[str, Any]]:
    """Load NIfTI file using NiBabel, extracting header metadata.

    Parameters
    ----------
    path : pathlib.Path
        Path to the NIfTI file (``.nii`` or ``.nii.gz``).

    Returns
    -------
    tuple[nib.nifti1.Nifti1Image or nib.nifti2.Nifti2Image, dict]
        - NiBabel NIfTI image object (with proxy data array).
        - Dictionary of attributes extracted from header.
    """
    import nibabel as nib

    img = nib.load(path)
    if not isinstance(img, nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image):
        raise ValueError(
            "Only NIfTI-1 and NIfTI-2 formats are supported when loading files with"
            " .nii or .nii.gz suffixes."
        )

    extractor = _NiftiHeaderExtractor(img.header)
    attrs = extractor.to_attrs()

    return img, attrs


def _create_coords_from_nifti(
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    affine: npt.NDArray[np.float64],
    tr: float | None = None,
) -> dict[str, xr.DataArray]:
    """Create coordinate arrays from NIfTI affine matrix and repetition time."""
    coords: dict[str, xr.DataArray] = {}

    spatial_dims = [d for d in dims if d in ("x", "y", "z")]
    spatial_indices = {d: dims.index(d) for d in spatial_dims}
    nifti_to_confusius_idx = {"x": 0, "y": 1, "z": 2}

    for dim in spatial_dims:
        idx_in_shape = spatial_indices[dim]
        dim_size = shape[idx_in_shape]
        nifti_col = nifti_to_confusius_idx[dim]
        voxel_vector = affine[:3, nifti_col]
        origin = affine[:3, 3]
        indices = np.arange(dim_size)
        coord_values = np.array([origin + voxel_vector * i for i in indices])
        coord_idx = nifti_col
        coords[dim] = xr.DataArray(
            coord_values[:, coord_idx],
            dims=[dim],
            attrs={"units": "mm"},
        )

    if "time" in dims:
        time_idx = dims.index("time")
        n_time = shape[time_idx]
        time_values = np.arange(n_time) * tr if tr is not None else np.arange(n_time)
        coords["time"] = xr.DataArray(
            time_values,
            dims=["time"],
            attrs={"units": "s" if tr is not None else "frame"},
        )

    return coords


def load_nifti(
    path: str | Path,
    chunks: int | tuple[int, ...] | str | None = "auto",
    load_sidecar: bool = True,
) -> xr.DataArray:
    """Load a NIfTI file as a lazy Xarray DataArray.

    Loads NIfTI files using nibabel's proxy arrays for memory-efficient access,
    wrapping the data in Dask arrays for chunked, parallel processing. The data
    is transposed to ConfUSIus conventions with dimensions ``(time, z, y, x)``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NIfTI file (.nii or .nii.gz). If ``load_sidecar`` is
        ``True``, a corresponding JSON sidecar file will be loaded if present.
    chunks : int or tuple[int, ...] or str or None, default: "auto"
        How to chunk the array. Must be one of the following forms:

        - A blocksize like ``1000``.
        - A blockshape like ``(1000, 1000)``.
        - Explicit sizes of all blocks along all dimensions like ``((1000, 1000,
          500), (400, 400))``.
        - A size in bytes, like ``"100 MiB"`` which will choose a uniform block-like
          shape.
        - The word ``"auto"`` to let Dask choose chunk sizes based on heuristics. See
          `dask.array.normalize_chunks` for more details on how chunk sizes are
          determined.
        - ``-1`` or ``None`` as a blocksize indicate the size of the corresponding
          dimension.

    load_sidecar : bool, default: True
        Whether to load a BIDS-style JSON sidecar file (same name with ``.json``
        extension) for additional metadata.

    Returns
    -------
    xarray.DataArray
        Lazy DataArray with dimensions in ConfUSIus order. Data is wrapped in
        a Dask array for out-of-core computation.

    Examples
    --------
    >>> import confusius as fusi
    >>> da = fusi.io.load_nifti("brain.nii.gz")
    >>> print(da.dims)
    ("time", "z", "y", "x")

    Notes
    -----
    - The data is not loaded into memory until computation is triggered.
    - Coordinate arrays are extracted from the NIfTI affine matrix when possible.
    - Complex data types are not natively supported by NIfTI and will raise an
      error.
    """
    path = check_path(path, type="file")

    img, attrs = _load_nifti_with_nibabel(path)

    if load_sidecar:
        if path.suffix == ".gz" and path.stem.endswith(".nii"):
            sidecar_path = path.with_suffix("").with_suffix(".json")
        else:
            sidecar_path = path.with_suffix(".json")

        if sidecar_path.exists():
            with open(sidecar_path) as f:
                sidecar_attrs = json.load(f)
                attrs.update(sidecar_attrs)

    # We use np.asanyarray to get the memory-mapped array behind Nibabel's proxy. This
    # will allow Dask to create a lazy array without loading the entire dataset into
    # memory.
    data_obj = np.asanyarray(img.dataobj)

    # NIfTI stores data with shape (x, y, z, time) in column-major order.
    # ConfUSIus uses (time, z, y, x) in row-major order. These two conventions are
    # equivalent and one can be obtained from the other by transposing the array. This
    # will not copy data.
    data_obj = data_obj.T

    dask_arr = da.from_array(data_obj, chunks=chunks, asarray=False)

    ndim = dask_arr.ndim
    confusius_dims = _NIFTI_DIM_ORDER[:ndim][::-1]
    confusius_shape = dask_arr.shape

    # Reverse voxdim to match ConfUSIus order (z, y, x) instead of NIfTI (x, y, z)
    attrs["voxdim"] = attrs["voxdim"][::-1][:ndim]

    # Extract coordinates from affine matrix using ConfUSIus shape/dims
    # nibabel's img.affine returns the best available affine (sform if valid, else
    # qform).
    affine = img.affine if img.affine is not None else np.eye(4)
    tr = attrs.get("repetition_time")
    coords = _create_coords_from_nifti(confusius_shape, confusius_dims, affine, tr)

    data_array = xr.DataArray(
        dask_arr,
        dims=confusius_dims,
        coords=coords,
        attrs=attrs,
        name="nifti_data",
    )

    return data_array


def save_nifti(
    data_array: xr.DataArray,
    path: str | Path,
    nifti_version: NiftiVersion = 1,
    save_sidecar: bool = True,
) -> None:
    """Save an Xarray DataArray to NIfTI format.

    Saves the DataArray to a NIfTI file, with optional JSON sidecar for additional
    metadata. The data is transposed to NIfTI convention ``(x, y, z, time)`` before
    saving.

    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray to save.
    path : str or pathlib.Path
        Output path for the NIfTI file, with ``.nii`` or ``.nii.gz`` extension. If
        ``.nii.gz`` is used, the file will be saved in compressed format.
    nifti_version : {1, 2}, default: 1
        NIfTI format version to use. Version 2 is a simple extension to support
        larger files and arrays with dimension sizes greater than 32,767.
    save_sidecar : bool, default: True
        Whether to save additional metadata as a BIDS-style JSON sidecar file.

    Examples
    --------
    >>> import confusius as fusi
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.random.rand(10, 32, 1, 64),
    ...                   dims=["time", "z", "y", "x"])
    >>> fusi.io.save_nifti(da, "output.nii.gz")
    PosixPath("output.nii.gz")
    """
    import nibabel as nib

    path = Path(path)
    if path.suffix != ".nii" and path.suffixes[-2:] != [".nii", ".gz"]:
        raise ValueError("Output file must have .nii or .nii.gz extension.")

    data = np.asarray(data_array)

    current_dims = data_array.dims
    target_order = []
    for dim in ("x", "y", "z", "time"):
        if dim in current_dims:
            target_order.append(current_dims.index(dim))

    for i, dim in enumerate(current_dims):
        if dim not in ("x", "y", "z", "time"):
            target_order.append(i)

    # ConfUSIus order is (time, z, y, x), NIfTI order is (x, y, z, time).
    if len(target_order) == len(current_dims):
        data = np.transpose(data, target_order)

    voxel_dims = data_array.attrs.get("voxdim", [1.0, 1.0, 1.0])
    if len(voxel_dims) < 3:
        voxel_dims = list(voxel_dims) + [1.0] * (3 - len(voxel_dims))

    tr = data_array.attrs.get("repetition_time")
    if tr is None and "time" in data_array.coords:
        time_coord = data_array.coords["time"]
        if len(time_coord) > 1:
            tr = float(time_coord[1] - time_coord[0])

    zooms = list(voxel_dims[:3])
    if "time" in current_dims:
        zooms.append(tr if tr is not None else 1.0)

    affine = np.eye(4)
    for i, dim in enumerate(("x", "y", "z")):
        if dim in data_array.coords:
            coord = data_array.coords[dim]
            if len(coord) > 1:
                spacing = float(coord[1] - coord[0])
                affine[i, i] = spacing
                affine[i, 3] = float(coord[0])

    if "affine" in data_array.attrs:
        stored_affine = np.array(data_array.attrs["affine"])
        if stored_affine.shape == (4, 4):
            affine = stored_affine

    if nifti_version == 1:
        img_class = nib.Nifti1Image
    else:
        img_class = nib.Nifti2Image

    nifti_img = img_class(data, affine)

    nifti_img.header.set_zooms(zooms)
    nifti_img.header.set_sform(affine)
    nifti_img.header.set_qform(affine)

    nifti_img.to_filename(path)

    if save_sidecar:
        sidecar_attrs = {
            k: v
            for k, v in data_array.attrs.items()
            if k not in ("affine",)  # Exclude fields stored in NIfTI header
        }

        for dim in ("x", "y", "z", "time"):
            if dim in data_array.coords:
                coord = data_array.coords[dim]
                sidecar_attrs[f"{dim}_coordinates"] = coord.values.tolist()
                if "units" in coord.attrs:
                    sidecar_attrs[f"{dim}_units"] = coord.attrs["units"]

        if path.suffix == ".gz":
            sidecar_path = path.with_suffix("").with_suffix(".json")
        else:
            sidecar_path = path.with_suffix(".json")

        with open(sidecar_path, "w") as f:
            json.dump(sidecar_attrs, f, indent=2, default=str)


def convert_nifti_to_zarr(
    input_path: str | Path,
    output_path: str | Path,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    load_sidecar: bool = True,
    overwrite: bool = False,
) -> zarr.Group:
    """Convert a NIfTI file to Zarr format compatible with Xarray.

    Converts NIfTI files to a Zarr group following ConfUSIus conventions,
    with dimensions ``(time, z, y, x)`` and coordinates stored as separate
    arrays.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the input NIfTI file (.nii or .nii.gz).
    output_path : str or pathlib.Path
        Path where the Zarr group will be saved.
    chunks : tuple[int, ...] or "auto", default: "auto"
        Chunk sizes for the Zarr array. If not provided, default are guessed based on
        the shape and dtype.
    load_sidecar : bool, default: True
        Whether to load a BIDS-style JSON sidecar file for additional metadata.
    overwrite : bool, default: False
        Whether to overwrite an existing Zarr store at ``output_path``.

    Returns
    -------
    zarr.Group
        The created Zarr group containing the data array and coordinate arrays.
        Can be opened directly with ``xarray.open_zarr()``.

    Examples
    --------
    >>> import confusius as fusi
    >>> zarr_group = fusi.io.convert_nifti_to_zarr(
    ...     "brain.nii.gz", "brain.zarr"
    ... )
    >>> import xarray as xr
    >>> ds = xr.open_zarr("brain.zarr")
    >>> print(ds)
    <xarray.Dataset>
    Dimensions:  (time: 100, z: 1, y: 64, x: 64)
    ...
    """
    input_path = check_path(input_path, type="file")
    output_path = Path(output_path)

    data_array = load_nifti(input_path, load_sidecar=load_sidecar)

    shape = data_array.shape
    dims = data_array.dims

    mode = "w" if overwrite else "w-"
    zarr_group = zarr.open_group(output_path, mode=mode)

    zarr_array = zarr_group.create_array(
        "data",
        shape=shape,
        chunks=chunks,
        dtype=data_array.dtype,
        dimension_names=[str(d) for d in dims],
    )

    # This will trigger Dask computation if lazy.
    zarr_array[:] = np.asarray(data_array)

    for dim in dims:
        if dim in data_array.coords:
            coord = data_array.coords[dim]
            coord_array = zarr_group.create_array(
                str(dim),
                shape=coord.shape,
                chunks=coord.shape,
                dtype=coord.dtype,
                dimension_names=[str(dim)],
            )
            coord_array[:] = coord.values

            for attr_key, attr_val in coord.attrs.items():
                coord_array.attrs[attr_key] = attr_val

    for attr_key, attr_val in data_array.attrs.items():
        if isinstance(attr_val, np.ndarray):
            zarr_group.attrs[attr_key] = attr_val.tolist()
        elif isinstance(attr_val, list):
            zarr_group.attrs[attr_key] = [
                item.item() if isinstance(item, np.generic) else item
                for item in attr_val
            ]
        elif isinstance(attr_val, np.generic):
            zarr_group.attrs[attr_key] = attr_val.item()
        else:
            zarr_group.attrs[attr_key] = attr_val

    # Consolidate metadata for faster opening with Xarray.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        zarr.consolidate_metadata(output_path)

    return zarr_group
