"""Coordinate handling for fUSI-BIDS timing metadata.

This module provides utilities for converting between fUSI-BIDS timing metadata and
ConfUSIus coordinate structures.
"""

from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

_SLICE_ENCODING_DIRECTION_TO_DIM: Final[dict[str, str]] = {
    "i": "x",
    "i-": "x",
    "j": "y",
    "j-": "y",
    "k": "z",
    "k-": "z",
}
"""Mapping from fUSI-BIDS `SliceEncodingDirection` values to ConfUSIus dimension names."""

_DIM_TO_SLICE_ENCODING_DIRECTION: Final[dict[str, str]] = {
    "x": "i",
    "y": "j",
    "z": "k",
}
"""Mapping from ConfUSIus dimension names to fUSI-BIDS `SliceEncodingDirection` values."""


def create_slice_time_coordinate(
    slice_timing: npt.ArrayLike,
    slice_encoding_direction: Literal["i", "j", "k", "i-", "j-", "k-"],
    units: str = "s",
) -> xr.DataArray:
    """Create a `slice_time` coordinate from fUSI-BIDS `SliceTiming` metadata.

    The `slice_time` coordinate stores the acquisition time of each slice within each
    volume.

    Parameters
    ----------
    slice_timing : (n_slices,) array-like
        Array of slice acquisition times within a volume.
    slice_encoding_direction : {"i", "j", "k", "i-", "j-", "k-"}
        Direction of slice acquisition: `"i"` → `x`, `"j"` → `y`, `"k"` → `z`. A
        trailing `-` indicates that `slice_timing` is defined in reverse order (the
        first entry corresponds to the slice with the largest index). The values are
        reversed internally so the stored coordinate is always aligned with natural
        slice index order.
    units : str
        Units of the slice timing values.

    Returns
    -------
    (n_slices,) xarray.DataArray
        Slice time coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> slice_timing = np.array([0.0, 0.1, 0.2, 0.3])  # 4 slices
    >>> coord = create_slice_time_coordinate(slice_timing, slice_encoding_direction="k")
    >>> coord.dims
    ('z',)

    >>> coord.shape
    (4,)
    """
    if slice_encoding_direction not in _SLICE_ENCODING_DIRECTION_TO_DIM:
        raise ValueError(
            f"Invalid SliceEncodingDirection: {slice_encoding_direction}. "
            f"Must be one of: {list(_SLICE_ENCODING_DIRECTION_TO_DIM.keys())}"
        )

    dim_name = _SLICE_ENCODING_DIRECTION_TO_DIM[slice_encoding_direction]
    slice_timing_arr = np.asarray(slice_timing)

    if slice_timing_arr.ndim != 1:
        raise ValueError(f"SliceTiming must be 1D, got shape {slice_timing_arr.shape}")

    if slice_encoding_direction.endswith("-"):
        slice_timing_arr = slice_timing_arr[::-1]

    return xr.DataArray(slice_timing_arr, dims=[dim_name], attrs={"units": units})


def extract_slice_timing_from_coordinate(
    slice_time_coord: xr.DataArray,
) -> tuple[np.ndarray, str]:
    """Extract fUSI-BIDS `SliceTiming` metadata from a `slice_time` coordinate.

    Parameters
    ----------
    slice_time_coord : (slice_encoding_direction,) xarray.DataArray
        Slice time coordinate with dimension corresponding to the slice encoding
        direction.

    Returns
    -------
    slice_timing : numpy.ndarray
        The slice acquisition times for one volume.
    slice_encoding_direction : {"i", "j", "k"}
        The direction of slice acquisition.

    Raises
    ------
    ValueError
        If the coordinate is not 1D or has an invalid spatial dimension.

    Examples
    --------
    >>> slice_timing, direction = extract_slice_timing_from_coordinate(coord)
    >>> slice_timing
    array([0. , 0.1, 0.2, 0.3])
    >>> direction
    'k'
    """
    if len(slice_time_coord.dims) != 1:
        raise ValueError(
            f"slice_time coordinate must be 1D, got {len(slice_time_coord.dims)}D: "
            f"{slice_time_coord.dims}"
        )

    spatial_dim = slice_time_coord.dims[0]
    if spatial_dim not in _DIM_TO_SLICE_ENCODING_DIRECTION:
        raise ValueError(
            f"slice_time coordinate must have one of spatial dimensions "
            f"{list(_DIM_TO_SLICE_ENCODING_DIRECTION.keys())}, got: {slice_time_coord.dims}"
        )

    return slice_time_coord.values, _DIM_TO_SLICE_ENCODING_DIRECTION[str(spatial_dim)]
