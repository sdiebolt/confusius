"""Coordinate handling for fUSI-BIDS timing metadata.

This module provides utilities for converting between fUSI-BIDS timing metadata and
ConfUSIus coordinate structures.
"""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import find_stack_level


def create_slice_time_coordinate(
    slice_timing: npt.ArrayLike,
    n_time: int,
    slice_encoding_direction: Literal["i", "j", "k", "i-", "j-", "k-"],
    spatial_shape: tuple[int, int, int],
) -> xr.DataArray:
    """Create a `slice_time` coordinate from fUSI-BIDS `SliceTiming` metadata.

    This creates a 2D coordinate, with dimensions depending on the
    `SliceEncodingDirection`. The `slice_time` coordinate stores the exact acquisition
    time of each slice at each time point.

    Parameters
    ----------
    slice_timing : array-like
        1D array of slice acquisition times within a volume (seconds).
    n_time : int
        Number of time points (volumes).
    slice_encoding_direction : {"i", "j", "k", "i-", "j-", "k-"}
        Direction of slice acquisition.
    spatial_shape : tuple[int, int, int]
        Shape of spatial dimensions `(z, y, x)`.

    Returns
    -------
    (n_time, n_slices) xarray.DataArray
        Slice time coordinates where `n_slices` is determined by the slice encoding
        direction.

    Raises
    ------
    ValueError
        If `slice_encoding_direction` is invalid or `slice_timing` length doesn't match
        the expected number of slices.

    Examples
    --------
    >>> import numpy as np
    >>> slice_timing = np.array([0.0, 0.1, 0.2, 0.3])  # 4 slices
    >>> coord = create_slice_time_coordinate(
    ...     slice_timing, n_time=10, slice_encoding_direction="k",
    ...     spatial_shape=(4, 64, 64)
    ... )
    >>> coord.dims
    ('time', 'z')

    >>> coord.shape
    (10, 4)

    Notes
    -----
    The slice encoding direction mapping:
    - "i" or "i-" → x dimension.
    - "j" or "j-" → y dimension.
    - "k" or "k-" → z dimension.

    The sign ("-" suffix) indicates reversed acquisition order but does not affect the
    coordinate dimensions.
    """
    direction_map = {
        "i": ("x", spatial_shape[2]),
        "i-": ("x", spatial_shape[2]),
        "j": ("y", spatial_shape[1]),
        "j-": ("y", spatial_shape[1]),
        "k": ("z", spatial_shape[0]),
        "k-": ("z", spatial_shape[0]),
    }

    if slice_encoding_direction not in direction_map:
        raise ValueError(
            f"Invalid SliceEncodingDirection: {slice_encoding_direction}. "
            f"Must be one of: {list(direction_map.keys())}"
        )

    dim_name, expected_n_slices = direction_map[slice_encoding_direction]
    slice_timing_arr = np.asarray(slice_timing)

    if slice_timing_arr.ndim != 1:
        raise ValueError(f"SliceTiming must be 1D, got shape {slice_timing_arr.shape}")

    if len(slice_timing_arr) != expected_n_slices:
        raise ValueError(
            f"SliceTiming length ({len(slice_timing_arr)}) does not match "
            f"expected number of slices ({expected_n_slices}) for "
            f"SliceEncodingDirection '{slice_encoding_direction}'"
        )

    slice_time_values = np.tile(slice_timing_arr, (n_time, 1))

    return xr.DataArray(
        slice_time_values,
        dims=["time", dim_name],
        attrs={"units": "s"},
    )


def extract_slice_timing_from_coordinate(
    slice_time_coord: xr.DataArray,
) -> tuple[np.ndarray, str]:
    """Extract fUSI-BIDS `SliceTiming` metadata from a `slice_time` coordinate.

    Parameters
    ----------
    slice_time_coord : xarray.DataArray
        Slice time coordinate with dimensions `(time, x|y|z)`.

    Returns
    -------
    slice_timing : numpy.ndarray
        The unique slice acquisition times for one volume.
    slice_encoding_direction : str
        The direction of slice acquisition ("i", "j", or "k").
        `slice_timing_array` is the unique slice acquisition times for one volume.

    Raises
    ------
    ValueError
        If the coordinate dimensions are invalid.

    Examples
    --------
    >>> slice_timing, direction = extract_slice_timing_from_coordinate(coord)
    >>> slice_timing
    array([0. , 0.1, 0.2, 0.3])
    >>> direction
    'k'
    """
    dim_to_direction = {
        "x": "i",
        "y": "j",
        "z": "k",
    }

    if len(slice_time_coord.dims) != 2:
        raise ValueError(
            f"slice_time coordinate must have 2 dimensions, "
            f"got {len(slice_time_coord.dims)}: {slice_time_coord.dims}"
        )

    spatial_dims = [d for d in slice_time_coord.dims if d != "time"]
    if len(spatial_dims) != 1:
        raise ValueError(
            f"slice_time coordinate must have exactly one spatial dimension, "
            f"got: {spatial_dims}"
        )

    spatial_dim = str(spatial_dims[0])
    if spatial_dim not in dim_to_direction:
        raise ValueError(
            f"Unknown spatial dimension: {spatial_dim}. "
            f"Expected one of: {list(dim_to_direction.keys())}"
        )

    slice_encoding_direction = dim_to_direction[spatial_dim]

    slice_timing = slice_time_coord.isel(time=0).values

    for t in range(1, slice_time_coord.sizes["time"]):
        if not np.allclose(slice_time_coord.isel(time=t).values, slice_timing):
            warnings.warn(
                "Slice timing varies across time points. "
                "Using timing from first volume only.",
                stacklevel=find_stack_level(),
            )
            break

    return slice_timing, slice_encoding_direction
