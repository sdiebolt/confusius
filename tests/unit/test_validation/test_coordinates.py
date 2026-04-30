"""Tests for coordinate comparison and validation helpers."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_mask, validate_matching_coordinates


def test_validate_matching_coordinates_accepts_numeric_drift():
    """Numeric coordinates match within tolerance."""
    left = xr.DataArray(np.arange(5), dims=["time"], coords={"time": np.arange(5) * 0.1})
    right = xr.DataArray(
        np.arange(5),
        dims=["time"],
        coords={"time": np.arange(5) * 0.1 + 1e-10},
    )

    validate_matching_coordinates(left, right, "time")


def test_validate_matching_coordinates_checks_shared_dimension_coords_by_default():
    """Default behavior checks shared dimension coordinates."""
    left = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=["time", "region"],
        coords={"time": np.arange(3) * 0.1, "region": ["a", "b"]},
    )
    right = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=["time", "region"],
        coords={"time": np.arange(3) * 0.1 + 1e-10, "region": ["a", "b"]},
    )

    validate_matching_coordinates(left, right)


def test_validate_matching_coordinates_accepts_iterable_coord_names():
    """Explicit iterables of coordinate names are accepted."""
    left = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=["time", "region"],
        coords={"time": np.arange(3), "region": ["a", "b"]},
    )
    right = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=["time", "region"],
        coords={"time": np.arange(3), "region": ["a", "b"]},
    )

    validate_matching_coordinates(left, right, ["time", "region"])


def test_validate_matching_coordinates_accepts_matching_object_coordinates():
    """Object-valued coordinates use exact equality."""
    region_values = np.empty(2, dtype=object)
    region_values[:] = [("a", 1), ("b", 2)]
    left = xr.DataArray(np.arange(2), dims=["region"], coords={"region": region_values})
    right = xr.DataArray(
        np.arange(2), dims=["region"], coords={"region": region_values.copy()}
    )

    validate_matching_coordinates(left, right, "region")


def test_validate_matching_coordinates_ignores_unrelated_attached_coords():
    """Attached scalar coordinates do not affect coordinate matching."""
    left = xr.DataArray(
        np.arange(5),
        dims=["time"],
        coords={"time": np.arange(5) * 0.1, "mask": "roi_a"},
    )
    right = xr.DataArray(
        np.arange(5), dims=["time"], coords={"time": np.arange(5) * 0.1}
    )

    validate_matching_coordinates(left, right, "time")


def test_validate_matching_coordinates_raises_on_mismatch():
    """Mismatched coordinate values raise a clear error."""
    left = xr.DataArray(np.arange(5), dims=["time"], coords={"time": np.arange(5)})
    right = xr.DataArray(np.arange(5), dims=["time"], coords={"time": np.arange(5) + 1})

    with pytest.raises(ValueError, match="Coordinate 'time' does not match"):
        validate_matching_coordinates(left, right, "time")


def test_validate_matching_coordinates_uses_exact_message_for_non_numeric_coords():
    """Non-numeric coordinate mismatches mention exact equality."""
    left = xr.DataArray(np.arange(2), dims=["region"], coords={"region": ["a", "b"]})
    right = xr.DataArray(
        np.arange(2), dims=["region"], coords={"region": ["a", "c"]}
    )

    with pytest.raises(ValueError, match="with exact equality"):
        validate_matching_coordinates(left, right, "region")


def test_validate_matching_coordinates_raises_for_missing_left_coordinate():
    """Missing coordinates on the left array raise a clear error."""
    left = xr.DataArray(np.arange(3), dims=["time"])
    right = xr.DataArray(np.arange(3), dims=["time"], coords={"time": np.arange(3)})

    with pytest.raises(ValueError, match="Left array is missing coordinate 'time'"):
        validate_matching_coordinates(left, right, "time")


def test_validate_matching_coordinates_raises_for_missing_right_coordinate():
    """Missing coordinates on the right array raise a clear error."""
    left = xr.DataArray(np.arange(3), dims=["time"], coords={"time": np.arange(3)})
    right = xr.DataArray(np.arange(3), dims=["time"])

    with pytest.raises(ValueError, match="Right array is missing coordinate 'time'"):
        validate_matching_coordinates(left, right, "time")


def test_validate_matching_coordinates_raises_on_shape_mismatch():
    """Different coordinate shapes raise a mismatch error."""
    left = xr.DataArray(np.arange(3), dims=["time"], coords={"time": np.arange(3)})
    right = xr.DataArray(np.arange(2), dims=["time"], coords={"time": np.arange(2)})

    with pytest.raises(ValueError, match="Coordinate 'time' does not match"):
        validate_matching_coordinates(left, right, "time")


def test_validate_mask_accepts_scalar_attached_coordinate(sample_4d_volume):
    """Single selected masks validate even if they keep a scalar `mask` coord."""
    mask = xr.DataArray(
        np.zeros((2, *sample_4d_volume.shape[1:]), dtype=int),
        dims=["mask", "z", "y", "x"],
        coords={
            "mask": ["roi_a", "roi_b"],
            "z": sample_4d_volume.coords["z"],
            "y": sample_4d_volume.coords["y"],
            "x": sample_4d_volume.coords["x"],
        },
    )
    mask[0, 0, :, :] = 1

    validate_mask(mask.isel(mask=0), sample_4d_volume)
