"""Unit tests for confusius.io.nifti module."""

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
import xarray as xr

from confusius.io.nifti import (
    convert_nifti_to_zarr,
    load_nifti,
    save_nifti,
)


@pytest.fixture
def nifti_2d_path(tmp_path: Path) -> Path:
    """Create a 2D NIfTI file for testing."""
    data = np.random.rand(8, 6).astype(np.float32)
    nifti_path = tmp_path / "test_2d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path


@pytest.fixture
def nifti_3d_path(tmp_path: Path) -> Path:
    """Create a 3D NIfTI file for testing."""
    data = np.random.rand(10, 8, 6).astype(np.float32)
    nifti_path = tmp_path / "test_3d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path


@pytest.fixture
def nifti_4d_path(tmp_path: Path) -> Path:
    """Create a 4D NIfTI file for testing."""
    data = np.random.rand(12, 10, 8, 6).astype(np.float64)
    nifti_path = tmp_path / "test_4d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path


@pytest.fixture
def nifti_with_sidecar(tmp_path: Path) -> Path:
    """Create a 3D NIfTI file with JSON sidecar."""
    data = np.random.rand(8, 6, 4).astype(np.float32)
    nifti_path = tmp_path / "test_sidecar.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

    sidecar = {"custom_meta": "test_value", "acquisition": "test_acq"}
    sidecar_path = tmp_path / "test_sidecar.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)

    return nifti_path


class TestLoadNifti:
    """Tests for load_nifti function."""

    def test_load_2d_nifti(self, nifti_2d_path: Path) -> None:
        """Loading 2D NIfTI creates DataArray with spatial dims only."""
        da = load_nifti(nifti_2d_path, load_sidecar=False)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("y", "x")
        assert da.shape == (6, 8)
        assert da.dtype == np.float32

    def test_load_3d_nifti(self, nifti_3d_path: Path) -> None:
        """Loading 3D NIfTI creates DataArray with spatial dims only."""
        da = load_nifti(nifti_3d_path, load_sidecar=False)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("z", "y", "x")
        assert da.shape == (6, 8, 10)
        assert da.dtype == np.float32

    def test_load_4d_nifti(self, nifti_4d_path: Path) -> None:
        """Loading 4D NIfTI creates DataArray with time dimension."""
        da = load_nifti(nifti_4d_path, load_sidecar=False)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("time", "z", "y", "x")
        assert da.shape == (6, 8, 10, 12)
        assert da.dtype == np.float64

    def test_load_nifti_lazy(self, tmp_path: Path) -> None:
        """Loading creates lazy Dask array."""
        import dask.array as dask_array

        data = np.random.rand(8, 6, 4).astype(np.float32)
        nifti_path = tmp_path / "test_lazy.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        result = load_nifti(nifti_path, load_sidecar=False)

        assert isinstance(result.data, dask_array.Array)

    def test_load_nifti_with_sidecar(self, nifti_with_sidecar: Path) -> None:
        """Loading with sidecar JSON merges metadata."""
        da = load_nifti(nifti_with_sidecar, load_sidecar=True)

        assert da.attrs.get("custom_meta") == "test_value"
        assert da.attrs.get("acquisition") == "test_acq"


class TestSaveNifti:
    """Tests for `save_nifti` function."""

    def test_save_3d_dataarray(self, tmp_path):
        """Saving 3D DataArray creates valid NIfTI file."""
        data = np.random.rand(6, 8, 10).astype(np.float32)
        da = xr.DataArray(data, dims=["z", "y", "x"])

        output_path = tmp_path / "output_3d.nii"
        save_nifti(da, output_path)

        assert output_path.exists()
        loaded = nib.nifti1.Nifti1Image.from_filename(output_path)
        assert loaded.shape == (10, 8, 6)  # NIfTI order: (x, y, z)
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.dataobj), data.transpose(2, 1, 0)
        )

    def test_save_4d_dataarray(self, tmp_path):
        """Saving 4D DataArray creates valid NIfTI file."""
        data = np.random.rand(4, 6, 8, 10).astype(np.float32)
        da = xr.DataArray(data, dims=["time", "z", "y", "x"])

        output_path = tmp_path / "output_4d.nii.gz"
        save_nifti(da, output_path)

        assert output_path.exists()
        assert output_path.suffixes == [".nii", ".gz"]
        loaded = nib.nifti1.Nifti1Image.from_filename(output_path)
        assert loaded.shape == (10, 8, 6, 4)  # NIfTI order: (x, y, z, time)

    def test_save_with_coordinates(self, tmp_path):
        """Saving preserves coordinate information."""
        data = np.random.rand(3, 4, 5).astype(np.float32)
        coords = {
            "z": np.linspace(0, 10, 3),
            "y": np.linspace(0, 8, 4),
            "x": np.linspace(0, 12, 5),
        }
        da = xr.DataArray(data, dims=["z", "y", "x"], coords=coords)

        output_path = tmp_path / "with_coords.nii"
        save_nifti(da, output_path, save_sidecar=True)

        sidecar_path = tmp_path / "with_coords.json"
        assert sidecar_path.exists()

    def test_save_creates_sidecar(self, tmp_path):
        """Saving with save_sidecar=True creates JSON sidecar."""
        data = np.random.rand(4, 3, 2).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.linspace(0, 10, 4),
                "y": np.linspace(0, 5, 3),
                "x": np.linspace(0, 3, 2),
            },
            attrs={"voxdim": [1.0, 1.0, 1.0], "custom": "value"},
        )

        output_path = tmp_path / "with_sidecar.nii.gz"
        save_nifti(da, output_path, save_sidecar=True)

        sidecar_path = tmp_path / "with_sidecar.json"
        assert sidecar_path.exists()

        import json

        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar.get("custom") == "value"
        assert "z_coordinates" in sidecar


class TestConvertNiftiToZarr:
    """Tests for convert_nifti_to_zarr function."""

    def test_convert_3d_nifti(self, nifti_3d_path: Path, tmp_path: Path) -> None:
        """Converting 3D NIfTI creates valid Zarr group."""
        zarr_path = tmp_path / "test_3d.zarr"
        zarr_group = convert_nifti_to_zarr(nifti_3d_path, zarr_path, load_sidecar=False)

        assert zarr_path.exists()
        assert "data" in zarr_group
        assert zarr_group["data"].shape == (6, 8, 10)  # ConfUSIus order: (z, y, x)

    def test_convert_4d_nifti(self, nifti_4d_path: Path, tmp_path: Path) -> None:
        """Converting 4D NIfTI creates valid Zarr group."""
        zarr_path = tmp_path / "test_4d.zarr"
        zarr_group = convert_nifti_to_zarr(nifti_4d_path, zarr_path, load_sidecar=False)

        assert zarr_path.exists()
        assert "data" in zarr_group
        assert zarr_group["data"].shape == (6, 8, 10, 12)  # (time, z, y, x)

    def test_convert_stores_coordinates(self, tmp_path: Path) -> None:
        """Conversion stores coordinate arrays in Zarr."""
        data = np.random.rand(3, 2, 1).astype(np.float32)
        nifti_path = tmp_path / "test_coords.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        zarr_path = tmp_path / "test_coords.zarr"
        zarr_group = convert_nifti_to_zarr(nifti_path, zarr_path, load_sidecar=False)

        assert "x" in zarr_group
        assert "y" in zarr_group
        assert "z" in zarr_group

    def test_convert_xarray_readable(self, tmp_path: Path) -> None:
        """Converted Zarr can be opened with xarray."""
        data = np.random.rand(4, 3, 2).astype(np.float32)
        nifti_path = tmp_path / "test_xr.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        zarr_path = tmp_path / "test_xr.zarr"
        convert_nifti_to_zarr(nifti_path, zarr_path, load_sidecar=False)

        ds = xr.open_zarr(zarr_path)
        assert "data" in ds
        assert ds.data.dims == ("z", "y", "x")


class TestRoundtrip:
    """Tests for save/load roundtrip consistency."""

    def test_roundtrip_3d(self, tmp_path):
        """Save and load preserves 3D data and attributes."""
        original_data = np.random.rand(6, 4, 2).astype(np.float32)
        original = xr.DataArray(
            original_data,
            dims=["z", "y", "x"],
            attrs={"voxdim": [1.0, 0.5, 2.0]},
        )

        nifti_path = tmp_path / "roundtrip_3d.nii.gz"
        save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path, load_sidecar=False)

        np.testing.assert_array_almost_equal(
            np.asarray(loaded), original_data, decimal=5
        )

    def test_roundtrip_4d(self, tmp_path):
        """Save and load preserves 4D data."""
        original_data = np.random.rand(3, 6, 4, 2).astype(np.float32)
        original = xr.DataArray(original_data, dims=["time", "z", "y", "x"])

        nifti_path = tmp_path / "roundtrip_4d.nii.gz"
        save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path, load_sidecar=False)

        np.testing.assert_array_almost_equal(
            np.asarray(loaded), original_data, decimal=5
        )
