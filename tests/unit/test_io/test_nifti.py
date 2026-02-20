"""Unit tests for confusius.io.nifti module."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import xarray as xr

from confusius.io.nifti import load_nifti, save_nifti


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
        da = load_nifti(nifti_2d_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("y", "x")
        assert da.shape == (6, 8)
        assert da.dtype == np.float32

    def test_load_3d_nifti(self, nifti_3d_path: Path) -> None:
        """Loading 3D NIfTI creates DataArray with spatial dims only."""
        da = load_nifti(nifti_3d_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("z", "y", "x")
        assert da.shape == (6, 8, 10)
        assert da.dtype == np.float32

    def test_load_4d_nifti(self, nifti_4d_path: Path) -> None:
        """Loading 4D NIfTI creates DataArray with time dimension."""
        da = load_nifti(nifti_4d_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("time", "z", "y", "x")
        assert da.shape == (6, 8, 10, 12)
        assert da.dtype == np.float64

    def test_load_nifti_units_from_header(self, tmp_path: Path) -> None:
        """Units on coordinate attrs come from the NIfTI header, not hardcoded."""
        data = np.random.rand(4, 3, 2).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nifti_path = tmp_path / "with_units.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        assert da.coords["z"].attrs["units"] == "mm"
        assert da.coords["y"].attrs["units"] == "mm"
        assert da.coords["x"].attrs["units"] == "mm"

    def test_load_nifti_unknown_units_omits_units_attr(self, tmp_path: Path) -> None:
        """Coordinates have no 'units' attr when the NIfTI header declares unknown."""
        data = np.random.rand(4, 3, 2).astype(np.float32)
        # Default nibabel image has xyzt_units=0 (unknown).
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_xyzt_units(xyz="unknown", t="unknown")
        nifti_path = tmp_path / "unknown_units.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        assert "units" not in da.coords["z"].attrs
        assert "units" not in da.coords["y"].attrs
        assert "units" not in da.coords["x"].attrs

    def test_load_nifti_both_affines_stores_qform_attrs(self, tmp_path: Path) -> None:
        """Loading a NIfTI with both valid affines stores qform direction attr."""
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        qform = np.diag([1.0, 1.0, 1.0, 1.0])
        data = np.random.rand(4, 3, 2).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        img.header.set_qform(qform, code=1)
        nifti_path = tmp_path / "both_affines.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        # Primary (z, y, x) coords come from sform.
        assert "z" in da.coords
        assert "y" in da.coords
        assert "x" in da.coords
        # Qform non-dimension coords are no longer created; only the ConfUSIus-
        # convention transform is stored in da.attrs["affines"].
        assert "z_qform" not in da.coords
        assert "probe_to_qform" in da.attrs["affines"]
        assert da.attrs["affines"]["probe_to_qform"].shape == (4, 4)

    def test_load_nifti_qform_only_uses_qform(self, tmp_path: Path) -> None:
        """Loading a NIfTI with only qform valid uses qform for primary coords."""
        qform = np.diag([2.0, 3.0, 4.0, 1.0])
        data = np.random.rand(4, 3, 2).astype(np.float32)
        img = nib.Nifti1Image(data, qform)
        img.header.set_sform(np.eye(4), code=0)
        img.header.set_qform(qform, code=1)
        nifti_path = tmp_path / "qform_only.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        assert "z" in da.coords
        assert "z_qform" not in da.coords
        # Primary z coord (NIfTI col 2, size 2) should reflect qform spacing of 4.0.
        np.testing.assert_allclose(da.coords["z"].values, [0.0, 4.0])
        # Affines dict uses the qform key, not sform.
        assert "probe_to_qform" in da.attrs["affines"]
        assert "probe_to_sform" not in da.attrs["affines"]

    def test_load_nifti_rotated_affine_probe_relative_coords(
        self, tmp_path: Path
    ) -> None:
        """Rotated affine produces probe-relative coords with correct spacing."""
        # 90° rotation around z maps x→y, y→-x with spacing [2, 3, 4].
        # The old diagonal approach (affine[col, col]) would give spacing 0 for
        # x and y; column norms give the correct values [2, 3, 4].
        spacing = np.array([2.0, 3.0, 4.0])
        origin = np.array([10.0, 20.0, 30.0])
        affine = np.array(
            [
                [0.0, -spacing[1], 0.0, origin[0]],
                [spacing[0], 0.0, 0.0, origin[1]],
                [0.0, 0.0, spacing[2], origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.random.rand(5, 4, 3).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "rotated.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        # NIfTI data shape is (x=5, y=4, z=3); ConfUSIus order is (z=3, y=4, x=5).
        # Coords start at origin and are stepped by column norm (not diagonal,
        # which would give spacing 0 for the rotated x and y axes here).
        np.testing.assert_allclose(
            da.coords["x"].values, [10.0, 12.0, 14.0, 16.0, 18.0]
        )
        np.testing.assert_allclose(da.coords["y"].values, [20.0, 23.0, 26.0, 29.0])
        np.testing.assert_allclose(da.coords["z"].values, [30.0, 34.0, 38.0])
        # Physical transform (4×4 probe→world affine) is stored in da.attrs["affines"].
        assert da.attrs["affines"]["probe_to_sform"].shape == (4, 4)

    def test_load_nifti_sheared_affine_correct_spacing(self, tmp_path: Path) -> None:
        """Sheared affine uses decompose44 spacings, not (wrong) column norms."""
        # Pure shear along x for the y axis: physical y-spacing is still 1,
        # but the column norm of the y column is sqrt(1 + 0.5^2) ≈ 1.118.
        affine = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.random.rand(3, 4, 5).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "sheared.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        # NIfTI data shape is (x=3, y=4, z=5); ConfUSIus order is (z=5, y=4, x=3).
        # All voxel spacings should be 1 (true pixdim), not sqrt(1.25) ≈ 1.118.
        np.testing.assert_allclose(da.coords["x"].values, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(da.coords["y"].values, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(da.coords["z"].values, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_load_nifti_physical_transform_maps_probe_to_world(
        self, tmp_path: Path
    ) -> None:
        """probe_to_sform correctly maps probe-space coords to world-space coords."""
        # Affine with rotation (90° around z) + zoom [2, 3, 4] + translation [10, 20, 30].
        spacing = np.array([2.0, 3.0, 4.0])
        origin = np.array([10.0, 20.0, 30.0])
        affine = np.array(
            [
                [0.0, -spacing[1], 0.0, origin[0]],
                [spacing[0], 0.0, 0.0, origin[1]],
                [0.0, 0.0, spacing[2], origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.zeros((5, 4, 3), dtype=np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "transform_check.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)
        A = da.attrs["affines"]["probe_to_sform"]

        # A_physical uses ConfUSIus (z, y, x) convention:
        # A @ [pz, py, px, 1] → [wz, wy, wx, 1].
        for i, j, k in [(0, 0, 0), (2, 1, 0), (4, 3, 2)]:
            world_xyz = (affine @ np.array([i, j, k, 1.0]))[:3]
            world_expected_zyx = world_xyz[[2, 1, 0]]
            px = float(da.coords["x"].values[i])
            py = float(da.coords["y"].values[j])
            pz = float(da.coords["z"].values[k])
            world_actual = (A @ np.array([pz, py, px, 1.0]))[:3]
            np.testing.assert_allclose(world_actual, world_expected_zyx, atol=1e-10)

    def test_load_nifti_no_valid_affine_warns(self, tmp_path: Path) -> None:
        """Loading a NIfTI with both codes 0 emits a warning."""
        import struct

        data = np.random.rand(4, 3, 2).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        # Use uncompressed .nii so we can patch the binary header directly;
        # nibabel resets codes to non-zero values during to_filename().
        nifti_path = tmp_path / "no_affine.nii"
        img.to_filename(nifti_path)

        # Patch qform_code (byte 252) and sform_code (byte 254) to 0.
        # Both are int16 LE per the NIfTI-1 specification.
        with open(nifti_path, "r+b") as f:
            f.seek(252)
            f.write(struct.pack("<hh", 0, 0))

        with pytest.warns(UserWarning, match="sform_code and qform_code are 0"):
            da = load_nifti(nifti_path)

        assert "z" in da.coords
        assert "z_qform" not in da.coords

    def test_load_nifti_lazy(self, tmp_path: Path) -> None:
        """Loading creates lazy Dask array."""
        import dask.array as dask_array

        data = np.random.rand(8, 6, 4).astype(np.float32)
        nifti_path = tmp_path / "test_lazy.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        result = load_nifti(nifti_path)

        assert isinstance(result.data, dask_array.Array)

    def test_load_nifti_with_sidecar(self, nifti_with_sidecar: Path) -> None:
        """Loading with sidecar JSON merges metadata."""
        da = load_nifti(nifti_with_sidecar)

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

    def test_save_non_uniform_coords_warns(self, tmp_path):
        """Saving a DataArray with non-uniform coordinate spacing emits a warning."""
        data = np.random.rand(4, 3).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "x"],
            coords={"z": [0.0, 1.0, 3.0, 6.0], "x": [0.0, 1.0, 2.0]},
        )

        with pytest.warns(UserWarning, match="non-uniform spacing"):
            save_nifti(da, tmp_path / "nonuniform.nii.gz")

    def test_save_creates_sidecar(self, tmp_path):
        """Saving always creates a JSON sidecar alongside the NIfTI file."""
        data = np.random.rand(4, 3, 2).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.linspace(0, 10, 4),
                "y": np.linspace(0, 5, 3),
                "x": np.linspace(0, 3, 2),
            },
            attrs={"custom": "value"},
        )

        output_path = tmp_path / "with_sidecar.nii.gz"
        save_nifti(da, output_path)

        sidecar_path = tmp_path / "with_sidecar.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar.get("custom") == "value"
        # Spatial coords are encoded in the NIfTI header, not duplicated in the sidecar.
        assert "x" not in sidecar
        assert "y" not in sidecar
        assert "z" not in sidecar


class TestRoundtrip:
    """Tests for save/load roundtrip consistency."""

    def test_roundtrip_preserves_full_affine(self, tmp_path):
        """Save and load roundtrip preserves the full affine (rotation + shear)."""
        # 45° rotation in XY plane + zoom + translation — not a diagonal affine.
        affine = np.array(
            [
                [1.41421356, -1.41421356, 0.0, 10.0],
                [1.41421356, 2.82842712, 0.0, -5.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.random.rand(3, 4, 5).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "rotated.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)
        out_path = tmp_path / "roundtrip_rotated.nii.gz"
        save_nifti(da, out_path)

        reloaded = nib.load(out_path)
        np.testing.assert_allclose(reloaded.header.get_sform(), affine, atol=1e-5)

    def test_roundtrip_3d(self, tmp_path):
        """Save and load preserves 3D data and attributes."""
        original_data = np.random.rand(6, 4, 2).astype(np.float32)
        original = xr.DataArray(
            original_data,
            dims=["z", "y", "x"],
        )

        nifti_path = tmp_path / "roundtrip_3d.nii.gz"
        save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path)

        np.testing.assert_array_almost_equal(
            np.asarray(loaded), original_data, decimal=5
        )

    def test_roundtrip_regular_timing(self, tmp_path):
        """Regular time coord roundtrips via RepetitionTime/DelayAfterTrigger."""
        time_values = np.array([0.5, 1.0, 1.5, 2.0])  # TR=0.5, delay=0.5
        original = xr.DataArray(
            np.random.rand(4, 6, 4, 2).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={"time": time_values},
        )

        nifti_path = tmp_path / "regular_timing.nii.gz"
        save_nifti(original, nifti_path)

        sidecar_path = tmp_path / "regular_timing.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["RepetitionTime"] == pytest.approx(0.5)
        assert sidecar["DelayAfterTrigger"] == pytest.approx(0.5)
        assert "VolumeTiming" not in sidecar

        loaded = load_nifti(nifti_path)
        np.testing.assert_allclose(loaded.coords["time"].values, time_values)
        assert "RepetitionTime" not in loaded.attrs
        assert "DelayAfterTrigger" not in loaded.attrs

    def test_roundtrip_regular_timing_no_delay(self, tmp_path):
        """Regular time coord starting at 0 omits DelayAfterTrigger."""
        time_values = np.array([0.0, 0.5, 1.0, 1.5])  # TR=0.5, no delay
        original = xr.DataArray(
            np.random.rand(4, 6, 4, 2).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={"time": time_values},
        )

        nifti_path = tmp_path / "no_delay.nii.gz"
        save_nifti(original, nifti_path)

        sidecar_path = tmp_path / "no_delay.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert "DelayAfterTrigger" not in sidecar

        loaded = load_nifti(nifti_path)
        np.testing.assert_allclose(loaded.coords["time"].values, time_values)

    def test_roundtrip_volume_timing(self, tmp_path):
        """Irregular time coord roundtrips via VolumeTiming; pixdim[4] is 0."""
        time_values = np.array([0.0, 1.5, 2.8, 4.6])  # non-uniform spacing
        original = xr.DataArray(
            np.random.rand(4, 6, 4, 2).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={"time": time_values},
        )

        nifti_path = tmp_path / "volume_timing.nii.gz"
        save_nifti(original, nifti_path)

        sidecar_path = tmp_path / "volume_timing.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["VolumeTiming"] == pytest.approx(time_values.tolist())
        assert "RepetitionTime" not in sidecar
        assert nib.load(nifti_path).header.get_zooms()[3] == pytest.approx(0.0)

        loaded = load_nifti(nifti_path)
        np.testing.assert_allclose(loaded.coords["time"].values, time_values)
        assert "VolumeTiming" not in loaded.attrs

    def test_load_repetition_time_pixdim_mismatch_warns(self, tmp_path):
        """Mismatched RepetitionTime in sidecar vs pixdim[4] emits a warning."""
        data = np.random.rand(3, 4, 5, 6).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms([1.0, 1.0, 1.0, 2.0])  # pixdim[4] = 2.0
        nifti_path = tmp_path / "mismatch.nii.gz"
        img.to_filename(nifti_path)

        sidecar = {"RepetitionTime": 1.0}  # contradicts pixdim[4]=2.0
        sidecar_path = tmp_path / "mismatch.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        with pytest.warns(UserWarning, match="RepetitionTime"):
            loaded = load_nifti(nifti_path)

        # Sidecar value wins: RepetitionTime=1.0, 6 volumes → [0, 1, 2, 3, 4, 5].
        np.testing.assert_allclose(loaded.coords["time"].values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_roundtrip_4d(self, tmp_path):
        """Save and load preserves 4D data."""
        original_data = np.random.rand(3, 6, 4, 2).astype(np.float32)
        original = xr.DataArray(original_data, dims=["time", "z", "y", "x"])

        nifti_path = tmp_path / "roundtrip_4d.nii.gz"
        save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path)

        np.testing.assert_array_almost_equal(
            np.asarray(loaded), original_data, decimal=5
        )
