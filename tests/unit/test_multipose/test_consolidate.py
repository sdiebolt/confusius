"""Unit tests for confusius.multipose module."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from confusius.io.scan import load_scan
from confusius.multipose import consolidate_poses

# ---------------------------------------------------------------------------
# Synthetic fixture helpers (shared with test_scan.py)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

# Shared probe geometry: typical linear probe (sizeY=1).
_SIZE_X = 8
_SIZE_Y = 1
_SIZE_Z = 12
_NPOSE = 3
_T = 5
_DX = 0.0003  # 0.3 mm lateral pitch (meters)
_DY = 0.0004  # elevation (meters)
_DZ = 0.0002  # 0.2 mm axial pitch (meters)

# voxelsToProbe: [dx, 0, 0, ox; 0, dy, 0, oy; 0, 0, -dz, oz; 0, 0, 0, 1]
_OX = -_DX * _SIZE_X / 2
_OY = 0.0
_OZ = -0.003
_VOXELS_TO_PROBE = np.array(
    [
        [_DX, 0.0, 0.0, _OX],
        [0.0, _DY, 0.0, _OY],
        [0.0, 0.0, -_DZ, _OZ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# probeToLab for multi-pose: (npose, 4, 4) — same rotation, varying translation.
_PROBE_TO_LAB_MULTI = np.stack(
    [
        np.eye(4, dtype=np.float64)
        + np.array([[0, 0, 0, i * 0.001], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        for i in range(_NPOSE)
    ]
)


def _write_scan_metadata(
    f: h5py.File,
    mode: str,
    size_x: int,
    size_y: int,
    size_z: int,
    npose: int,
    nscan_repeat: int | None,
    nblock_repeat: int | None,
    voxels_to_probe: np.ndarray,
    probe_to_lab: np.ndarray,
    time_data: np.ndarray,
) -> None:
    """Write common metadata groups to a synthetic SCAN HDF5 file."""
    acq = f.create_group("acqMetaData")

    acq.create_dataset(
        "acquisitionMode", data=np.array([[mode.encode()]], dtype=object)
    )

    img = acq.create_group("imgDim")
    img.create_dataset("sizeX", data=np.array([[float(size_x)]]))
    img.create_dataset("sizeY", data=np.array([[float(size_y)]]))
    img.create_dataset("sizeZ", data=np.array([[float(size_z)]]))
    img.create_dataset("npose", data=np.array([[float(npose)]]))
    img.create_dataset(
        "nscanRepeat", data=np.array([[float(nscan_repeat if nscan_repeat else 1)]])
    )
    img.create_dataset(
        "nblockRepeat",
        data=np.array([[float(nblock_repeat if nblock_repeat else 1)]]),
    )
    img.create_dataset("dim7", data=np.array([[1.0]]))

    vox = acq.create_group("voxDim")
    vox.create_dataset("dx", data=np.array([[_DX]]))
    vox.create_dataset("dy", data=np.array([[_DY]]))
    vox.create_dataset("dz", data=np.array([[_DZ]]))
    vox.create_dataset("dt", data=np.array([[0.05]]))

    acq.create_dataset("time", data=time_data)
    acq.create_dataset("timeOriginal", data=time_data)
    acq.create_dataset("probeToLab", data=probe_to_lab)
    acq.create_dataset("voxelsToProbe", data=voxels_to_probe)

    meta = f.create_group("scanMetaData")
    for key, val in [
        ("Subject_tag", "sub-01"),
        ("Session_tag", "ses-01"),
        ("Scan_tag", "scan-01"),
        ("Project_tag", "proj-01"),
        ("Date", "2025-01-01"),
        ("Neuroscan_version", "1.0"),
        ("Machine_SN", "SN-0001"),
        ("User_name", "user"),
        ("Type", "source "),
        ("Code", " neuroSoft"),
        ("Comment", ""),
        ("Tag", ""),
    ]:
        meta.create_dataset(key, data=np.array([[val.encode()]], dtype=object))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scan_3d_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file (linear probe, sizeY=1)."""
    path = tmp_path / "test_3dscan.scan"
    time_data = np.zeros((_NPOSE, 1), dtype=np.float64)
    data = _RNG.random((_NPOSE, 1, _SIZE_Z, _SIZE_Y, _SIZE_X), dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SIZE_X,
            size_y=_SIZE_Y,
            size_z=_SIZE_Z,
            npose=_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_VOXELS_TO_PROBE,
            probe_to_lab=_PROBE_TO_LAB_MULTI,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_4d_path(tmp_path: Path) -> Path:
    """Create a synthetic 4Dscan HDF5 file (linear probe, sizeY=1)."""
    path = tmp_path / "test_4dscan.scan"
    time_flat = np.arange(_T * _NPOSE, dtype=np.float64).reshape(_T * _NPOSE, 1) * 0.1
    data = _RNG.random((_T, _NPOSE, 1, _SIZE_Z, _SIZE_Y, _SIZE_X), dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="4Dscan",
            size_x=_SIZE_X,
            size_y=_SIZE_Y,
            size_z=_SIZE_Z,
            npose=_NPOSE,
            nscan_repeat=_T,
            nblock_repeat=None,
            voxels_to_probe=_VOXELS_TO_PROBE,
            probe_to_lab=_PROBE_TO_LAB_MULTI,
            time_data=time_flat,
        )
    return path


@pytest.fixture
def scan_2d_path(tmp_path: Path) -> Path:
    """Create a synthetic 2Dscan HDF5 file (linear probe, sizeY=1)."""
    path = tmp_path / "test_2dscan.scan"
    time_data = np.linspace(0.0, (_T - 1) * 0.05, _T).reshape(1, _T)
    data = _RNG.random((_T, _SIZE_Z, _SIZE_Y, _SIZE_X), dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="2Dscan",
            size_x=_SIZE_X,
            size_y=_SIZE_Y,
            size_z=_SIZE_Z,
            npose=1,
            nscan_repeat=None,
            nblock_repeat=_T,
            voxels_to_probe=_VOXELS_TO_PROBE,
            probe_to_lab=np.eye(4, dtype=np.float64),
            time_data=time_data,
        )
    return path


# probeToLab for a non-uniform sweep: pose translations are non-evenly spaced,
# so consolidated z positions will be irregular.
_PROBE_TO_LAB_IRREGULAR = np.stack(
    [
        np.eye(4, dtype=np.float64)
        + np.array([[0, 0, 0, t], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        for t in [0.0, 0.001, 0.005]  # gaps: 1 mm, 4 mm — clearly non-uniform
    ]
)

# probeToLab for a varying-rotation sweep: each pose has a different rotation block,
# so consolidate_poses should raise ValueError.
_ANGLE = np.pi / 6  # 30° rotation increments
_PROBE_TO_LAB_VARYING_ROTATION = np.stack(
    [
        np.array(
            [
                [np.cos(i * _ANGLE), 0, np.sin(i * _ANGLE), i * 0.001],
                [0, 1, 0, 0],
                [-np.sin(i * _ANGLE), 0, np.cos(i * _ANGLE), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        for i in range(_NPOSE)
    ]
)

# probeToLab for a 2D sweep (non-purely-1D): poses vary in both x and y directions.
_PROBE_TO_LAB_2D_SWEEP = np.stack(
    [
        np.eye(4, dtype=np.float64)
        + np.array(
            [
                [0, 0, 0, i * 0.001],
                [0, 0, 0, i * 0.001],  # both x and y vary → 2D sweep
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        for i in range(_NPOSE)
    ]
)


@pytest.fixture
def scan_3d_irregular_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file with non-uniformly spaced poses."""
    path = tmp_path / "test_3dscan_irregular.scan"
    time_data = np.zeros((_NPOSE, 1), dtype=np.float64)
    data = _RNG.random((_NPOSE, 1, _SIZE_Z, _SIZE_Y, _SIZE_X), dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SIZE_X,
            size_y=_SIZE_Y,
            size_z=_SIZE_Z,
            npose=_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_VOXELS_TO_PROBE,
            probe_to_lab=_PROBE_TO_LAB_IRREGULAR,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d_varying_rotation_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file with a different rotation per pose."""
    path = tmp_path / "test_3dscan_varying_rotation.scan"
    time_data = np.zeros((_NPOSE, 1), dtype=np.float64)
    data = _RNG.random((_NPOSE, 1, _SIZE_Z, _SIZE_Y, _SIZE_X), dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SIZE_X,
            size_y=_SIZE_Y,
            size_z=_SIZE_Z,
            npose=_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_VOXELS_TO_PROBE,
            probe_to_lab=_PROBE_TO_LAB_VARYING_ROTATION,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d_2d_sweep_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file with a 2D (non-purely-1D) sweep.

    Uses a matrix probe (sizeY=4) so the elevation spread creates variation in
    both the sweep direction and the perpendicular direction, making the sweep
    detectable as non-purely-1D by the SVD check.
    """
    size_y_matrix = 4
    path = tmp_path / "test_3dscan_2d_sweep.scan"
    time_data = np.zeros((_NPOSE, 1), dtype=np.float64)
    data = _RNG.random((_NPOSE, 1, _SIZE_Z, size_y_matrix, _SIZE_X), dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SIZE_X,
            size_y=size_y_matrix,
            size_z=_SIZE_Z,
            npose=_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_VOXELS_TO_PROBE,
            probe_to_lab=_PROBE_TO_LAB_2D_SWEEP,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d(scan_3d_path: Path) -> xr.DataArray:
    """Load a synthetic 3Dscan DataArray (linear probe, sizeY=1)."""
    return load_scan(scan_3d_path)


@pytest.fixture
def scan_4d(scan_4d_path: Path) -> xr.DataArray:
    """Load a synthetic 4Dscan DataArray (linear probe, sizeY=1)."""
    return load_scan(scan_4d_path)


@pytest.fixture
def scan_2d(scan_2d_path: Path) -> xr.DataArray:
    """Load a synthetic 2Dscan DataArray (linear probe, sizeY=1)."""
    return load_scan(scan_2d_path)


# ---------------------------------------------------------------------------
# Tests: consolidate_poses
# ---------------------------------------------------------------------------


class TestConsolidatePoses:
    """Tests for consolidate_poses."""

    def test_3dscan_output_dims(self, scan_3d: xr.DataArray) -> None:
        """3Dscan consolidation yields dims (z, y, x) with no pose dimension."""
        result = consolidate_poses(scan_3d)
        assert result.dims == ("z", "y", "x")
        assert "pose" not in result.dims

    def test_4dscan_output_dims(self, scan_4d: xr.DataArray) -> None:
        """4Dscan consolidation yields dims (time, z, y, x) with no pose dimension."""
        result = consolidate_poses(scan_4d)
        assert result.dims == ("time", "z", "y", "x")
        assert "pose" not in result.dims

    def test_consolidated_z_length(self, scan_3d: xr.DataArray) -> None:
        """Consolidated z length equals npose * sizeY."""
        result = consolidate_poses(scan_3d)
        assert result.sizes["z"] == _NPOSE * _SIZE_Y

    def test_consolidated_z_monotonic(self, scan_3d: xr.DataArray) -> None:
        """Consolidated z coordinate is monotonically increasing."""
        result = consolidate_poses(scan_3d)
        assert np.all(np.diff(result.coords["z"].values) > 0)

    def test_consolidated_z_regular(self, scan_3d: xr.DataArray) -> None:
        """Consolidated z coordinate is regularly spaced (within default rtol)."""
        result = consolidate_poses(scan_3d)
        diffs = np.diff(result.coords["z"].values)
        np.testing.assert_allclose(diffs, diffs.mean(), rtol=0.01)

    def test_consolidated_z_units_mm(self, scan_3d: xr.DataArray) -> None:
        """Consolidated z coordinate has units='mm'."""
        result = consolidate_poses(scan_3d)
        assert result.coords["z"].attrs.get("units") == "mm"

    def test_consolidated_z_voxdim(self, scan_3d: xr.DataArray) -> None:
        """Consolidated z voxdim equals the mean spacing in mm."""
        result = consolidate_poses(scan_3d)
        diffs = np.diff(result.coords["z"].values)
        np.testing.assert_allclose(
            result.coords["z"].attrs["voxdim"], abs(diffs.mean()), rtol=1e-10
        )

    def test_physical_to_lab_consolidated_shape(self, scan_3d: xr.DataArray) -> None:
        """physical_to_lab becomes (4, 4) after consolidation."""
        result = consolidate_poses(scan_3d)
        A = np.asarray(result.attrs["affines"]["physical_to_lab"])
        assert A.shape == (4, 4)

    def test_physical_to_lab_consolidated_rotation_orthogonal(
        self, scan_3d: xr.DataArray
    ) -> None:
        """Consolidated affine rotation block is orthogonal (non-singular)."""
        result = consolidate_poses(scan_3d)
        A = np.asarray(result.attrs["affines"]["physical_to_lab"])
        R = A[:3, :3]
        # R^T @ R should be the identity for an orthogonal matrix.
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_4dscan_preserves_time_coord(self, scan_4d: xr.DataArray) -> None:
        """4Dscan consolidation preserves the time coordinate unchanged."""
        result = consolidate_poses(scan_4d)
        np.testing.assert_array_equal(
            result.coords["time"].values, scan_4d.coords["time"].values
        )

    def test_4dscan_pose_time_present(self, scan_4d: xr.DataArray) -> None:
        """4Dscan consolidation produces a pose_time coordinate."""
        result = consolidate_poses(scan_4d)
        assert "pose_time" in result.coords

    def test_4dscan_pose_time_dims(self, scan_4d: xr.DataArray) -> None:
        """Consolidated pose_time has dims (time, z)."""
        result = consolidate_poses(scan_4d)
        assert result.coords["pose_time"].dims == ("time", "z")

    def test_4dscan_pose_time_shape(self, scan_4d: xr.DataArray) -> None:
        """Consolidated pose_time has shape (nscanRepeat, npose * sizeY)."""
        result = consolidate_poses(scan_4d)
        assert result.coords["pose_time"].shape == (_T, _NPOSE * _SIZE_Y)

    def test_4dscan_pose_time_units(self, scan_4d: xr.DataArray) -> None:
        """Consolidated pose_time has units='s'."""
        assert consolidate_poses(scan_4d).coords["pose_time"].attrs.get("units") == "s"

    def test_4dscan_pose_time_values(self, scan_4d: xr.DataArray) -> None:
        """Each consolidated z-slice has the timestamp of its source pose."""
        result = consolidate_poses(scan_4d)
        orig_pt = scan_4d.coords["pose_time"].values  # (T, npose)
        # Recover which original pose each consolidated z-slice came from.
        ptl = np.asarray(scan_4d.attrs["affines"]["physical_to_lab"])
        z_mm = scan_4d.coords["z"].values
        r0 = ptl[:, :3, 0]
        t_lab = ptl[:, :3, 3]
        lab_pos = (
            r0[:, np.newaxis, :] * z_mm[np.newaxis, :, np.newaxis]
            + t_lab[:, np.newaxis, :]
        )
        lab_pos_flat = lab_pos.reshape(-1, 3)
        centered = lab_pos_flat - lab_pos_flat.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        sweep_axis = vt[0]
        if sweep_axis[np.argmax(np.abs(sweep_axis))] < 0:
            sweep_axis = -sweep_axis
        proj = lab_pos_flat @ sweep_axis
        sorted_flat = np.argsort(proj)
        pose_idx = sorted_flat // len(z_mm)
        expected = orig_pt[:, pose_idx]
        np.testing.assert_array_equal(result.coords["pose_time"].values, expected)

    def test_3dscan_no_pose_time(self, scan_3d: xr.DataArray) -> None:
        """3Dscan consolidation produces no pose_time coordinate."""
        result = consolidate_poses(scan_3d)
        assert "pose_time" not in result.coords

    def test_no_pose_dim_raises(self, scan_2d: xr.DataArray) -> None:
        """consolidate_poses raises ValueError when there is no pose dimension."""
        with pytest.raises(ValueError, match="no 'pose' dimension"):
            consolidate_poses(scan_2d)

    def test_irregular_positions_raises(self, scan_3d_irregular_path: Path) -> None:
        """consolidate_poses raises ValueError when positions are not regularly spaced."""
        da = load_scan(scan_3d_irregular_path)
        with pytest.raises(ValueError, match="not regularly spaced"):
            consolidate_poses(da)

    def test_non_1d_sweep_warns(self, scan_3d_2d_sweep_path: Path) -> None:
        """consolidate_poses warns when the sweep has a significant secondary component.

        The 2D sweep fixture also produces irregular spacings after projection onto the
        diagonal axis, so a ValueError follows the warning. Both are expected here.
        """
        da = load_scan(scan_3d_2d_sweep_path)
        with pytest.warns(UserWarning, match="not purely 1D"):
            with pytest.raises(ValueError):
                consolidate_poses(da)

    def test_varying_rotation_raises(self, scan_3d_varying_rotation_path: Path) -> None:
        """consolidate_poses raises ValueError when rotation varies across poses."""
        da = load_scan(scan_3d_varying_rotation_path)
        with pytest.raises(ValueError, match="not constant across poses"):
            consolidate_poses(da)

    def test_invalid_sweep_dim_raises(self, scan_3d: xr.DataArray) -> None:
        """consolidate_poses raises ValueError for an unrecognised sweep_dim."""
        with pytest.raises(ValueError, match="sweep_dim must be one of"):
            consolidate_poses(scan_3d, sweep_dim="w")

    def test_custom_affines_key(self, scan_3d: xr.DataArray) -> None:
        """consolidate_poses uses the affines_key argument to select the affine."""
        # Copy the existing affine under a custom key and verify the result is
        # identical to the default call.
        affine = scan_3d.attrs["affines"]["physical_to_lab"]
        custom_attrs = {
            **scan_3d.attrs,
            "affines": {"my_affine": affine},
        }
        da_custom = scan_3d.assign_attrs(custom_attrs)
        result_default = consolidate_poses(scan_3d)
        result_custom = consolidate_poses(da_custom, affines_key="my_affine")
        np.testing.assert_array_equal(result_default.values, result_custom.values)
        np.testing.assert_array_equal(
            result_default.coords["z"].values, result_custom.coords["z"].values
        )

    @pytest.mark.parametrize("sweep_dim", ["y", "x"])
    def test_non_z_sweep_dims(self, sweep_dim: str) -> None:
        """consolidate_poses correctly merges poses for y- and x-axis sweeps.

        This test constructs a DataArray whose affine translates along the requested
        sweep column and verifies that:

        - the output dims are ``(sweep_dim, <other1>, <other2>)`` with no ``pose``,
        - the consolidated length equals ``npose * n_sweep``,
        - each consolidated slice contains exactly the data values from the correct
          ``(pose, sweep_dim)`` combination.
        """
        npose = 3
        sizes = {"z": 2, "y": 4, "x": 3}
        intra_step = 0.2  # mm voxel pitch

        _SWEEP_DIM_TO_COL = {"z": 0, "y": 1, "x": 2}
        sweep_col = _SWEEP_DIM_TO_COL[sweep_dim]
        n_sweep = sizes[sweep_dim]
        inter_step = n_sweep * intra_step  # poses tile without gaps

        rng = np.random.default_rng(7)
        data = rng.random((npose, sizes["z"], sizes["y"], sizes["x"]))

        affines = np.stack([np.eye(4) for _ in range(npose)])
        for i in range(npose):
            affines[i, :3, 3][sweep_col] = i * inter_step

        coords: dict[str, xr.DataArray] = {
            "pose": xr.DataArray(np.arange(npose), dims=["pose"]),
            **{
                d: xr.DataArray(
                    np.arange(sizes[d]) * intra_step,
                    dims=[d],
                    attrs={"units": "mm", "voxdim": intra_step},
                )
                for d in ("z", "y", "x")
            },
        }
        da = xr.DataArray(
            data,
            dims=["pose", "z", "y", "x"],
            coords=coords,
            attrs={"affines": {"physical_to_lab": affines}},
        )

        result = consolidate_poses(da, sweep_dim=sweep_dim)

        other_dims = [d for d in ["z", "y", "x"] if d != sweep_dim]
        assert result.dims == tuple([sweep_dim] + other_dims)
        assert "pose" not in result.dims
        assert result.sizes[sweep_dim] == npose * n_sweep

        # Verify data values: for each pose p and local sweep index si, the
        # consolidated flat index is p*n_sweep + si (poses are sorted ascending).
        for p in range(npose):
            for si in range(n_sweep):
                flat_idx = p * n_sweep + si
                # Expected slice: fix pose and sweep dim, free other dims.
                dim_order = ["z", "y", "x"]
                idx_dict: dict[str, int | slice] = {d: slice(None) for d in dim_order}
                idx_dict[sweep_dim] = si
                idx_tuple = (p,) + tuple(idx_dict[d] for d in dim_order)
                expected = data[idx_tuple]
                np.testing.assert_array_equal(result.values[flat_idx], expected)
