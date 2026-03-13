"""Unit tests for confusius.io.autc module."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from confusius.io.autc import AUTCDAT, AUTCDATsLoader, load_autc_metadata


def _create_autc_metadata(
    path: Path,
    n_x: int = 4,
    n_z: int = 6,
    transmit_frequency_mhz: float = 15.0,
    speed_of_sound_mm_us: float = 1.54,
    n_elements: int = 128,
    pitch_mm: float = 0.3,
    angles_deg: list | None = None,
    dt_bf_s: float = 0.002,
    dt_single_pw_s: float = 0.0002,
) -> None:
    """Create a synthetic AUTC MAT metadata file (HDF5 / MAT v7.3).

    Parameters
    ----------
    path : Path
        Path to create the MAT file.
    n_x : int, default: 4
        Number of lateral samples.
    n_z : int, default: 6
        Number of axial samples.
    transmit_frequency_mhz : float, default: 15.0
        Transmit frequency in MHz (as stored in the MAT file).
    speed_of_sound_mm_us : float, default: 1.54
        Speed of sound in mm/µs (as stored in the MAT file).
    n_elements : int, default: 128
        Number of probe elements.
    pitch_mm : float, default: 0.3
        Inter-element pitch in mm.
    angles_deg : list, optional
        Plane wave angles in degrees. Defaults to [-10, 0, 10].
    dt_bf_s : float, default: 0.002
        Compound frame period in seconds.
    dt_single_pw_s : float, default: 0.0002
        Single plane wave period in seconds.
    """
    if angles_deg is None:
        angles_deg = [-10.0, 0.0, 10.0]

    with h5py.File(path, "w") as f:
        doppler = f.create_group("doppler")
        doppler["xAxis"] = np.linspace(0, n_x * 0.3, n_x)
        doppler["zAxis"] = np.linspace(0, n_z * 0.15, n_z)
        doppler["dtBF"] = np.array([dt_bf_s])
        doppler["dtSinglePlanewave"] = np.array([dt_single_pw_s])

        params = doppler.create_group("params")
        par_seq = params.create_group("parSeq")
        par_seq["TF"] = np.array([transmit_frequency_mhz])
        par_seq["Tne"] = np.array([n_elements])  # codespell:ignore
        par_seq["Tdx"] = np.array([pitch_mm])
        par_seq["c"] = np.array([speed_of_sound_mm_us])

        hq = par_seq.create_group("HQ")
        hq["angles"] = np.array(angles_deg)


def _create_autc_dat_file(
    path: Path,
    n_blocks: int = 1,
    n_x: int = 4,
    n_z: int = 6,
    n_frames: int = 3,
    start_index: int = 1,
    known_pattern: bool = False,
) -> None:
    """Create a synthetic AUTC DAT file for testing.

    Parameters
    ----------
    path : Path
        Path to create the DAT file.
    n_blocks : int, default: 1
        Number of blocks in the file.
    n_x : int, default: 4
        Number of lateral samples.
    n_z : int, default: 6
        Number of depth samples.
    n_frames : int, default: 3
        Number of frames per block.
    start_index : int, default: 1
        Starting acquisition block index (1-based in file).
    known_pattern : bool, default: False
        If True, fill each block with a constant complex value equal to
        `complex(block_idx + 1, 1)` for deterministic value checks.
    """
    header_dtype = np.dtype(
        [
            ("acquisition_block_index", np.int32),
            ("n_data_items", np.int32),
            ("n_z", np.int32),
            ("n_x", np.int32),
            ("n_frames", np.int32),
            ("field5", np.int32),
            ("field6", np.int32),
            ("field7", np.int32),
            ("field8", np.int32),
            ("field9", np.int32),
        ]
    )

    n_data_items = n_x * n_z * n_frames

    with open(path, "wb") as f:
        for block_idx in range(n_blocks):
            acquisition_index = start_index + block_idx * 4

            header = np.array(
                [
                    (
                        acquisition_index,
                        n_data_items,
                        n_z,
                        n_x,
                        n_frames,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                ],
                dtype=header_dtype,
            )

            data = np.random.randn(n_x, n_z, n_frames).astype(np.complex64) * (1 + 1j)
            if known_pattern:
                data = np.full(
                    (n_x, n_z, n_frames),
                    complex(block_idx + 1.0, 1.0),
                    dtype=np.complex64,
                )

            f.write(header.tobytes())
            f.write(data.tobytes())


@pytest.fixture
def single_autc_dat_file(tmp_path):
    """Create a single synthetic AUTC DAT file."""
    dat_path = tmp_path / "bf0part000.dat"
    _create_autc_dat_file(dat_path, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=1)
    return dat_path


@pytest.fixture
def single_autc_dat_file_known(tmp_path):
    """Create a single synthetic AUTC DAT file with a known pattern."""
    dat_path = tmp_path / "bf0part000.dat"
    _create_autc_dat_file(
        dat_path,
        n_blocks=2,
        n_x=4,
        n_z=6,
        n_frames=3,
        start_index=1,
        known_pattern=True,
    )
    return dat_path


@pytest.fixture
def multi_autc_dat_files(tmp_path):
    """Create multiple synthetic AUTC DAT files."""
    file1 = tmp_path / "bf0part000.dat"
    file2 = tmp_path / "bf0part001.dat"
    file3 = tmp_path / "bf0part002.dat"

    _create_autc_dat_file(file1, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=1)
    _create_autc_dat_file(file2, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=9)
    _create_autc_dat_file(file3, n_blocks=1, n_x=4, n_z=6, n_frames=3, start_index=17)

    return tmp_path


@pytest.fixture
def empty_autc_dat_file(tmp_path):
    """Create an empty AUTC DAT file (0 bytes)."""
    dat_path = tmp_path / "bf0part000.dat"
    dat_path.touch()
    return dat_path


class TestAUTCDAT:
    """Tests for `AUTCDAT` class."""

    def test_autcdat_loads_file(self, single_autc_dat_file):
        """`AUTCDAT` loads a valid DAT file successfully."""
        autc = AUTCDAT(single_autc_dat_file)

        assert autc.path.is_file()
        assert autc.n_blocks == 2
        assert autc.n_frames_per_block == 3

    def test_autcdat_shape_property(self, single_autc_dat_file):
        """`AUTCDAT` shape property returns correct tuple."""
        autc = AUTCDAT(single_autc_dat_file)

        assert autc.shape == (2, 4, 6, 3)  # (n_blocks, n_x, n_z, n_frames)

    def test_autcdat_len(self, single_autc_dat_file):
        """`AUTCDAT` `__len__` returns number of blocks."""
        autc = AUTCDAT(single_autc_dat_file)

        assert len(autc) == 2

    def test_autcdat_indexing_single_block(self, single_autc_dat_file):
        """`AUTCDAT` indexing with int returns single block."""
        autc = AUTCDAT(single_autc_dat_file)

        data = autc[0]

        assert data.shape == (4, 6, 3)
        assert data.dtype == np.complex64

    def test_autcdat_indexing_slice(self, single_autc_dat_file):
        """`AUTCDAT` indexing with slice returns multiple blocks."""
        autc = AUTCDAT(single_autc_dat_file)

        data = autc[0:2]

        assert data.shape == (2, 4, 6, 3)

    def test_autcdat_indexing_tuple(self, single_autc_dat_file):
        """`AUTCDAT` indexing with tuple returns sliced data."""
        autc = AUTCDAT(single_autc_dat_file)

        data = autc[0, :, :2, :2]

        assert data.shape == (4, 2, 2)

    def test_autcdat_acquisition_indices(self, single_autc_dat_file):
        """`AUTCDAT` acquisition_block_indices property works."""
        autc = AUTCDAT(single_autc_dat_file)

        indices = autc.acquisition_block_indices

        assert len(indices) == 2
        # Indices are returned as stored in file (1-based).
        # With start_index=1 and step=4: [1, 5].
        assert indices[0] == 1
        assert indices[1] == 5

    def test_autcdat_nonexistent_file(self, tmp_path):
        """`AUTCDAT` raises `ValueError` for non-existent file."""
        non_existent = tmp_path / "nonexistent.dat"

        with pytest.raises(ValueError):
            AUTCDAT(non_existent)

    def test_autcdat_indexing_values_known_pattern(self, single_autc_dat_file_known):
        """`AUTCDAT` indexing returns correct values for a known pattern."""
        autc = AUTCDAT(single_autc_dat_file_known)

        # Block 0 was filled with complex(1, 1), block 1 with complex(2, 1).
        block0 = autc[0]
        block1 = autc[1]

        assert block0[0, 0, 0] == complex(1.0, 1.0), "Block 0 value corrupted"
        assert block1[0, 0, 0] == complex(2.0, 1.0), "Block 1 value corrupted"


class TestAUTCDATsLoader:
    """Tests for `AUTCDATsLoader` class."""

    def test_autcdats_loader_loads_multiple_files(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexes multiple DAT files."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        assert len(loader.dats) == 3
        assert loader.shape[0] == 5  # Total blocks: 2 + 2 + 1

    def test_autcdats_loader_single_file(self, single_autc_dat_file):
        """`AUTCDATsLoader` works with single DAT file in a directory."""
        # AUTCDATsLoader expects a directory, so we use the parent directory
        loader = AUTCDATsLoader(single_autc_dat_file.parent, show_progress=False)

        assert len(loader.dats) == 1
        assert loader.shape[0] == 2

    def test_autcdats_loader_blocks_property(self, multi_autc_dat_files):
        """`AUTCDATsLoader` blocks property returns sorted acquisition indices."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        blocks = loader.blocks

        assert len(blocks) == 5
        # Blocks are returned as stored in file (1-based).
        # File 1: [1, 5], File 2: [9, 13], File 3: [17].
        assert np.array_equal(blocks, [1, 5, 9, 13, 17])

    def test_autcdats_loader_indexing_single_block(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexing with int returns single block."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        data = loader[0]

        assert data.shape == (1, 4, 6, 3)

    def test_autcdats_loader_indexing_slice(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexing with slice returns multiple blocks."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        data = loader[0:3]

        assert data.shape == (3, 4, 6, 3)

    def test_autcdats_loader_indexing_across_files(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexing across file boundaries works."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        data = loader[1:4]  # Block 1 (from file 1), blocks 8, 12 (from file 2).

        assert data.shape == (3, 4, 6, 3)

    def test_autcdats_loader_empty_file_warns(self, multi_autc_dat_files, monkeypatch):
        """`AUTCDATsLoader` skips empty DAT files with warning."""
        import warnings

        empty_file = multi_autc_dat_files / "bf0part003.dat"
        empty_file.touch()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

            assert len(loader.dats) == 3  # Empty file skipped.
            assert any(
                "Skipping empty AUTC DAT file" in str(warning.message) for warning in w
            )

    def test_autcdats_loader_no_matching_files(self, tmp_path):
        """`AUTCDATsLoader` raises `FileNotFoundError` when no files match pattern."""
        with pytest.raises(FileNotFoundError, match="No AUTC DAT files found"):
            AUTCDATsLoader(tmp_path, show_progress=False)

    def test_autcdats_loader_nonexistent_root(self, tmp_path):
        """`AUTCDATsLoader` raises `ValueError` for non-existent directory."""
        non_existent = tmp_path / "nonexistent"

        with pytest.raises(ValueError):
            AUTCDATsLoader(non_existent, show_progress=False)

    def test_autcdats_loader_indexing_tuple(self, multi_autc_dat_files):
        """`AUTCDATsLoader` tuple indexing returns correctly sliced data."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        # (block_slice, x_slice, z_slice, frame_slice)
        data = loader[0:2, :2, :3, :]

        assert data.shape == (2, 2, 3, 3)

    def test_autcdats_loader_indexing_tuple_int_subdim(self, multi_autc_dat_files):
        """`AUTCDATsLoader` tuple indexing with int sub-dimension returns size-1 dim."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        # An int in a sub-dimension should produce a size-1 output along that axis.
        data = loader[0:2, 0, :3, :]

        assert data.shape == (2, 1, 3, 3)

    def test_autcdats_loader_invalid_indexing_type(self, multi_autc_dat_files):
        """`AUTCDATsLoader` raises `TypeError` for invalid indexing type."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        with pytest.raises(
            TypeError, match="slice_object must be an int, slice, or tuple"
        ):
            loader["invalid"]  # type: ignore

    def test_autcdats_loader_indexing_with_list(self, multi_autc_dat_files):
        """`AUTCDATsLoader` raises `TypeError` for list indexing."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        with pytest.raises(TypeError):
            loader[[0, 1, 2]]  # type: ignore


@pytest.fixture
def autc_meta_file(tmp_path):
    """Create a synthetic AUTC MAT metadata file."""
    meta_path = tmp_path / "session_fus.mat"
    _create_autc_metadata(meta_path)
    return meta_path


class TestLoadAUTCMetadata:
    """Tests for `load_autc_metadata` function."""

    def test_unit_conversion_transmit_frequency(self, autc_meta_file):
        """`load_autc_metadata` converts transmit frequency from MHz to Hz."""
        meta = load_autc_metadata(autc_meta_file)

        # 15.0 MHz → 15_000_000 Hz.
        assert meta["transmit_frequency"] == pytest.approx(15.0e6)

    def test_unit_conversion_speed_of_sound(self, autc_meta_file):
        """`load_autc_metadata` converts speed of sound from mm/µs to m/s."""
        meta = load_autc_metadata(autc_meta_file)

        # 1.54 mm/µs → 1540 m/s.
        assert meta["speed_of_sound"] == pytest.approx(1540.0)

    def test_unit_conversion_compound_sampling_frequency(self, autc_meta_file):
        """`load_autc_metadata` inverts dtBF to get compound sampling frequency."""
        meta = load_autc_metadata(autc_meta_file)

        # dtBF = 0.002 s → CSF = 500 Hz.
        assert meta["compound_sampling_frequency"] == pytest.approx(500.0)

    def test_unit_conversion_pulse_repetition_frequency(self, autc_meta_file):
        """`load_autc_metadata` inverts dtSinglePlanewave to get PRF."""
        meta = load_autc_metadata(autc_meta_file)

        # dtSinglePlanewave = 0.0002 s → PRF = 5000 Hz.
        assert meta["pulse_repetition_frequency"] == pytest.approx(5000.0)

    def test_missing_file(self, tmp_path):
        """`load_autc_metadata` raises `ValueError` for a missing file."""
        with pytest.raises(ValueError):
            load_autc_metadata(tmp_path / "nonexistent.mat")
