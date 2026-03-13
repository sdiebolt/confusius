"""Integration tests for AUTCDAT to Zarr conversion."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import warnings
import xarray as xr
import zarr

from confusius.io.autc import convert_autc_dats_to_zarr


class TestAUTCConversion:
    """Integration tests for AUTCDAT conversion to Zarr."""

    def test_conversion(self, synthetic_autc_session, tmp_path):
        """Test full round-trip conversion and output verification.

        Tests:
        - Conversion of multiple DAT files.
        - Coordinates and metadata read from MAT file.
        - Block times provided by user.
        - Output verification using Xarray.
        """
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        # Block times for the 4 acquired blocks.
        block_times = np.array([1.0, 3.0, 5.0, 7.0])

        convert_autc_dats_to_zarr(
            session_dir,
            meta_path,
            output_path,
            block_times=block_times,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 12
            assert ds["iq"].dtype == np.complex64
            assert np.all(np.isfinite(ds["iq"].values))

            iq = ds["iq"].values
            np.testing.assert_allclose(iq[0, 0, 0, 0], 1 + 1j)
            np.testing.assert_allclose(iq[3, 0, 0, 0], 101 + 101j)

            # Coordinates come from the MAT file (linspace(-10, 10, 4) / linspace(0, 20, 6)).
            np.testing.assert_allclose(ds["x"].values, np.linspace(-10, 10, 4))
            np.testing.assert_allclose(ds["y"].values, np.linspace(0, 20, 6))
            np.testing.assert_allclose(ds["z"].values, np.array([0.0]))

            # Time computed from block_times + intra-block offset at CSF = 500 Hz.
            expected_times = np.array(
                [
                    1.0,
                    1.002,
                    1.004,
                    3.0,
                    3.002,
                    3.004,
                    5.0,
                    5.002,
                    5.004,
                    7.0,
                    7.002,
                    7.004,
                ]
            )
            np.testing.assert_allclose(ds["time"].values, expected_times, rtol=1e-5)

            assert ds["time"].attrs["units"] == "s"
            assert ds["z"].attrs["units"] == "mm"
            assert ds["y"].attrs["units"] == "mm"
            assert ds["x"].attrs["units"] == "mm"

            # Metadata attributes from MAT file.
            assert ds["iq"].attrs["transmit_frequency"] == pytest.approx(3.0e6)
            assert ds["iq"].attrs["probe_n_elements"] == 64
            assert ds["iq"].attrs["probe_pitch"] == pytest.approx(0.25)
            assert ds["iq"].attrs["sound_velocity"] == pytest.approx(1480.0)
            assert ds["iq"].attrs["plane_wave_angles"] == [-15.0, 0.0, 15.0]

    def test_overwrite_existing(self, synthetic_autc_session, tmp_path):
        """Test overwriting existing Zarr output."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            session_dir, meta_path, output_path, show_progress=False
        )
        first_shape = zarr.open_group(output_path, mode="r")["iq"].shape

        convert_autc_dats_to_zarr(
            session_dir, meta_path, output_path, overwrite=True, show_progress=False
        )
        second_shape = zarr.open_group(output_path, mode="r")["iq"].shape

        assert first_shape == second_shape

    def test_frames_per_shard_not_multiple(self, synthetic_autc_session, tmp_path):
        """Raise `ValueError` when `frames_per_shard` is not a multiple of `frames_per_chunk`."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        with pytest.raises(
            ValueError,
            match="frames_per_shard.*must be a multiple of.*frames_per_chunk",
        ):
            convert_autc_dats_to_zarr(
                session_dir,
                meta_path,
                output_path,
                frames_per_chunk=3,
                frames_per_shard=5,
                show_progress=False,
            )

    def test_block_times_without_frequency_error(
        self, synthetic_autc_session, tmp_path
    ):
        """Raise `ValueError` when `block_times` length mismatches block count."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        with pytest.raises(
            ValueError,
            match="block_times length.*does not match",
        ):
            convert_autc_dats_to_zarr(
                session_dir,
                meta_path,
                output_path,
                block_times=[0.0, 1.0, 2.0],  # 3 times for 4 blocks → mismatch.
                show_progress=False,
            )

    def test_zarr_kwargs_override_warning(self, synthetic_autc_session, tmp_path):
        """Warn when `zarr_kwargs` contains keys that are handled by function parameters."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            convert_autc_dats_to_zarr(
                session_dir,
                meta_path,
                output_path,
                zarr_kwargs={"shape": (10, 6, 1, 4)},
                show_progress=False,
            )

        assert any(
            "zarr_kwargs contains keys that are handled" in str(warning.message)
            for warning in w
        )

    def test_with_sharding(self, synthetic_autc_session, tmp_path):
        """Test conversion with Zarr sharding enabled."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            session_dir,
            meta_path,
            output_path,
            frames_per_chunk=3,
            frames_per_shard=6,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 12
            assert ds["iq"].dtype == np.complex64
            assert np.all(np.isfinite(ds["iq"].values))

        iq_array = zarr.open_group(output_path, mode="r")["iq"]
        iq_metadata = iq_array.metadata
        assert hasattr(iq_metadata, "shards") or "shards" in dir(iq_metadata)

    def test_cleanup_on_error(self, synthetic_autc_session, tmp_path):
        """Test that incomplete zarr store is cleaned up on error."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        mock_array = MagicMock()
        mock_array.__setitem__ = MagicMock(
            side_effect=Exception("Simulated error during conversion")
        )

        class MockZarrGroup:
            def __init__(self, real_group):
                self._real_group = real_group

            def create_array(self, name, **kwargs):
                if name == "iq":
                    return mock_array
                return self._real_group.create_array(name, **kwargs)

            def __getitem__(self, key):
                return self._real_group[key]

            def __getattr__(self, name):
                return getattr(self._real_group, name)

        original_open_group = zarr.open_group

        def mock_open_group(*args, **kwargs):
            real_group = original_open_group(*args, **kwargs)
            return MockZarrGroup(real_group)

        with patch("confusius.io.autc.zarr.open_group", side_effect=mock_open_group):
            with pytest.raises(Exception, match="Simulated error during conversion"):
                convert_autc_dats_to_zarr(
                    session_dir,
                    meta_path,
                    output_path,
                    show_progress=False,
                )

        assert not output_path.exists()

    def test_time_from_sampling_frequency(self, synthetic_autc_session, tmp_path):
        """Test time coordinate computed from compound_sampling_frequency (no block_times).

        When block_times is not provided, time is computed as:
        time[i] = i / compound_sampling_frequency (from MAT file, 500 Hz).
        """
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            session_dir, meta_path, output_path, show_progress=False
        )

        with xr.open_zarr(output_path) as ds:
            assert "time" in ds.coords
            # CSF = 500 Hz from the synthetic MAT file.
            expected_times = np.arange(12) / 500.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_first_blocks(self, synthetic_autc_session, tmp_path):
        """Test skipping the first blocks during conversion."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            session_dir,
            meta_path,
            output_path,
            skip_first_blocks=1,
            show_progress=False,
        )

        with xr.open_zarr(output_path) as ds:
            # 4 blocks * 3 frames = 12, after skipping 1 block = 9 frames.
            assert ds["iq"].shape[0] == 9
            # Data starts from block 2 (values 101+101j).
            np.testing.assert_allclose(ds["iq"].values[0, 0, 0, 0], 101 + 101j)
            # Time is continuous starting from frame 3 (CSF = 500 Hz).
            expected_times = np.arange(3, 12) / 500.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_last_blocks(self, synthetic_autc_session, tmp_path):
        """Test skipping the last blocks during conversion."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            session_dir,
            meta_path,
            output_path,
            skip_last_blocks=1,
            show_progress=False,
        )

        with xr.open_zarr(output_path) as ds:
            # 4 blocks * 3 frames = 12, after skipping 1 block = 9 frames.
            assert ds["iq"].shape[0] == 9
            np.testing.assert_allclose(ds["iq"].values[0, 0, 0, 0], 1 + 1j)
            expected_times = np.arange(9) / 500.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_both_first_and_last(self, synthetic_autc_session, tmp_path):
        """Test skipping both first and last blocks with block_times."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        # block_times for the 2 retained middle blocks.
        block_times = np.array([1.0, 2.0])

        convert_autc_dats_to_zarr(
            session_dir,
            meta_path,
            output_path,
            block_times=block_times,
            skip_first_blocks=1,
            skip_last_blocks=1,
            show_progress=False,
        )

        with xr.open_zarr(output_path) as ds:
            # 4 blocks * 3 frames = 12, after skipping 2 blocks = 6 frames.
            assert ds["iq"].shape[0] == 6
            expected_times = np.array([1.0, 1.002, 1.004, 2.0, 2.002, 2.004])
            np.testing.assert_allclose(ds["time"].values, expected_times, rtol=1e-5)

    def test_skip_blocks_error(self, synthetic_autc_session, tmp_path):
        """Test error when trying to skip too many blocks."""
        session_dir, meta_path = synthetic_autc_session
        output_path = tmp_path / "output.zarr"

        with pytest.raises(ValueError, match="Cannot skip"):
            convert_autc_dats_to_zarr(
                session_dir,
                meta_path,
                output_path,
                skip_first_blocks=2,
                skip_last_blocks=2,  # Would skip all 4 blocks.
                show_progress=False,
            )
