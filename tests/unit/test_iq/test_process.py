"""Unit tests for IQ processing functions."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius.iq.process import (
    compute_axial_velocity_volume,
    compute_power_doppler_volume,
    compute_processed_volume_times,
    process_iq_blocks,
    process_iq_to_axial_velocity,
    process_iq_to_power_doppler,
)


class TestComputeProcessedVolumeTimes:
    """Tests for compute_processed_volume_times function."""

    @pytest.mark.parametrize(
        ("timing_ref", "expected"),
        [
            ("start", [0.0]),
            ("end", [0.9]),
            ("center", [0.45]),  # Index 4.5 interpolated: 0.4*0.5 + 0.5*0.5 = 0.45.
        ],
    )
    def test_single_window_timing_reference(self, timing_ref, expected):
        """Single window with different timing references."""
        volume_times = np.arange(10) * 0.1  # 0.0, 0.1, ..., 0.9
        result = compute_processed_volume_times(
            volume_times,
            n_input_volumes=10,
            clutter_window_width=10,
            clutter_window_stride=10,
            inner_window_width=10,
            inner_window_stride=10,
            timing_reference=timing_ref,
        )
        if timing_ref == "center":
            assert_allclose(result, expected)
        else:
            assert_array_equal(result, expected)

    def test_multiple_windows_correct_count(self):
        """Multiple windows produce correct number of output timestamps."""
        volume_times = np.arange(100) * 0.1
        result = compute_processed_volume_times(
            volume_times,
            n_input_volumes=100,
            clutter_window_width=50,
            clutter_window_stride=50,
            inner_window_width=25,
            inner_window_stride=25,
            timing_reference="center",
        )
        # 2 outer windows, 2 inner windows each = 4 output times.
        assert len(result) == 4

    def test_invalid_timing_reference_raises(self):
        """Invalid timing_reference raises ValueError."""
        volume_times = np.arange(10) * 0.1
        with pytest.raises(ValueError, match="Unknown timing_reference"):
            compute_processed_volume_times(
                volume_times,
                n_input_volumes=10,
                clutter_window_width=10,
                clutter_window_stride=10,
                inner_window_width=10,
                inner_window_stride=10,
                timing_reference="invalid",  # type: ignore
            )

    def test_matches_docstring_example(self):
        """Result matches the example from the docstring."""
        volume_times = np.arange(100) * 0.1
        result = compute_processed_volume_times(
            volume_times,
            n_input_volumes=100,
            clutter_window_width=50,
            clutter_window_stride=50,
            inner_window_width=50,
            inner_window_stride=50,
            timing_reference="center",
        )
        assert_allclose(result, [2.45, 7.45])


class TestComputePowerDopplerVolume:
    """Tests for compute_power_doppler_volume function."""

    def test_matches_reference_implementation(self, sample_iq_block_4d):
        """Result matches reference: mean(|filtered_block|^2)."""
        # No filtering (no cutoffs) so filtered_block = block.
        result = compute_power_doppler_volume(sample_iq_block_4d)

        expected = np.mean(np.abs(sample_iq_block_4d) ** 2, axis=0)
        # Result has shape (1, z, y, x) due to single window.
        assert_allclose(result[0], expected)

    def test_matches_reference_with_svd_filter(self, sample_iq_block_4d, spatial_mask):
        """Result with SVD filter matches reference implementation."""
        low_cutoff, high_cutoff = 2, 8
        time, z, y, x = sample_iq_block_4d.shape

        # Reference: apply SVD filter manually then compute power Doppler.
        signals = sample_iq_block_4d.reshape(time, -1)
        masked_signals = signals[:, spatial_mask.ravel()].astype(np.cdouble)
        u, _, _ = np.linalg.svd(masked_signals, full_matrices=False)
        clutter_vectors = np.concatenate(
            [u[:, :low_cutoff], u[:, high_cutoff:]], axis=1
        )
        filtered_signals = (
            signals - clutter_vectors @ clutter_vectors.conj().T @ signals
        )
        filtered_block = filtered_signals.reshape(sample_iq_block_4d.shape)
        expected = np.mean(np.abs(filtered_block) ** 2, axis=0)

        result = compute_power_doppler_volume(
            sample_iq_block_4d,
            filter_method="svd_indices",
            clutter_mask=spatial_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )

        assert_allclose(result[0], expected, rtol=1e-5, atol=1e-10)

    def test_butterworth_without_fs_raises(self, sample_iq_block_4d_long):
        """Butterworth filter without fs raises ValueError."""
        with pytest.raises(ValueError, match="sampling frequency must be provided"):
            compute_power_doppler_volume(
                sample_iq_block_4d_long,
                filter_method="butterworth",
                low_cutoff=5.0,
            )

    def test_invalid_filter_method_raises(self, sample_iq_block_4d):
        """Invalid filter_method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown clutter filter method"):
            compute_power_doppler_volume(
                sample_iq_block_4d,
                filter_method="invalid",  # type: ignore
            )

    def test_svd_indices_with_float_cutoffs_raises(self, sample_iq_block_4d):
        """SVD indices filter with float cutoffs raises ValueError."""
        with pytest.raises(ValueError, match="must both be integers"):
            compute_power_doppler_volume(
                sample_iq_block_4d,
                filter_method="svd_indices",
                low_cutoff=1.5,
            )


class TestComputeAxialVelocityVolume:
    """Tests for compute_axial_velocity_volume function."""

    def test_matches_reference_kasai_estimator(self, sample_iq_block_4d):
        """Result matches reference Kasai estimator implementation."""
        fs = 100.0
        ultrasound_frequency = 15.625e6
        sound_velocity = 1540.0
        lag = 1

        # Reference Kasai estimator (average_angle method).
        block_rolled_conjugate = np.roll(sample_iq_block_4d, lag, axis=0).conj()
        block_rolled_conjugate[:lag, ...] = 0
        autocorrelation = sample_iq_block_4d * block_rolled_conjugate
        autocorrelation = autocorrelation[lag:]
        autocorrelation_phase = np.angle(autocorrelation)
        average_phase = autocorrelation_phase.mean(0)
        expected = (
            average_phase * fs * sound_velocity / (4 * np.pi * ultrasound_frequency)
        )

        result = compute_axial_velocity_volume(
            sample_iq_block_4d,
            fs=fs,
            ultrasound_frequency=ultrasound_frequency,
            sound_velocity=sound_velocity,
            lag=lag,
            estimation_method="average_angle",
        )

        assert_allclose(result[0], expected, rtol=1e-5)

    def test_angle_average_method(self, sample_iq_block_4d):
        """Angle_average method computes angle of average autocorrelation."""
        fs = 100.0
        ultrasound_frequency = 15.625e6
        sound_velocity = 1540.0
        lag = 1

        # Reference: angle of average autocorrelation.
        block_rolled_conjugate = np.roll(sample_iq_block_4d, lag, axis=0).conj()
        block_rolled_conjugate[:lag, ...] = 0
        autocorrelation = sample_iq_block_4d * block_rolled_conjugate
        autocorrelation = autocorrelation[lag:]
        average_phase = np.angle(autocorrelation.mean(0))
        expected = (
            average_phase * fs * sound_velocity / (4 * np.pi * ultrasound_frequency)
        )

        result = compute_axial_velocity_volume(
            sample_iq_block_4d,
            fs=fs,
            ultrasound_frequency=ultrasound_frequency,
            sound_velocity=sound_velocity,
            lag=lag,
            estimation_method="angle_average",
        )

        assert_allclose(result[0], expected, rtol=1e-5)

    def test_invalid_estimation_method_raises(self, sample_iq_block_4d):
        """Invalid estimation_method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown estimation method"):
            compute_axial_velocity_volume(
                sample_iq_block_4d,
                fs=100.0,
                estimation_method="invalid",  # type: ignore
            )


class TestProcessIqBlocks:
    """Tests for process_iq_blocks function."""

    def test_stride_greater_than_width_raises(self, sample_iq_dataset):
        """window_stride > window_width raises ValueError."""
        iq = sample_iq_dataset["iq"]
        dask_iq = da.from_array(iq.data)

        with pytest.raises(ValueError, match="must be less than or equal"):
            process_iq_blocks(
                dask_iq,
                process_func=compute_power_doppler_volume,
                window_width=5,
                window_stride=10,
            )

    def test_warns_when_frames_dropped(self, sample_iq_dataset):
        """Warns when input volumes don't fit into complete windows."""
        iq = sample_iq_dataset["iq"]
        dask_iq = da.from_array(iq.data)

        with pytest.warns(UserWarning, match="input volumes will be dropped"):
            # 20 volumes, width=7, stride=7 -> 2 windows (14 used), 6 dropped.
            process_iq_blocks(
                dask_iq,
                process_func=compute_power_doppler_volume,
                window_width=7,
                window_stride=7,
            )


class TestProcessIqToPowerDoppler:
    """Tests for process_iq_to_power_doppler function."""

    def test_wrong_dimensions_raises(self, rng):
        """DataArray with wrong dimensions raises ValueError."""
        data = rng.random((10, 4, 6)) + 1j * rng.random((10, 4, 6))
        iq = xr.DataArray(
            data,
            dims=("time", "z", "y"),  # Missing x dimension.
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
            },
        )
        with pytest.raises(ValueError, match="Expected dimensions"):
            process_iq_to_power_doppler(iq)

    def test_non_complex_data_raises(self, rng):
        """Non-complex data raises TypeError."""
        data = rng.random((10, 4, 6, 8))  # Real, not complex.
        iq = xr.DataArray(
            data,
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
                "x": np.arange(8),
            },
        )
        with pytest.raises(TypeError, match="complex-valued"):
            process_iq_to_power_doppler(iq)

    def test_output_has_correct_attributes(self, sample_iq_dataset):
        """Output DataArray has expected attributes."""
        result = process_iq_to_power_doppler(
            sample_iq_dataset["iq"],
            clutter_window_width=10,
            clutter_window_stride=10,
            low_cutoff=1,
            high_cutoff=8,
        )

        assert result.name == "power_doppler"
        assert result.attrs["units"] == "a.u."
        assert result.attrs["clutter_filter_method"] == "svd_indices"
        assert result.attrs["clutter_window_width"] == 10
        assert result.attrs["clutter_low_cutoff"] == 1
        assert result.attrs["clutter_high_cutoff"] == 8

    def test_uses_compound_sampling_frequency_from_attrs(self, sample_iq_dataset):
        """Uses compound_sampling_frequency attribute when fs not provided."""
        # Update the attribute on the iq DataArray.
        sample_iq_dataset["iq"].attrs["compound_sampling_frequency"] = 100.0

        # Should not raise even with Butterworth (which requires fs).
        result = process_iq_to_power_doppler(
            sample_iq_dataset["iq"],
            clutter_window_width=20,
            filter_method="butterworth",
            low_cutoff=5.0,
        )
        assert result is not None


class TestProcessIqToAxialVelocity:
    """Tests for process_iq_to_axial_velocity function."""

    def test_missing_required_attributes_raises(self, sample_iq_dataset):
        """Raises ValueError when required attributes are missing."""
        # Remove required attribute from the iq DataArray.
        sample_iq_dataset["iq"].attrs.pop("compound_sampling_frequency", None)

        with pytest.raises(ValueError, match="Missing required DataArray attributes"):
            process_iq_to_axial_velocity(sample_iq_dataset["iq"])

    def test_output_has_correct_attributes(self, sample_iq_dataset):
        """Output DataArray has expected attributes."""
        result = process_iq_to_axial_velocity(
            sample_iq_dataset["iq"],
            clutter_window_width=10,
            clutter_window_stride=10,
            lag=2,
            absolute_velocity=True,
        )

        assert result.name == "axial_velocity"
        assert result.attrs["units"] == "m/s"
        assert result.attrs["lag"] == 2
        assert result.attrs["absolute_velocity"] is True

    def test_uses_attrs_for_parameters(self, sample_iq_dataset):
        """Uses DataArray attributes for fs, ultrasound_frequency and sound_velocity."""
        # Set specific attribute values on the iq DataArray to verify they're used.
        sample_iq_dataset["iq"].attrs["compound_sampling_frequency"] = 100.0
        sample_iq_dataset["iq"].attrs["transmit_frequency"] = 10e6
        sample_iq_dataset["iq"].attrs["sound_velocity"] = 1500.0

        result = process_iq_to_axial_velocity(sample_iq_dataset["iq"])

        assert result.attrs["ultrasound_frequency"] == 10e6
        assert result.attrs["sound_velocity"] == 1500.0


class TestDataArrayClutterMask:
    """Tests for DataArray clutter mask support in wrapper functions."""

    def test_dataarray_mask_matches_reference_power_doppler(
        self, sample_iq_dataset, spatial_mask
    ):
        """DataArray mask matches numpy reference for power Doppler."""
        iq = sample_iq_dataset["iq"]
        n_time = iq.sizes["time"]

        # Create DataArray mask with matching coordinates.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "z": iq.coords["z"],
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        result = process_iq_to_power_doppler(
            iq,
            clutter_window_width=n_time,
            clutter_window_stride=n_time,
            doppler_window_width=n_time,
            doppler_window_stride=n_time,
            clutter_mask=mask_dataarray,
            low_cutoff=2,
            high_cutoff=8,
        )

        expected = compute_power_doppler_volume(
            iq.values,
            filter_method="svd_indices",
            clutter_mask=mask_dataarray.values,
            low_cutoff=2,
            high_cutoff=8,
        )

        assert_allclose(result.values[0], expected[0])

    def test_dataarray_mask_matches_reference_axial_velocity(
        self, sample_iq_dataset, spatial_mask
    ):
        """DataArray mask matches numpy reference for axial velocity."""
        iq = sample_iq_dataset["iq"]
        n_time = iq.sizes["time"]

        # Create DataArray mask with matching coordinates.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "z": iq.coords["z"],
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        result = process_iq_to_axial_velocity(
            iq,
            clutter_window_width=n_time,
            clutter_window_stride=n_time,
            velocity_window_width=n_time,
            velocity_window_stride=n_time,
            clutter_mask=mask_dataarray,
            low_cutoff=2,
            high_cutoff=8,
        )

        expected = compute_axial_velocity_volume(
            iq.values,
            fs=iq.attrs["compound_sampling_frequency"],
            filter_method="svd_indices",
            clutter_mask=mask_dataarray.values,
            low_cutoff=2,
            high_cutoff=8,
            ultrasound_frequency=iq.attrs["transmit_frequency"],
            sound_velocity=iq.attrs["sound_velocity"],
        )

        assert_allclose(result.values[0], expected[0])

    def test_dataarray_mask_coordinate_mismatch_raises(
        self, sample_iq_dataset, spatial_mask
    ):
        """DataArray mask with mismatched coordinates raises ValueError."""
        iq = sample_iq_dataset["iq"]

        # Create mask with different coordinates.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "z": iq.coords["z"] + 1.0,  # Shifted coordinates.
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        with pytest.raises(ValueError, match="do not match data coordinates"):
            process_iq_to_power_doppler(
                iq,
                clutter_mask=mask_dataarray,
                low_cutoff=2,
                high_cutoff=8,
            )

    def test_dataarray_mask_missing_coordinate_raises(
        self, sample_iq_dataset, spatial_mask
    ):
        """DataArray mask missing a coordinate raises ValueError."""
        iq = sample_iq_dataset["iq"]

        # Create mask missing 'z' coordinate.
        mask_dataarray = xr.DataArray(
            spatial_mask,
            dims=("z", "y", "x"),
            coords={
                "y": iq.coords["y"],
                "x": iq.coords["x"],
            },
        )

        with pytest.raises(ValueError, match="missing coordinate"):
            process_iq_to_power_doppler(
                iq,
                clutter_mask=mask_dataarray,
                low_cutoff=2,
                high_cutoff=8,
            )

    def test_dataarray_mask_shape_mismatch_raises(self, sample_iq_dataset):
        """DataArray mask with wrong shape raises ValueError."""
        iq = sample_iq_dataset["iq"]

        # Create mask with wrong shape.
        wrong_mask = np.ones((2, 2, 2), dtype=bool)
        mask_dataarray = xr.DataArray(
            wrong_mask,
            dims=("z", "y", "x"),
            coords={
                "z": np.arange(2),
                "y": np.arange(2),
                "x": np.arange(2),
            },
        )

        with pytest.raises(ValueError, match="do not match data coordinates"):
            process_iq_to_power_doppler(
                iq,
                clutter_mask=mask_dataarray,
                low_cutoff=2,
                high_cutoff=8,
            )
