"""Tests for the signal.clean pipeline."""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from confusius.signal import clean, filter_butterworth


def test_clean_no_processing_returns_original(sample_timeseries):
    """Test clean returns input when no steps are requested."""
    signals = sample_timeseries()

    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        low_cutoff=None,
        high_cutoff=None,
    )

    assert_allclose(result.values, signals.values)
    assert result.dims == signals.dims
    assert_allclose(result.coords["time"].values, signals.coords["time"].values)


def test_clean_detrend_and_standardize(sample_timeseries):
    """Test clean detrends and standardizes signals."""
    signals = sample_timeseries()

    result = clean(
        signals,
        detrend_order=1,
        standardize_method="zscore",
        low_cutoff=None,
        high_cutoff=None,
    )

    mean_per_voxel = result.mean(dim="time")
    std_per_voxel = result.std(dim="time", ddof=1)

    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)
    assert_allclose(std_per_voxel.values, 1.0, rtol=1e-10)


def test_clean_with_confounds_reduces_correlation(sample_timeseries):
    """Test confound regression reduces correlation with confounds."""
    signals = sample_timeseries(n_time=200, n_voxels=3)
    time = np.arange(signals.sizes["time"]) / 100.0
    confound = np.sin(2 * np.pi * time)
    weights = np.array([2.0, -1.0, 0.5])
    signals = xr.DataArray(
        signals.values + confound[:, None] * weights[None, :],
        dims=signals.dims,
        coords=signals.coords,
    )

    before = np.corrcoef(confound, signals.values[:, 0])[0, 1]

    confounds = xr.DataArray(
        confound, dims=["time"], coords={"time": signals.coords["time"]}
    )
    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        confounds=confounds,
    )

    after = np.corrcoef(confound, result.values[:, 0])[0, 1]

    assert abs(after) < abs(before) * 1e-2


def test_clean_scrub_censors_after_filter(sample_timeseries):
    """Test scrubbing interpolates then censors samples when filtering."""
    signals = sample_timeseries(n_time=100, sampling_rate=100.0)
    mask_values = np.ones(100, dtype=bool)
    mask_values[[10, 25, 60]] = False
    sample_mask = xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )

    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        high_cutoff=5.0,
        sample_mask=sample_mask,
    )

    assert result.sizes["time"] == np.sum(mask_values)


def test_clean_censors_first_without_filter_or_detrend(sample_timeseries):
    """Test censoring occurs immediately when no detrend/filter requested."""
    signals = sample_timeseries(n_time=100, sampling_rate=100.0)
    mask_values = np.ones(100, dtype=bool)
    mask_values[[10, 25, 60]] = False
    sample_mask = xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )

    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        sample_mask=sample_mask,
    )

    assert result.sizes["time"] == np.sum(mask_values)


def test_clean_filter_low_pass_matches_filter_butterworth(sample_timeseries):
    """Test low_pass matches high_cutoff argument."""
    signals = sample_timeseries(n_time=200, sampling_rate=100.0)

    expected = filter_butterworth(signals, high_cutoff=5.0)
    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        high_cutoff=5.0,
    )

    assert_allclose(result.values, expected.values)
