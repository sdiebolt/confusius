"""Tests for confusius.decomposition.PCA."""

import numpy as np
import pytest
import xarray as xr
from sklearn.utils.validation import check_is_fitted

from confusius.decomposition import PCA


@pytest.fixture
def sample_data(rng):
    """Create a small reproducible `(time, y, x)` test DataArray."""
    return xr.DataArray(
        rng.standard_normal((30, 4, 5)),
        dims=["time", "y", "x"],
        coords={
            "time": np.linspace(0.0, 2.9, 30),
            "y": np.linspace(-1.0, 1.0, 4),
            "x": np.linspace(0.0, 2.0, 5),
        },
        name="power_doppler",
        attrs={"units": "a.u."},
    )


def test_fit_transform_returns_dataarray(sample_data):
    """fit_transform returns `(time, component)` DataArray with coords."""
    model = PCA(n_components=6, random_state=0)

    signals = model.fit_transform(sample_data)

    assert isinstance(signals, xr.DataArray)
    assert signals.dims == ("time", "component")
    assert signals.shape == (sample_data.sizes["time"], 6)
    np.testing.assert_allclose(signals.coords["time"], sample_data.coords["time"])
    np.testing.assert_array_equal(signals.coords["component"], np.arange(6))

    assert model.components_.dims == ("component", "y", "x")
    assert model.components_.shape == (6, 4, 5)


def test_inverse_transform_reconstructs_with_all_components(sample_data):
    """Using all components reconstructs the original data."""
    model = PCA(random_state=0)

    signals = model.fit_transform(sample_data)
    reconstructed = model.inverse_transform(signals)

    assert reconstructed.dims == sample_data.dims
    np.testing.assert_allclose(reconstructed.coords["time"], sample_data.coords["time"])
    np.testing.assert_allclose(reconstructed.values, sample_data.values, atol=1e-10)
    assert reconstructed.name == sample_data.name
    assert reconstructed.attrs == sample_data.attrs


def test_inverse_transform_from_numpy_returns_dataarray(sample_data):
    """inverse_transform accepts ndarray input and returns DataArray."""
    model = PCA(n_components=5, random_state=0)
    scores = model.fit_transform(sample_data).values

    reconstructed = model.inverse_transform(scores)

    assert isinstance(reconstructed, xr.DataArray)
    assert reconstructed.dims == sample_data.dims
    np.testing.assert_array_equal(
        reconstructed.coords["time"], np.arange(sample_data.sizes["time"])
    )


def test_fit_requires_time_dimension(sample_data):
    """fit raises when the input has no `time` dimension."""
    no_time = sample_data.isel(time=0, drop=True)

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        PCA().fit(no_time)


def test_transform_checks_spatial_layout(sample_data):
    """transform raises if spatial layout differs from fit."""
    model = PCA(n_components=4, random_state=0).fit(sample_data)
    bad = sample_data.isel(x=slice(0, 4))

    with pytest.raises(ValueError, match="Spatial dimension 'x' has size"):
        model.transform(bad)


def test_sklearn_interface_fitted_state(sample_data):
    """Estimator exposes sklearn fitted-state behavior."""
    model = PCA(n_components=3, random_state=0)
    with pytest.raises(Exception):
        check_is_fitted(model)

    check_is_fitted(model.fit(sample_data))
