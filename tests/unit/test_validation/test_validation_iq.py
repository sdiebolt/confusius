"""Tests for IQ data validation utilities."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_iq


class TestValidateIq:
    """Tests for `validate_iq` function."""

    @pytest.fixture
    def valid_iq_dataarray(self):
        """Create a valid IQ DataArray with all required attributes."""
        return xr.DataArray(
            np.ones((10, 4, 6, 8), dtype=np.complex64),
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
                "x": np.arange(8),
            },
            attrs={
                "compound_sampling_frequency": 1000.0,
                "transmit_frequency": 15.625e6,
                "sound_velocity": 1540.0,
            },
        )

    def test_valid_dataarray_returns_dataarray(self, valid_iq_dataarray):
        """Valid DataArray returns the same DataArray."""
        result = validate_iq(valid_iq_dataarray)

        assert isinstance(result, xr.DataArray)
        assert result.shape == (10, 4, 6, 8)

    def test_wrong_dimensions_raises(self, valid_iq_dataarray):
        """DataArray with wrong dimensions raises `ValueError`."""
        iq = valid_iq_dataarray.rename({"time": "t"})

        with pytest.raises(ValueError, match="Expected dimensions"):
            validate_iq(iq)

    def test_missing_coordinates_raises(self):
        """Missing required coordinates raises `ValueError`."""
        iq = xr.DataArray(
            np.ones((10, 4, 6, 8), dtype=np.complex64),
            dims=("time", "z", "y", "x"),
            # Missing 'x' coordinate
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
            },
            attrs={
                "compound_sampling_frequency": 1000.0,
                "transmit_frequency": 15.625e6,
                "sound_velocity": 1540.0,
            },
        )

        with pytest.raises(ValueError, match="Missing required coordinates"):
            validate_iq(iq)

    def test_non_complex_data_raises(self, valid_iq_dataarray):
        """Non-complex IQ data raises TypeError."""
        iq = valid_iq_dataarray.real

        with pytest.raises(TypeError, match="Expected complex-valued data"):
            validate_iq(iq)

    @pytest.mark.parametrize(
        "missing_attr",
        [
            "compound_sampling_frequency",
            "transmit_frequency",
            "sound_velocity",
        ],
    )
    def test_missing_required_attribute_raises(self, valid_iq_dataarray, missing_attr):
        """Missing any required attribute raises `ValueError`."""
        iq = valid_iq_dataarray.copy()
        del iq.attrs[missing_attr]

        with pytest.raises(ValueError, match="Missing required DataArray attributes"):
            validate_iq(iq)

    def test_require_attrs_false_skips_attribute_validation(self, valid_iq_dataarray):
        """require_attrs=False skips attribute validation."""
        iq = valid_iq_dataarray.copy()
        del iq.attrs["compound_sampling_frequency"]

        # Should not raise when require_attrs=False.
        result = validate_iq(iq, require_attrs=False)
        assert result is not None

    def test_multiple_missing_attributes_in_error_message(self, valid_iq_dataarray):
        """Error message lists all missing attributes."""
        iq = valid_iq_dataarray.copy()
        del iq.attrs["compound_sampling_frequency"]
        del iq.attrs["transmit_frequency"]

        with pytest.raises(ValueError) as exc_info:
            validate_iq(iq)

        error_msg = str(exc_info.value)
        assert "compound_sampling_frequency" in error_msg
        assert "transmit_frequency" in error_msg
