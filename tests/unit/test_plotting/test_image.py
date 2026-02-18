"""Unit tests for confusius.plotting.image module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from confusius.plotting.image import plot_carpet, plot_napari


class TestPlotNapari:
    """Tests for plot_napari function."""

    @pytest.fixture
    def sample_data_4d(self):
        """Create 4D sample DataArray with time dimension.

        Uses ConfUSIus convention: (time, z, y, x) where
        z=elevation, y=depth, x=lateral.
        """
        return xr.DataArray(
            np.random.rand(10, 20, 64, 128),
            dims=["time", "z", "y", "x"],
            coords={
                "time": np.arange(10),
                "z": np.linspace(0, 4, 20),  # elevation/stacking
                "y": np.linspace(0, 20, 64),  # depth
                "x": np.linspace(0, 10, 128),  # lateral
            },
            attrs={"voxdim": [0.2, 0.3175, 0.0787]},  # [z, y, x] spacing
        )

    @pytest.fixture
    def sample_data_3d(self):
        """Create 3D sample DataArray without time dimension.

        Uses ConfUSIus convention: (z, y, x) where
        z=elevation, y=depth, x=lateral.
        """
        return xr.DataArray(
            np.random.rand(20, 64, 128),
            dims=["z", "y", "x"],
            coords={
                "z": np.linspace(0, 4, 20),  # elevation/stacking
                "y": np.linspace(0, 20, 64),  # depth
                "x": np.linspace(0, 10, 128),  # lateral
            },
            attrs={"voxdim": [0.2, 0.3175, 0.0787]},  # [z, y, x] spacing
        )

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_with_defaults_4d(self, mock_napari, sample_data_4d):
        """plot_napari works with default parameters on 4D data."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        viewer, layer = plot_napari(sample_data_4d)

        # Check napari.imshow was called.
        assert mock_napari.imshow.called
        call_args = mock_napari.imshow.call_args

        # Check scale parameter matches voxdim [z, y, x] by default.
        assert call_args[1]["scale"] == [0.2, 0.3175, 0.0787]

        assert viewer is mock_viewer
        assert layer is mock_layer

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_with_defaults_3d(self, mock_napari, sample_data_3d):
        """plot_napari works with default parameters on 3D data."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        viewer, layer = plot_napari(sample_data_3d)

        # Check napari.imshow was called.
        assert mock_napari.imshow.called

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_applies_db_scaling_by_default(
        self, mock_napari, sample_data_4d
    ):
        """plot_napari applies db scaling by default."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        plot_napari(sample_data_4d)

        # Get the data that was passed to imshow.
        call_args = mock_napari.imshow.call_args
        passed_data = call_args[0][0]

        # Check that scaling was applied (should have dB units attribute).
        assert passed_data.attrs.get("units") == "dB"

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_with_custom_scale_method(self, mock_napari, sample_data_4d):
        """plot_napari accepts custom scaling method."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        plot_napari(sample_data_4d, scale_method="log")

        call_args = mock_napari.imshow.call_args
        passed_data = call_args[0][0]

        # Check that log scaling was applied.
        assert passed_data.attrs.get("scaling") == "log(x)"

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_with_scale_kwargs(self, mock_napari, sample_data_4d):
        """plot_napari accepts scale_kwargs."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        plot_napari(sample_data_4d, scale_method="db", scale_kwargs={"factor": 20})

        call_args = mock_napari.imshow.call_args
        passed_data = call_args[0][0]

        # Check that custom factor was used (check scaling attribute).
        assert "20*log10" in passed_data.attrs.get("scaling", "")

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_without_scaling(self, mock_napari, sample_data_4d):
        """plot_napari can display without scaling."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        plot_napari(sample_data_4d, scale_method=None)

        call_args = mock_napari.imshow.call_args
        passed_data = call_args[0][0]

        # Original data should not have scaling attributes.
        assert "units" not in passed_data.attrs or passed_data.attrs["units"] != "dB"

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_passes_imshow_kwargs(self, mock_napari, sample_data_4d):
        """plot_napari passes kwargs to imshow."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        plot_napari(sample_data_4d, contrast_limits=(-15, 0), colormap="viridis")

        call_args = mock_napari.imshow.call_args

        # Check that kwargs were passed through.
        assert call_args[1]["contrast_limits"] == (-15, 0)
        assert call_args[1]["colormap"] == "viridis"

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_without_voxdim(self, mock_napari):
        """plot_napari works without voxdim attribute."""
        data = xr.DataArray(
            np.random.rand(10, 20, 64, 128),
            dims=["time", "z", "y", "x"],
        )

        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        viewer, layer = plot_napari(data)

        # Should work, scale should be None.
        call_args = mock_napari.imshow.call_args
        assert call_args[1]["scale"] is None

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_with_custom_dim_order(self, mock_napari, sample_data_4d):
        """plot_napari accepts custom dim_order parameter."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        # Test with different dimension ordering (y, z, x)
        plot_napari(sample_data_4d, dim_order=("y", "z", "x"))

        call_args = mock_napari.imshow.call_args
        # voxdim stays in data dimension order [z, y, x], not reordered.
        assert call_args[1]["scale"] == [0.2, 0.3175, 0.0787]
        # Order should be (time, y, z, x) = (0, 2, 1, 3)
        assert call_args[1]["order"] == [0, 2, 1, 3]

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_invalid_dim_order(self, mock_napari, sample_data_4d):
        """plot_napari raises ValueError for invalid dim_order."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        with pytest.raises(ValueError, match="dim_order"):
            plot_napari(sample_data_4d, dim_order=("a", "b", "c"))

    @patch("confusius.plotting.image.napari")
    def test_plot_napari_invalid_scale_method(self, mock_napari, sample_data_4d):
        """plot_napari raises ValueError for invalid scale_method."""
        mock_viewer = MagicMock()
        mock_layer = MagicMock()
        mock_napari.imshow.return_value = (mock_viewer, mock_layer)

        with pytest.raises(ValueError, match="Unknown scale_method"):
            plot_napari(sample_data_4d, scale_method="invalid")


class TestPlotCarpet:
    """Tests for plot_carpet function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample 4D DataArray for carpet plot testing."""
        np.random.seed(42)
        return xr.DataArray(
            np.random.randn(100, 20, 1, 30) + 1j * np.random.randn(100, 20, 1, 30),
            dims=["time", "z", "y", "x"],
            coords={
                "time": np.linspace(0, 10, 100),
                "z": np.linspace(0, 5, 20),
                "y": [0.0],
                "x": np.linspace(0, 10, 30),
            },
        )

    @pytest.fixture
    def sample_mask_xarray(self):
        """Create sample xarray mask."""
        mask = xr.DataArray(
            np.ones((20, 1, 30), dtype=bool),
            dims=["z", "y", "x"],
            coords={
                "z": np.linspace(0, 5, 20),
                "y": [0.0],
                "x": np.linspace(0, 10, 30),
            },
        )
        mask[:5, :, :] = False  # Exclude first 5 z slices.
        return mask

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_basic(self, mock_subplots, sample_data):
        """plot_carpet works with basic parameters."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data)

        assert fig is mock_fig
        assert ax is mock_ax
        mock_subplots.assert_called_once()

    def test_plot_carpet_missing_time_dimension(self):
        """plot_carpet raises error when time dimension is missing."""
        # Create 3D data without time dimension.
        data_3d = xr.DataArray(
            np.random.rand(20, 1, 30),
            dims=["z", "y", "x"],
        )
        with pytest.raises(ValueError, match="time"):
            plot_carpet(data_3d)

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_xarray_mask(
        self, mock_subplots, sample_data, sample_mask_xarray
    ):
        """plot_carpet works with xarray mask."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, mask=sample_mask_xarray)

        assert fig is mock_fig
        assert ax is mock_ax

    def test_plot_carpet_mask_wrong_dimensions(self, sample_data):
        """plot_carpet raises error when mask dimensions don't match."""
        wrong_mask = xr.DataArray(
            np.ones((10, 10), dtype=bool),
            dims=["a", "b"],
        )

        with pytest.raises(ValueError, match="dimensions"):
            plot_carpet(sample_data, mask=wrong_mask)

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_detrend(self, mock_subplots, sample_data):
        """plot_carpet works with detrend_order enabled."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, detrend_order=1)

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_without_detrend(self, mock_subplots, sample_data):
        """plot_carpet works with detrend_order disabled."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, detrend_order=None)

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_standardize(self, mock_subplots, sample_data):
        """plot_carpet works with standardize enabled."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, standardize=True)

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_without_standardize(self, mock_subplots, sample_data):
        """plot_carpet works with standardize disabled."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, standardize=False)

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_custom_vmin_vmax(self, mock_subplots, sample_data):
        """plot_carpet accepts custom vmin and vmax."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, vmin=-3, vmax=3)

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_custom_cmap(self, mock_subplots, sample_data):
        """plot_carpet accepts custom colormap."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, cmap="viridis")

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_title(self, mock_subplots, sample_data):
        """plot_carpet accepts custom title."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, title="Test Carpet Plot")

        assert fig is mock_fig
        # Check that set_title was called with the correct title.
        assert any(
            call[0][0] == "Test Carpet Plot"
            for call in mock_ax.set_title.call_args_list
        )

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_existing_axes(self, mock_subplots, sample_data):
        """plot_carpet can use existing axes."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.figure = mock_fig

        fig, ax = plot_carpet(sample_data, ax=mock_ax)

        assert fig is mock_fig
        assert ax is mock_ax
        # Should not create new figure when ax is provided.
        mock_subplots.assert_not_called()

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_handles_complex_data(self, mock_subplots, sample_data):
        """plot_carpet converts complex data to magnitude."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # sample_data is complex by default in fixture.
        fig, ax = plot_carpet(sample_data)

        assert fig is mock_fig
        # Should not raise error with complex data.

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_decimation_threshold(self, mock_subplots, sample_data):
        """plot_carpet respects decimation threshold."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, decimation_threshold=50)

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_without_decimation(self, mock_subplots, sample_data):
        """plot_carpet works without decimation."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = plot_carpet(sample_data, decimation_threshold=None)

        assert fig is mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_carpet_with_dask_array(self, mock_subplots):
        """`plot_carpet` works with Dask-backed arrays without explicit compute."""
        import dask.array as da

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Create a Dask-backed DataArray (simulating lazy-loaded data).
        np.random.seed(42)
        dask_array = da.from_array(
            np.random.randn(100, 20, 1, 30) + 1j * np.random.randn(100, 20, 1, 30),
            # Keep time axis unchunked for detrend/standardize operations.
            chunks=(100, 10, 1, 15),
        )
        dask_data = xr.DataArray(
            dask_array,
            dims=["time", "z", "y", "x"],
            coords={
                "time": np.linspace(0, 10, 100),
                "z": np.linspace(0, 5, 20),
                "y": [0.0],
                "x": np.linspace(0, 10, 30),
            },
        )

        # This should not raise NotImplementedError about .item() on Dask arrays.
        fig, ax = plot_carpet(dask_data)

        assert fig is mock_fig
