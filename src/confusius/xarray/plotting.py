"""Plotting operations and accessor for fUSI data."""

from typing import TYPE_CHECKING, Any, Literal

import xarray as xr
from numpy.typing import ArrayLike

from confusius.plotting import plot_carpet, plot_napari

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from napari import Viewer
    from napari.layers import Image


class FUSIPlotAccessor:
    """Accessor for plotting fUSI data.

    This accessor provides convenient plotting methods for functional
    ultrasound imaging data, with specialized support for napari visualization.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> viewer, layer = data.fusi.plot.napari()
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def __call__(self, **kwargs) -> tuple[Any, Any]:
        """Call the napari plotting method by default.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `napari`.
        """
        return self.napari(**kwargs)

    def napari(
        self,
        scale_method: Literal["db", "log", "power"] | None = "db",
        scale_kwargs: dict[str, Any] | None = None,
        show_colorbar: bool = True,
        show_scale_bar: bool = True,
        **imshow_kwargs,
    ) -> "tuple[Viewer, list[Image]]":
        """Display data in napari viewer.

        Parameters
        ----------
        scale_method : {"db", "log", "power", None}, default: "db"
            Scaling method to apply before display. Use ``"db"`` for decibel scaling,
            ``"log"`` for natural log, ``"power"`` for power scaling, or ``None`` to
            display without scaling.
        scale_kwargs : dict, optional
            Keyword arguments to pass to the scaling method. For example,
            ``{"factor": 20}`` for db scaling or ``{"exponent": 0.5}`` for power
            scaling.
        show_colorbar : bool, default: True
            Whether to show the colorbar.
        show_scale_bar : bool, default: True
            Whether to show the scale bar.
        **imshow_kwargs
            Additional keyword arguments passed to `napari.imshow`, such as
            ``contrast_limits``, ``colormap``, etc.

        Returns
        -------
        viewer : napari.Viewer
            The napari viewer instance.
        layer : napari.layers.Image
            The image layer added to the viewer.

        Examples
        --------
        >>> # Default: decibel scaling
        >>> data = xr.open_zarr("output.zarr")["iq"]
        >>> viewer, layer = data.fusi.plot.napari()

        >>> # Custom contrast limits
        >>> viewer, layer = data.fusi.plot.napari(contrast_limits=(-15, 0))

        >>> # Amplitude scaling (factor=20)
        >>> viewer, layer = data.fusi.plot.napari(
        ...     scale_method="db", scale_kwargs={"factor": 20}
        ... )

        >>> # No scaling
        >>> viewer, layer = data.fusi.plot.napari(scale_method=None)
        """
        return plot_napari(
            self._obj,
            scale_method=scale_method,
            scale_kwargs=scale_kwargs,
            show_colorbar=show_colorbar,
            show_scale_bar=show_scale_bar,
            **imshow_kwargs,
        )

    def carpet(
        self,
        mask: xr.DataArray | ArrayLike | None = None,
        detrend: bool = True,
        standardize: bool = True,
        scale_method: Literal["db", "log", "power"] | None = None,
        scale_kwargs: dict[str, Any] | None = None,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        decimation_threshold: int | None = 800,
        figsize: tuple[float, float] = (10, 5),
        title: str | None = None,
        ax: "Axes | None" = None,
    ) -> tuple["Figure | SubFigure", "Axes"]:
        """Plot voxel intensities across time as a raster image.

        A carpet plot (also known as "grayplot" or "Power plot") displays voxel
        intensities as a 2D raster image with time on the x-axis and voxels on
        the y-axis. Each row represents one voxel's time series, typically
        standardized to z-scores.

        Parameters
        ----------
        mask : array_like, optional
            Mask with same spatial dimensions as data ``(z, y, x)``. Non-zero values
            in `mask` indicate voxels to include. If not provided, all non-zero voxels
            from the data are included.
        detrend : bool, default: True
            Whether to remove linear trend from each voxel's time series.
        standardize : bool, default: True
            Whether to standardize each voxel's time series to z-scores
            (zero mean, unit variance).
        scale_method : {"db", "log", "power"}, optional
            Scaling method to apply before processing. Use ``"db"`` for decibel
            scaling, ``"log"`` for natural log, ``"power"`` for power scaling.
            If not provided, no scaling is applied.
        scale_kwargs : dict, optional
            Keyword arguments to pass to the scaling method. For example,
            ``{"factor": 20}`` for db scaling or ``{"exponent": 0.5}`` for power
            scaling.
        cmap : str, default: "gray"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for colormap. If not provided, uses ``mean - 2*std``.
        vmax : float, optional
            Maximum value for colormap. If not provided, uses ``mean + 2*std``.
        decimation_threshold : int or None, default: 800
            If the number of timepoints exceeds this value, data is downsampled
            along the time axis to improve plotting performance. Set to ``None`` to
            disable downsampling.
        figsize : tuple[float, float], default: (10, 5)
            Figure size in inches ``(width, height)``.
        title : str, optional
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If not provided, creates new figure and axes.

        Returns
        -------
        figure : matplotlib.figure.Figure or matplotlib.figure.SubFigure
            Figure object containing the carpet plot.
        axes : matplotlib.axes.Axes
            Axes object with the carpet plot.

        Notes
        -----
        Carpet plots were originally introduced by :cite:t:`Smyser2011` and
        popularized by :cite:t:`Power2012`. This function was inspired by Nilearn's
        `nilearn.plotting.plot_carpet`.

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["iq"]
        >>> fig, ax = data.fusi.plot.carpet()

        >>> # Without detrending
        >>> fig, ax = data.fusi.plot.carpet(detrend=False)

        >>> # With mask (xarray)
        >>> mask = data.isel(time=0).pipe(np.abs) > threshold
        >>> fig, ax = data.fusi.plot.carpet(mask=mask)

        >>> # With mask (numpy array)
        >>> import numpy as np
        >>> mask_array = np.ones(data.shape[1:], dtype=bool)  # (z, y, x)
        >>> mask_array[:10, :, :] = False  # Exclude first 10 z slices
        >>> fig, ax = data.fusi.plot.carpet(mask=mask_array)
        """
        return plot_carpet(
            self._obj,
            mask=mask,
            detrend=detrend,
            standardize=standardize,
            scale_method=scale_method,
            scale_kwargs=scale_kwargs,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            decimation_threshold=decimation_threshold,
            figsize=figsize,
            title=title,
            ax=ax,
        )
