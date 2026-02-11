"""Image visualization utilities for fUSI data."""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from napari import Viewer
    from napari.layers import Image

# Import napari at module level for backward compatibility and test mocking.
# This is a lightweight import (~111ms) compared to matplotlib/scipy.
import napari  # noqa: F401


def plot_napari(
    data: xr.DataArray,
    scale_method: Literal["db", "log", "power"] | None = "db",
    scale_kwargs: dict[str, Any] | None = None,
    show_colorbar: bool = True,
    show_scale_bar: bool = True,
    dim_order: tuple[str, ...] = ("z", "y", "x"),
    **imshow_kwargs,
) -> tuple["Viewer", list["Image"]]:
    """Display fUSI data using the Napari viewer.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to visualize. Expected dimensions are (time, z, y, x) where
        z is the elevation/stacking axis, y is depth, and x is lateral. Use
        ``dim_order`` to specify a different dimension ordering.
    scale_method : {"db", "log", "power", None}, default: "db"
        Scaling method to apply before display. Use ``"db"`` for decibel scaling,
        ``"log"`` for natural log, ``"power"`` for power scaling, or ``None`` to display
        without scaling.
    scale_kwargs : dict, optional
        Keyword arguments to pass to the scaling method. For example,
        ``{"factor": 20}`` for db scaling or ``{"exponent": 0.5}`` for power
        scaling.
    show_colorbar : bool, default: True
        Whether to show the colorbar.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    dim_order : tuple of str, default: ("z", "y", "x")
        Dimension ordering for spatial axes. Default is ``("z", "y", "x")``
        (elevation, depth, lateral). Use this to override for data with
        different conventions, e.g., ``("y", "z", "x")`` or ``("x", "y", "z")``.
    **imshow_kwargs
        Additional keyword arguments passed to `napari.imshow`, such as
        ``contrast_limits``, ``colormap``, etc.

    Returns
    -------
    viewer : napari.Viewer
        The napari viewer instance.
    layer : napari.layers.Image
        The image layer added to the viewer.

    Notes
    -----
    - If the data has a ``voxdim`` attribute, it will be used as the scale parameter
      for napari to ensure correct spatial scaling. The voxdim should be ordered as
      ``[z, y, x]`` (elevation, depth, lateral) by default.
    - Future versions may support an ``orientation`` attribute (like BrainGlobe's
      "asl" convention) to automatically determine dimension ordering.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_napari
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> viewer, layer = plot_napari(data)

    >>> # Custom contrast limits
    >>> viewer, layer = plot_napari(data, contrast_limits=(-15, 0))

    >>> # Amplitude scaling (factor=20)
    >>> viewer, layer = plot_napari(
    ...     data, scale_method="db", scale_kwargs={"factor": 20}
    ... )

    >>> # No scaling
    >>> viewer, layer = plot_napari(data, scale_method=None)

    >>> # Different dimension ordering (e.g., depth, elevation, lateral)
    >>> viewer, layer = plot_napari(data, dim_order=("y", "z", "x"))
    """
    if scale_method is not None:
        if scale_kwargs is None:
            scale_kwargs = {}

        if scale_method == "db":
            data = data.fusi.scale.db(**scale_kwargs)
        elif scale_method == "log":
            data = data.fusi.scale.log(**scale_kwargs)
        elif scale_method == "power":
            data = data.fusi.scale.power(**scale_kwargs)
        else:
            raise ValueError(
                f"Unknown scale_method: {scale_method}. "
                "Use 'db', 'log', 'power', or None."
            )

    # Determine dimension ordering for napari display.
    # Default is (time, z, y, x) where z=elevation, y=depth, x=lateral.
    # Napari displays spatial dimensions, so we map accordingly.

    # Get current dimension names
    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    # Validate dim_order matches spatial dimensions
    if set(dim_order) != set(spatial_dims):
        raise ValueError(
            f"dim_order {dim_order} does not match spatial dimensions {spatial_dims}. "
            "Ensure dim_order contains all spatial dimension names."
        )

    # Build scale parameter: voxdim corresponds to spatial dimensions in data order.
    # Scale should be in the order dimensions appear in the data array.
    scale = None
    if "voxdim" in data.attrs:
        voxdim = data.attrs["voxdim"]
        if len(voxdim) == len(spatial_dims):
            # voxdim is in the order of spatial_dims as they appear in the data.
            # Scale must match the data dimension order.
            scale = list(voxdim)

    # Build order parameter for napari.imshow.
    # This specifies which dimensions are displayed and in what order.
    # The last 2 (2D) or 3 (3D) dimensions are the displayed spatial axes.
    order = []
    if time_dim in all_dims:
        order.append(all_dims.index(time_dim))
    for dim in dim_order:
        if dim in all_dims:
            order.append(all_dims.index(dim))

    viewer, image_layer = napari.imshow(
        data,
        scale=scale,
        order=order,
        **imshow_kwargs,
    )

    if show_colorbar:
        viewer.layers[0].colorbar.visible = True  # type: ignore[attr-defined]

    if show_scale_bar:
        viewer.scale_bar.visible = True
        viewer.scale_bar.gridded = True
        viewer.scale_bar.unit = "mm"

    return viewer, image_layer


def plot_carpet(
    data: xr.DataArray,
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
    data : xarray.DataArray
        Input data array with a 'time' dimension.
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
        Scaling method to apply before processing. Use ``"db"`` for decibel scaling,
        ``"log"`` for natural log, ``"power"`` for power scaling. If not provided, no
        scaling is applied.
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
    Complex-valued data is converted to magnitude before processing, and time axis
    labels are derived from the 'time' coordinate.

    Carpet plots were originally introduced by :cite:t:`Smyser2011` and
    popularized by :cite:t:`Power2012`. This function was inspired by Nilearn's
    `nilearn.plotting.plot_carpet`.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_carpet
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> fig, ax = plot_carpet(data)

    >>> # Without detrending
    >>> fig, ax = plot_carpet(data, detrend=False)

    >>> # With mask (xarray)
    >>> mask = data.isel(time=0).pipe(np.abs) > threshold
    >>> fig, ax = plot_carpet(data, mask=mask)

    >>> # With mask (numpy array)
    >>> import numpy as np
    >>> mask_array = np.ones(data.shape[1:], dtype=bool)  # (z, y, x)
    >>> mask_array[:10, :, :] = False  # Exclude first 10 z slices
    >>> fig, ax = plot_carpet(data, mask=mask_array)
    """
    import matplotlib.pyplot as plt
    from scipy import signal

    if np.iscomplexobj(data.values):
        data = xr.ufuncs.abs(data)

    if "time" not in data.dims or "time" not in data.coords:
        raise ValueError("Data must have 'time' dimension and coordinates.")

    n_timepoints = data.sizes["time"]

    spatial_dims = [d for d in data.dims if d != "time"]
    # Creating a multi-index of spatial dimensions would make Xarray's plotting
    # functions fail.
    signals = data.stack(voxels=spatial_dims, create_index=False)

    if mask is not None:
        if not isinstance(mask, xr.DataArray):
            mask = xr.DataArray(
                np.asarray(mask),
                dims=spatial_dims,
            )

        mask_spatial_dims = list(mask.dims)
        if set(mask_spatial_dims) != set(spatial_dims):
            msg = (
                f"Mask dimensions {mask_spatial_dims} do not match "
                f"data spatial dimensions {spatial_dims}."
            )
            raise ValueError(msg)

        mask_1d = mask.stack(voxels=spatial_dims)

        mask_1d = mask_1d.reindex_like(signals.isel(time=0))

        mask_bool = mask_1d.values != 0

        signals = signals.isel(voxels=mask_bool)
    else:
        non_zero_voxels = (signals != 0).any(dim="time")
        signals = signals.isel(voxels=non_zero_voxels.values)

    if detrend:
        signals = xr.apply_ufunc(
            signal.detrend,
            signals,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
        )

    if standardize:
        mean = signals.mean(dim="time")
        std = signals.std(dim="time")
        # Avoid division by zero.
        std = std.where(std != 0, other=1)
        signals = (signals - mean) / std
        signals.attrs["units"] = "z-score"

    if vmin is None or vmax is None:
        std_val = float(signals.std(axis=0).mean().values)
        default_vmin = float(signals.mean().values - (2 * std_val))
        default_vmax = float(signals.mean().values + (2 * std_val))
        vmin = vmin or default_vmin
        vmax = vmax or default_vmax

    if decimation_threshold is not None and n_timepoints > decimation_threshold:
        # For decimation, we get the smallest power of 2 greater than the number of
        # volumes divided by the threshold.
        n_decimations = int(
            np.ceil(np.log2(np.ceil(n_timepoints / decimation_threshold)))
        )
        decimation_factor = 2**n_decimations
        signals = signals[::decimation_factor, :]

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    signals.plot(cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, yincrease=False)

    ax.grid(False)
    ax.set_yticks([])
    ax.set_ylabel("Voxels")
    # TODO: we could get time units from attrs, or maybe use pint?
    ax.set_xlabel("Time (s)")

    if title:
        ax.set_title(title)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))

    return figure, ax
