"""Detrending functions for signal preprocessing."""

import warnings
from typing import Literal

import numpy as np
import scipy.signal
import xarray as xr


def _polynomial_detrend_wrapper(data, axis, order):
    """Wrapper for polynomial detrending that works with xr.apply_ufunc."""
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    n_timepoints = data.shape[0]
    time_vals = np.arange(n_timepoints)

    if data.ndim > 2:
        original_shape = data.shape
        data_2d = data.reshape(n_timepoints, -1)
        poly_coeffs = np.polyfit(time_vals, data_2d, order)
        poly_trend = np.polyval(poly_coeffs, time_vals[:, np.newaxis])
        poly_trend = poly_trend.reshape(original_shape)
    else:
        poly_coeffs = np.polyfit(time_vals, data, order)
        poly_trend = np.polyval(poly_coeffs, time_vals[:, np.newaxis])

    result = data - poly_trend

    if axis != 0:
        result = np.moveaxis(result, 0, axis)

    return result


def detrend(
    signals: xr.DataArray,
    type: Literal["constant", "linear", "polynomial"] = "linear",
    order: int = 1,
) -> xr.DataArray:
    """Remove trends from signals across time.

    This function operates along the ``time`` dimension and works with arrays of any
    shape, making it flexible for both extracted signals and full fUSI data.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to detrend. Must have a ``time`` dimension. Can be any shape, e.g.,
        extracted signals ``(time, voxels)``, full 3D+t imaging data ``(time, z, y,
        x)``, or regional signals ``(time, regions)``.

        !!! warning "Chunking along time is not supported"
            The ``time`` dimension must NOT be chunked. Chunk only spatial dimensions:
            ``data.chunk({'time': -1})``.

    type : {"constant", "linear", "polynomial"}, default: "linear"
        Detrending method:

        - ``"constant"``: remove mean.
        - ``"linear"``: remove linear trend using least squares regression.
        - ``"polynomial"``: remove polynomial trend of specified order.

    order : int, default: 1
        Polynomial order. Only used when ``type="polynomial"``. For ``order=1``, this is
        equivalent to ``type="linear"``.

    Returns
    -------
    xarray.DataArray
        Detrended signals with same shape and coordinates as input.

    Raises
    ------
    ValueError
        If `type` is not recognized. If `signals` does not have a ``time`` dimension. If
        `order` is negative.

    Warns
    -----
    UserWarning
        If `signals` have only one timepoint (detrending is skipped).

    Notes
    -----
    - For ``"linear"`` and ``"constant"``, uses `scipy.signal.detrend`.
    - For ``"polynomial"``, fits a polynomial of the specified order and subtracts it.

    Examples
    --------
    Remove linear trend from extracted signals:

    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create signals with linear drift.
    >>> time = np.arange(100)
    >>> drift = time[:, None] * 0.5  # Linear drift.
    >>> noise = np.random.randn(100, 50)
    >>> signals = xr.DataArray(
    ...     drift + noise,
    ...     dims=["time", "voxels"]
    ... )
    >>> detrended = detrend(signals, type="linear")
    >>> # Drift is removed, only noise remains.

    Remove polynomial trend:

    >>> # Create signals with quadratic drift.
    >>> quadratic_drift = (time[:, None] ** 2) * 0.01
    >>> signals_quad = xr.DataArray(
    ...     quadratic_drift + noise,
    ...     dims=["time", "voxels"]
    ... )
    >>> detrended_quad = detrend(signals_quad, type="polynomial", order=2)

    Works on 3D+t data:

    >>> imaging_3dt = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"]
    ... )
    >>> detrended_3dt = detrend(imaging_3dt, type="linear")

    With Dask arrays (ensure time is not chunked):

    >>> import dask.array as da
    >>> # Create Dask array - chunk only spatial dimensions, NOT time!
    >>> dask_data = xr.DataArray(
    ...     da.from_array(np.random.randn(100, 50), chunks=(100, 25)),
    ...     dims=["time", "voxels"]
    ... )
    >>> detrended_dask = detrend(dask_data, type="linear")
    >>> # If your data is chunked along time, rechunk first:
    >>> # dask_data = dask_data.chunk({'time': -1})
    """
    if type not in {"linear", "constant", "polynomial"}:
        raise ValueError(
            f"type must be 'linear', 'constant', or 'polynomial', got '{type}'"
        )

    if "time" not in signals.dims:
        raise ValueError("signals must have a 'time' dimension")

    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")

    if signals.sizes["time"] == 1:
        warnings.warn(
            "Detrending of signals with only 1 timepoint would lead to "
            "zero or undefined values. Returning unchanged signals.",
        )
        return signals.copy()

    # Applying a ufunc with dask="parallelized" along a chunked dimension would apply
    # the detrending separately to each chunk (since we're not declaring time as a core
    # dimension to avoid transposing the data), which is not correct. Detrending
    # requires the full time series to fit the trend.
    if hasattr(signals.data, "chunks"):
        time_axis = signals.get_axis_num("time")
        time_chunks = signals.data.chunks[time_axis]
        if len(time_chunks) > 1:
            raise ValueError(
                f"Data is chunked along the 'time' dimension ({len(time_chunks)} "
                f"chunks), but detrending requires the full time series. "
                f"Rechunk your data so 'time' is not chunked: "
                f"data.chunk({{'time': -1}}) or use chunks=({signals.sizes['time']}, ...)"
            )

    time_axis = signals.get_axis_num("time")

    if type == "polynomial":
        result = xr.apply_ufunc(
            _polynomial_detrend_wrapper,
            signals,
            kwargs={"axis": time_axis, "order": order},
            dask="parallelized",
            output_dtypes=[signals.dtype],
        )
    else:
        result = xr.apply_ufunc(
            scipy.signal.detrend,
            signals,
            kwargs={"axis": time_axis, "type": type},
            dask="parallelized",
        )

    return result
