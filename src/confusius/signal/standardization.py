"""Standardization functions for signal preprocessing."""

import warnings
from typing import Literal

import numpy as np
import xarray as xr


def standardize(
    signals: xr.DataArray, method: Literal["zscore", "psc"] = "zscore"
) -> xr.DataArray:
    """Standardize signals across time.

    This function operates along the ``time`` dimension and works with arrays of any
    shape, making it flexible for both extracted signals and full fUSI data.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to standardize. Must have a ``time`` dimension. Can be any shape, e.g.,
        extracted signals ``(time, voxels)``, full 3D+t imaging data ``(time, z, y,
        x)``, or regional signals ``(time, regions)``.
    method : {"zscore", "psc"}, default: "zscore"
        Standardization method:

        - ``"zscore"``: ``(x - mean) / std`` using sample standard deviation with
          Bessel's correction. Elements with zero or near-zero variance will be set to
          ``numpy.nan``.
        - ``"psc"``: Percent signal change: ``(x - mean) / |mean| * 100``. Elements
          with zero or near-zero mean will be set to ``numpy.nan``.

    Returns
    -------
    xarray.DataArray
        Standardized array with same shape and coordinates as input.

    Raises
    ------
    ValueError
        If `method` is not ``"zscore"`` or ``"psc"``.
        If `signals` does not have a ``time`` dimension.

    Warns
    -----
    UserWarning
        If `signals` have only one timepoint (standardization is skipped).

    Examples
    --------
    Standardize extracted signals:

    >>> import xarray as xr
    >>> import numpy as np
    >>> signals = xr.DataArray(
    ...     np.random.randn(100, 50),
    ...     dims=["time", "voxels"]
    ... )
    >>> standardized = standardize(signals, method="zscore")
    >>> standardized.mean(dim="time").values  # Should be close to 0.
    >>> standardized.std(dim="time", ddof=1).values  # Should be close to 1.

    Standardize 3D+t imaging data directly:

    >>> imaging_data = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"]
    ... )
    >>> standardized_3dt = standardize(imaging_data, method="psc")
    >>> # Each voxel is independently standardized across time.
    """
    if method not in {"zscore", "psc"}:
        raise ValueError(f"method must be 'zscore' or 'psc', got '{method}'")

    if "time" not in signals.dims:
        raise ValueError("signals must have a 'time' dimension")

    if signals.sizes["time"] == 1:
        warnings.warn(
            "Standardization of signals with only 1 timepoint would lead to "
            "zero or undefined values. Returning unchanged signals.",
        )
        return signals.copy()

    mean = signals.mean(dim="time")
    result = signals - mean

    # We avoid numerical problems with zero-variance (zscore) or zero-mean (psc) voxels
    # by setting them to NaN.
    eps = np.finfo(np.float64).eps
    if method == "zscore":
        std = signals.std(dim="time", ddof=1)

        invalid_mask = std < eps

        # Xarray doesn't print warnings for division by zero.
        result = result / std
    elif method == "psc":
        invalid_mask = np.abs(mean) < eps

        # Xarray doesn't print warnings for division by zero.
        result = result / np.abs(mean) * 100

    result = result.where(~invalid_mask, np.nan)

    return result
