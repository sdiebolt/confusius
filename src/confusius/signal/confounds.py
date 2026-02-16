"""Confound regression functions for signal preprocessing."""

import numpy as np
import numpy.typing as npt
import scipy.linalg
import xarray as xr

from confusius.signal._utils import validate_time_series


def _validate_confounds(
    signals: xr.DataArray, confounds: xr.DataArray | npt.NDArray
) -> np.ndarray:
    """Validate confounds array matches signals.

    Parameters
    ----------
    signals : xarray.DataArray
        Signals with 'time' dimension.
    confounds : numpy.ndarray
        Confound regressors to validate.

    Returns
    -------
    numpy.ndarray
        Validated 2D confounds array.

    Raises
    ------
    ValueError
        If confounds have incorrect shape or time dimension mismatch.
    TypeError
        If confounds have unsupported type.
    """
    if isinstance(confounds, xr.DataArray):
        if "time" not in confounds.dims:
            raise ValueError("confounds DataArray must have a 'time' dimension")
        if "time" in signals.coords and "time" in confounds.coords:
            if not np.array_equal(
                signals.coords["time"].values, confounds.coords["time"].values
            ):
                raise ValueError(
                    "confounds time coordinates do not match signals time coordinates"
                )
        confounds = confounds.values

    if not isinstance(confounds, np.ndarray):
        raise TypeError(
            "confounds must be an xarray.DataArray or numpy.ndarray, got "
            f"{type(confounds).__name__}"
        )

    if confounds.ndim == 1:
        confounds = np.atleast_2d(confounds).T
    elif confounds.ndim != 2:
        raise ValueError(f"confounds must be 1D or 2D, got {confounds.ndim}D")

    n_time = signals.sizes["time"]
    if confounds.shape[0] != n_time:
        raise ValueError(
            f"confounds time dimension ({confounds.shape[0]}) does not match "
            f"signals time dimension ({n_time})"
        )

    return confounds


def _standardize_confounds(confounds: np.ndarray) -> np.ndarray:
    """Standardize confounds by max absolute value for numerical stability.

    Following nilearn's approach: divide each confound by its maximum absolute
    value to control the range while preserving constant terms. This improves
    numerical stability without removing the mean (unlike z-scoring).

    Parameters
    ----------
    confounds : numpy.ndarray
        Confound regressors, shape (time, n_confounds).

    Returns
    -------
    numpy.ndarray
        Standardized confounds with values in approximately [-1, 1] range.
        Constant columns are preserved (not zeroed out).

    Notes
    -----
    Based on nilearn.signal.clean implementation which standardizes by max
    absolute value to improve numerical stability while keeping constant
    contributions intact.
    """
    confound_max = np.max(np.abs(confounds), axis=0)
    confound_max[confound_max == 0] = 1
    confounds = confounds / confound_max

    return confounds


def _regress_confounds_numpy(
    signals: np.ndarray,
    confounds: np.ndarray,
    standardize_confounds: bool = True,
) -> np.ndarray:
    """Core confound regression using QR decomposition.

    Uses QR decomposition with column pivoting for numerical stability.
    Projects signals onto the orthogonal complement of confound space.

    Parameters
    ----------
    signals : numpy.ndarray
        Signals array of any shape, with time along first axis.
    confounds : numpy.ndarray
        Confound regressors, shape (time, n_confounds).
    standardize_confounds : bool, default=True
        Whether to standardize confounds before regression.

    Returns
    -------
    numpy.ndarray
        Residuals after confound regression, same shape as signals.

    Notes
    -----
    Based on nilearn.signal.clean implementation which follows
    Friston et al. (1994) for confound removal via projection onto
    the orthogonal of the signal space.
    """
    if standardize_confounds:
        confounds = _standardize_confounds(confounds)

    Q, R, pivot = scipy.linalg.qr(confounds, mode="economic", pivoting=True)

    tol = np.finfo(np.float64).eps * 100.0
    rank = np.sum(np.abs(np.diag(R)) > tol)
    Q = Q[:, :rank]

    original_shape = signals.shape
    if signals.ndim > 2:
        signals_2d = signals.reshape(signals.shape[0], -1)
    else:
        signals_2d = signals

    projection = Q @ (Q.T @ signals_2d)
    residuals_2d = signals_2d - projection

    if signals.ndim > 2:
        residuals = residuals_2d.reshape(original_shape)
    else:
        residuals = residuals_2d

    return residuals


def _regress_confounds_wrapper(data, axis, confounds, standardize_confounds):
    """Wrapper for confound regression that works with xr.apply_ufunc.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    axis : int
        Axis along which to apply regression (should be time axis).
    confounds : numpy.ndarray
        Confound regressors.
    standardize_confounds : bool
        Whether to standardize confounds.

    Returns
    -------
    numpy.ndarray
        Residuals after confound regression.
    """
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    result = _regress_confounds_numpy(data, confounds, standardize_confounds)

    if axis != 0:
        result = np.moveaxis(result, 0, axis)

    return result


def regress_confounds(
    signals: xr.DataArray,
    confounds: npt.NDArray | xr.DataArray,
    standardize_confounds: bool = True,
) -> xr.DataArray:
    """Remove confounds from signals via linear regression.

    This function performs confound regression by projecting the signals onto the
    orthogonal complement of the confound space. This removes variance in the signals
    that can be explained by the confounds.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals to clean. Must have a ``time`` dimension. Can be any shape,
        e.g., extracted signals ``(time, voxels)``, full 3D+t imaging data
        ``(time, z, y, x)``, or regional signals ``(time, regions)``.
    confounds : (time, n_confounds) numpy.ndarray or xarray.DataArray
        Confound regressors to remove. Can have shape ``(time,)`` for a single
        confound. The time dimension (and coordinates when passing a DataArray) must
        match the signals exactly.
    standardize_confounds : bool, default: True
        If ``True``, standardize confounds by their maximum absolute value before
        regression. This improves numerical stability while preserving constant terms.

    Returns
    -------
    xarray.DataArray
        Residuals after confound regression, same shape and coordinates
        as input signals.

    Raises
    ------
    ValueError
        If ``signals`` does not have a ``time`` dimension, or if
        ``confounds`` have mismatched time dimension or invalid shape.
    TypeError
        If ``confounds`` is not a numpy array or xarray DataArray.

    Notes
    -----
    - Uses QR decomposition with column pivoting for numerical stability.
    - Handles rank-deficient confound matrices (e.g., collinear confounds)
      by removing redundant columns.
    - Based on the projection method from Friston et al. (1994).
    - Similar to `nilearn.signal.clean` but adapted for xarray DataArrays.

    References
    ----------
    [^1]:
        Friston, K. J., Holmes, A. P., Worsley, K. J., Poline, J. P., Frith, C. D., &
        Frackowiak, R. S. (1994). Statistical parametric maps in functional imaging: a
        general linear approach. Human brain mapping, 2(4), 189-210.

    Examples
    --------
    Remove motion parameters from extracted signals:

    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.signal import regress_confounds
    >>> # Create signals (100 timepoints, 50 voxels)
    >>> signals = xr.DataArray(
    ...     np.random.randn(100, 50),
    ...     dims=["time", "voxels"],
    ...     coords={"time": np.arange(100) * 0.1}
    ... )
    >>> # Create motion confounds (6 motion parameters)
    >>> motion_params = np.random.randn(100, 6)
    >>> # Remove motion effects
    >>> cleaned = regress_confounds(signals, motion_params)

    Works on 3D+t imaging data:

    >>> imaging_data = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"]
    ... )
    >>> cleaned_imaging = regress_confounds(
    ...     imaging_data, motion_params
    ... )
    """
    time_axis = validate_time_series(signals, "confound regression")

    confounds = _validate_confounds(signals, confounds)

    result = xr.apply_ufunc(
        _regress_confounds_wrapper,
        signals,
        kwargs={
            "axis": time_axis,
            "confounds": confounds,
            "standardize_confounds": standardize_confounds,
        },
        dask="parallelized",
        output_dtypes=[signals.dtype],
    )

    return result
