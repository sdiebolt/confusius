"""Timing utilities for ConfUSIus."""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import find_stack_level, get_representative_step

_TIME_UNIT_TO_SECONDS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "usec": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
}
"""Mapping from common time-unit strings to seconds."""

_TIMING_REF_FACTORS: dict[str, float] = {"start": 0.0, "center": 0.5, "end": 1.0}
"""Mapping from timing reference name to fractional position within a volume."""


def _get_time_in_seconds(data: xr.DataArray) -> npt.NDArray[np.float64]:
    """Return a coordinate converted to seconds.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the coordinate to convert.

    Returns
    -------
    numpy.ndarray
        Coordinate values converted to seconds.

    Warns
    -----
    UserWarning
        If the coordinate has missing or unknown units. In that case, seconds are
        assumed.
    """
    if "time" not in data.coords:
        raise ValueError("DataArray must have a `time` coordinate.")

    values = np.asarray(data.coords["time"].values, dtype=np.float64)
    return values * _get_time_to_seconds_factor(data)


def _get_time_to_seconds_factor(data: xr.DataArray) -> float:
    """Return the multiplicative factor that converts the time coordinate to seconds.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the time coordinate.

    Returns
    -------
    float
        Multiplicative factor that converts values expressed in the time coordinate's
        units to seconds. Returns `1.0` when units are missing or unknown.

    Raises
    ------
    ValueError
        If the DataArray does not have a `time` coordinate.

    Warns
    -----
    UserWarning
        If the coordinate has missing or unknown units. In that case, seconds are
        assumed.
    """
    if "time" not in data.coords:
        raise ValueError("DataArray must have a `time` coordinate.")

    units = data.coords["time"].attrs.get("units")
    if units is None:
        warnings.warn(
            "Time coordinate has no `units` attribute. Assuming seconds.",
            stacklevel=find_stack_level(),
        )
        return 1.0

    unit = str(units).lower()
    if unit not in _TIME_UNIT_TO_SECONDS:
        warnings.warn(
            f"Time coordinate uses unknown units {units!r}. Assuming seconds.",
            stacklevel=find_stack_level(),
        )
        return 1.0

    return _TIME_UNIT_TO_SECONDS[unit]


def _get_representative_time_step(
    data: xr.DataArray,
    *,
    in_seconds: bool = False,
    uniformity_tolerance: float = 1e-5,
) -> tuple[float | None, bool]:
    """Return a representative time step for the time coordinate.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the time coordinate.
    in_seconds : bool, default: False
        Whether to first convert the time coordinate to seconds.
    uniformity_tolerance : float, default: 1e-5
        Maximum allowed relative range `(max_diff - min_diff) / median_diff` for the
        time coordinate to be considered uniform.

    Returns
    -------
    step : float or None
        Representative time step in seconds when `in_seconds=True`, otherwise in the
        native units of the time coordinate. Returns `None` if fewer than two time
        points are available.
    approximate : bool
        Whether the returned step is a median approximation derived from non-uniform
        sampling.
    """
    time_values = (
        _get_time_in_seconds(data) if in_seconds else data.coords["time"].values
    )
    return get_representative_step(
        np.asarray(time_values, dtype=np.float64),
        uniformity_tolerance=uniformity_tolerance,
    )


def convert_time_reference(
    time: npt.NDArray[np.floating],
    volume_duration: float,
    from_reference: Literal["start", "center", "end"],
    to_reference: Literal["start", "center", "end"],
) -> npt.NDArray[np.floating]:
    """Convert timings from one volume acquisition reference to another.

    Parameters
    ----------
    time : numpy.ndarray
        Input timings in any physical time unit.
    volume_duration : float
        Duration of one volume in the same units as `time`.
    from_reference : {"start", "center", "end"}
        Reference point used by the input timings.
    to_reference : {"start", "center", "end"}
        Reference point to convert the timings to.

    Returns
    -------
    numpy.ndarray
        Converted timings in the same units as `time`.
    """
    if from_reference == to_reference:
        return time

    from_factor = _TIMING_REF_FACTORS[from_reference]
    to_factor = _TIMING_REF_FACTORS[to_reference]
    offset = (to_factor - from_factor) * volume_duration

    return time + offset
