"""Utility functions for the signal module."""

import xarray as xr


def validate_time_series(
    signals: xr.DataArray,
    operation_name: str,
    check_time_chunks: bool = True,
) -> int:
    """Validate time series for signal processing operations.

    Performs common validation checks:

    1. Signals have a ``time`` dimension.
    2. Time dimension has more than 1 timepoint.
    3. Time dimension is not chunked for Dask arrays (optional).

    Parameters
    ----------
    signals : xarray.DataArray
        Input signals to validate. Must have a 'time' dimension.
    operation_name : str
        Name of the operation (used in error/warning messages).
    check_time_chunks : bool, default=True
        If True, raises an error when time dimension is chunked in a Dask array.
        Set to False for operations that can handle chunked time (e.g., standardize).

    Returns
    -------
    int
        Axis number for the 'time' dimension.

    Raises
    ------
    ValueError
        If signals has no 'time' dimension.
        If time dimension has only 1 timepoint.
        If time dimension is chunked in a Dask array (when check_time_chunks=True).
    """
    if "time" not in signals.dims:
        raise ValueError("signals must have a 'time' dimension")

    if signals.sizes["time"] == 1:
        raise ValueError(
            f"{operation_name.capitalize()} requires more than 1 timepoint, "
            f"got {signals.sizes['time']}"
        )

    time_axis = signals.get_axis_num("time")

    if check_time_chunks and hasattr(signals.data, "chunks"):
        time_chunks = signals.data.chunks[time_axis]
        if len(time_chunks) > 1:
            raise ValueError(
                f"Data is chunked along the 'time' dimension ({len(time_chunks)} "
                f"chunks), but {operation_name} requires the full time series. "
                f"Rechunk your data so 'time' is not chunked: "
                f"data.chunk({{'time': -1}})"
            )

    return time_axis
