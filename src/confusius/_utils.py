"""Package-level utilities for confusius."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import xarray as xr


def find_stack_level() -> int:
    """Find the first place in the stack that is not inside confusius.

    Adapted from
    [pandas](https://github.com/pandas-dev/pandas/tree/main/pandas/util/_exceptions.py#L37)
    and
    [Nilearn](https://github.com/nilearn/nilearn/blob/2d1a2c6d901ef4aba2737ed84e08ad1956afd123/nilearn/_utils/logger.py#L150).

    Returns
    -------
    int
        Stack level pointing to the first frame outside the confusius package.
    """
    import confusius

    pkg_dir = Path(confusius.__file__).parent

    frame = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if not filename.startswith(str(pkg_dir)):
                break
            frame = frame.f_back
            n += 1
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n


def _compute_spacing(data: xr.DataArray) -> dict[str, float | None]:
    """Compute coordinate spacing for all dimensions of a DataArray.

    For each dimension:

    - If the coordinate has two or more points and is uniformly sampled, returns the
      median step size.
    - If the coordinate has a single point, returns the ``voxdim`` coordinate attribute
      if present, otherwise ``None`` with a warning.
    - If the coordinate is missing or has non-uniform spacing, returns ``None`` with a
      warning.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray whose coordinate spacing to compute.

    Returns
    -------
    dict[str, float | None]
        Spacing per dimension in DataArray dimension order. ``None`` indicates that
        spacing is undefined for that dimension.
    """
    result: dict[str, float | None] = {}
    for dim in (str(d) for d in data.dims):
        if dim not in data.coords:
            warnings.warn(
                f"Dimension '{dim}' has no coordinate; spacing is undefined.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result[dim] = None
            continue
        coord = data.coords[dim]
        if len(coord) < 2:
            if "voxdim" in coord.attrs:
                result[dim] = float(coord.attrs["voxdim"])
            else:
                warnings.warn(
                    f"Dimension '{dim}' has a single coordinate point and no "
                    "'voxdim' attribute; spacing is undefined.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                result[dim] = None
            continue
        diffs = np.diff(coord.values)
        if not np.allclose(diffs, diffs[0], rtol=1e-5):
            warnings.warn(
                f"Coordinate '{dim}' has non-uniform sampling; spacing is undefined.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result[dim] = None
        else:
            result[dim] = float(np.median(diffs))
    return result


def _one_level_deeper() -> int:
    """Call `find_stack_level` one level deeper.

    Used in tests for `find_stack_level`. Must be defined in a ConfUSIus module (not a
    test file) so the stack level counter increments past this frame.

    Returns
    -------
    int
        Result of `find_stack_level` called from within ConfUSIus.
    """
    return find_stack_level()
