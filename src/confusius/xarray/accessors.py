"""Xarray accessor for fUSI-specific operations."""

import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr

from confusius._utils import find_stack_level
from confusius.xarray.extract import FUSIExtractAccessor
from confusius.xarray.io import FUSIIOAccessor
from confusius.xarray.iq import FUSIIQAccessor
from confusius.xarray.plotting import FUSIPlotAccessor
from confusius.xarray.registration import FUSIRegistrationAccessor
from confusius.xarray.scale import FUSIScaleAccessor

__all__ = ["FUSIAccessor"]


@xr.register_dataarray_accessor("fusi")
class FUSIAccessor:
    """Xarray accessor for fUSI-specific operations.

    Provides convenient methods for functional ultrasound imaging data analysis.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The ``DataArray`` to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius import xarray as cxr  # Registers the accessor
    >>> data = xr.DataArray([1, 10, 100, 1000])
    >>> data.fusi.scale.db(factor=20)
    <xarray.DataArray (dim_0: 4)>
    array([-60., -40., -20.,   0.])
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    @property
    def scale(self) -> FUSIScaleAccessor:
        """Access scaling operations.

        Returns
        -------
        FUSIScaleAccessor
            Accessor for scaling transformations.

        Examples
        --------
        >>> data = xr.DataArray([1, 10, 100, 1000])
        >>> data.fusi.scale.db(factor=10)
        <xarray.DataArray (dim_0: 4)>
        array([-30., -20., -10.,   0.])
        """
        return FUSIScaleAccessor(self._obj)

    @property
    def plot(self) -> FUSIPlotAccessor:
        """Access plotting operations.

        Returns
        -------
        FUSIPlotAccessor
            Accessor for plotting methods.

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["iq"]
        >>> viewer, layer = data.fusi.plot.napari()
        """
        return FUSIPlotAccessor(self._obj)

    @property
    def register(self) -> FUSIRegistrationAccessor:
        """Access registration operations.

        Returns
        -------
        FUSIRegistrationAccessor
            Accessor for registration methods.

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["pwd"]
        >>> registered = data.fusi.register.volumewise(reference_time=0)
        """
        return FUSIRegistrationAccessor(self._obj)

    @property
    def io(self) -> FUSIIOAccessor:
        """Access IO operations.

        Returns
        -------
        FUSIIOAccessor
            Accessor for IO methods (to_nii, etc.).

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["pwd"]
        >>> data.fusi.io.to_nii("recording.nii.gz")
        """
        return FUSIIOAccessor(self._obj)

    @property
    def iq(self) -> FUSIIQAccessor:
        """Access IQ processing operations.

        Returns
        -------
        FUSIIQAccessor
            Accessor for IQ processing methods.

        Examples
        --------
        >>> import xarray as xr
        >>> ds = xr.open_zarr("output.zarr")
        >>> iq = ds["iq"]
        >>> pwd = iq.fusi.iq.process_to_power_doppler()
        >>> velocity = iq.fusi.iq.process_to_axial_velocity()
        """
        return FUSIIQAccessor(self._obj)

    @property
    def extract(self) -> FUSIExtractAccessor:
        """Access signal extraction operations.

        Returns
        -------
        FUSIExtractAccessor
            Accessor for extracting signals from fUSI data and reconstructing fUSI data
            from processed signals.

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["pwd"]
        >>> mask = xr.open_zarr("brain_mask.zarr")["mask"]
        >>> signals = data.fusi.extract.with_mask(mask)
        >>> # ... process signals ...
        >>> restored = signals.fusi.extract.unmask(mask)
        """
        return FUSIExtractAccessor(self._obj)

    def get_spacing(
        self,
        dims: Sequence[str] | None = None,
        rtol: float = 1e-5,
    ) -> dict[str, float | None]:
        """Get coordinate spacing for spatial dimensions.

        Parameters
        ----------
        dims : sequence of str, optional
            Dimensions to compute spacing for. Defaults to all dimensions except
            ``"time"``, in DataArray dimension order.
        rtol : float, default: 1e-5
            Relative tolerance for the uniformity check.

        Returns
        -------
        dict[str, float | None]
            Spacing per dimension, in DataArray dimension order. Returns ``None`` for
            dimensions with non-uniform or undefined spacing, with a warning.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((10, 20)),
        ...     dims=["y", "x"],
        ...     coords={"y": np.arange(10) * 0.2, "x": np.arange(20) * 0.1},
        ... )
        >>> data.fusi.get_spacing()
        {'y': 0.2, 'x': 0.1}
        """
        active_dims: list[str] = (
            [str(d) for d in self._obj.dims if d != "time"]
            if dims is None
            else list(dims)
        )
        result: dict[str, float | None] = {}
        for dim in active_dims:
            if dim not in self._obj.coords or len(self._obj.coords[dim]) < 2:
                warnings.warn(
                    f"Dimension '{dim}' has fewer than two coordinate points; "
                    "spacing is undefined.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                result[dim] = None
                continue
            coord = self._obj.coords[dim].values
            diffs = np.diff(coord)
            if not np.allclose(diffs, diffs[0], rtol=rtol):
                warnings.warn(
                    f"Coordinate '{dim}' has non-uniform spacing; returning None.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                result[dim] = None
            else:
                result[dim] = float(np.median(diffs))
        return result

    def get_dt(self, rtol: float = 1e-5) -> float | None:
        """Get the temporal sampling period.

        Parameters
        ----------
        rtol : float, default: 1e-5
            Relative tolerance for the uniformity check.

        Returns
        -------
        float or None
            Temporal spacing (dt). Returns ``None`` if time is not uniformly sampled,
            with a warning.

        Raises
        ------
        ValueError
            If the DataArray has no ``"time"`` dimension.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((10, 5)),
        ...     dims=["time", "x"],
        ...     coords={"time": np.arange(10) * 0.05, "x": np.arange(5) * 0.1},
        ... )
        >>> data.fusi.get_dt()
        0.05
        """
        if "time" not in self._obj.dims:
            raise ValueError("DataArray has no 'time' dimension.")
        time_coord = self._obj.coords["time"]
        if len(time_coord) < 2:
            warnings.warn(
                "Time coordinate has fewer than two points; dt is undefined.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            return None
        diffs = np.diff(time_coord.values)
        if not np.allclose(diffs, diffs[0], rtol=rtol):
            warnings.warn(
                "Time coordinate has non-uniform spacing; returning None.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            return None
        return float(np.median(diffs))
