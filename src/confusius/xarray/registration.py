"""Xarray accessor for registration."""

import xarray as xr

from confusius.registration.volumewise import register_volumewise


class FUSIRegistrationAccessor:
    """Accessor for registration operations on fUSI data.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.open_zarr("output.zarr")["pwd"]
    >>> registered = data.fusi.register.volumewise(reference_time=0)
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def volumewise(
        self,
        reference_time: int = 0,
        n_jobs: int = -1,
        allow_rotation: bool = True,
        rotation_penalty: float = 100.0,
    ) -> xr.DataArray:
        """
        Register all volumes to a reference time point.

        Parameters
        ----------
        reference_time : int, default: 0
            Index of the time point to use as registration target.
        n_jobs : int, default: -1
            Number of parallel jobs. -1 uses all available CPUs.
            Use 1 for serial processing.
        allow_rotation : bool, default: True
            Whether to allow rotation in addition to translation.
            Uses conservative optimizer settings to prevent spurious rotations.
            Set to False for translation-only registration.
        rotation_penalty : float, default: 100.0
            Penalty factor for rotation (only used if allow_rotation=True).
            Higher values more strongly constrain rotation. Typical values:
            1.0=no constraint, 50.0=moderate, 100.0=strong, 200.0=very strong.

        Returns
        -------
        xarray.DataArray
            Registered data with same coordinates and attributes.

        Examples
        --------
        >>> data.fusi.register.volumewise(reference_time=0)
        >>> data.fusi.register.volumewise(reference_time=0, allow_rotation=True)
        """
        return register_volumewise(
            self._obj,
            reference_time=reference_time,
            n_jobs=n_jobs,
            allow_rotation=allow_rotation,
            rotation_penalty=rotation_penalty,
        )
