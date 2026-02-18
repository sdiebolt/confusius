"""Xarray accessor for signal extraction."""

import xarray as xr


class FUSIExtractAccessor:
    """Xarray accessor for signal extraction operations.

    Provides convenient methods for extracting signals from N-D fUSI data by flattening
    spatial dimensions, and reconstructing N-D volumes from processed signals.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>>
    >>> # 3D+t data: (time, z, y, x)
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"],
    ... )
    >>> mask = xr.DataArray(
    ...     np.random.rand(10, 20, 30) > 0.5,
    ...     dims=["z", "y", "x"],
    ... )
    >>>
    >>> # Extract signals
    >>> signals = data.fusi.extract.with_mask(mask)
    >>> signals.dims
    ("time", "voxels")
    >>>
    >>> # Reconstruct full spatial volume from signals
    >>> reconstructed = signals.fusi.extract.unmask(mask)
    >>> reconstructed.dims
    ("time", "z", "y", "x")
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def with_mask(self, mask: xr.DataArray) -> xr.DataArray:
        """Extract signals using a boolean mask.

        Parameters
        ----------
        mask : xarray.DataArray
            Boolean mask with same spatial dimensions and coordinates as data.

        Returns
        -------
        xarray.DataArray
            Array with spatial dimensions flattened into a ``voxels`` dimension.
            All non-spatial dimensions are preserved. The ``voxels`` dimension has a
            MultiIndex storing spatial coordinates.

            For simple round-trip reconstruction, use ``.unstack("voxels")`` which
            re-creates the original DataArray using the smallest bounding box. For full
            mask shape reconstruction, use ``.fusi.extract.unmask()``.

        Examples
        --------
        >>> signals = data.fusi.extract.with_mask(mask)
        >>> signals.dims
        ("time", "voxels")
        >>>
        >>> # Quick bounding box reconstruction
        >>> bbox = signals.unstack("voxels")
        >>>
        >>> # Full mask shape reconstruction
        >>> full = signals.fusi.extract.unmask(mask)
        """
        from confusius.extract.mask import extract_with_mask

        return extract_with_mask(self._obj, mask)

    def unmask(
        self,
        mask: xr.DataArray,
        fill_value: float = 0.0,
    ) -> xr.DataArray:
        """Reconstruct N-D volume from masked signals.

        Reconstructs the full spatial volume from a DataArray of signals, which must
        have a ``voxels`` dimension. This is a convenience wrapper around
        `confusius.extract.unmask()`.

        Parameters
        ----------
        mask : xarray.DataArray
            Boolean mask used for the original extraction. Provides spatial dimensions
            and coordinates for reconstruction.
        fill_value : float, default: 0.0
            Value to fill in non-masked voxels.

        Returns
        -------
        xarray.DataArray
            Reconstructed DataArray with shape ``(..., z, y, x)`` where spatial
            coordinates come from the mask.

        Examples
        --------
        >>> signals = data.fusi.extract.with_mask(mask)
        >>> signals.dims
        ("time", "voxels")
        >>> reconstructed = signals.fusi.extract.unmask(mask)
        >>> reconstructed.dims
        ("time", "z", "y", "x")
        """
        from confusius.extract.reconstruction import unmask

        return unmask(self._obj, mask=mask, fill_value=fill_value)
