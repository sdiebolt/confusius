"""Xarray accessor for signal extraction."""

import numpy as np
import xarray as xr


class FUSIExtractAccessor:
    """Xarray accessor for signal extraction operations.

    Provides convenient methods for extracting signals from N-D fUSI data by
    flattening spatial dimensions, and reconstructing N-D volumes from processed
    signals.

    Parameters
    ----------
    xarray_obj : xr.DataArray
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
    >>> # Process with sklearn
    >>> from sklearn.decomposition import PCA
    >>> components = PCA(n_components=5).fit_transform(signals.values)
    >>>
    >>> # Unmask back to full shape
    >>> spatial_components = signals.fusi.extract.unmask(
    ...     components.T,  # (5, n_voxels)
    ...     mask=mask,
    ...     new_dims=["component"],
    ... )
    >>> spatial_components.dims
    ("component", "z", "y", "x")

    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def with_mask(self, mask: xr.DataArray) -> xr.DataArray:
        """Extract signals using a boolean mask.

        Parameters
        ----------
        mask : xr.DataArray
            Boolean mask with same spatial dimensions and coordinates as data.

        Returns
        -------
        xr.DataArray
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
        >>> full = signals.fusi.extract.unmask(signals.values, mask)

        """
        from confusius.extract.mask import with_mask

        return with_mask(self._obj, mask)

    def unmask(
        self,
        signals: np.ndarray | xr.DataArray,
        mask: xr.DataArray,
        new_dims: list[str] | None = None,
        new_dims_coords: dict[str, np.ndarray] | None = None,
        attrs: dict | None = None,
        fill_value: float = 0.0,
    ) -> xr.DataArray:
        """Reconstruct N-D volume from signals using a mask.

        This is a convenience wrapper around `confusius.extract.unmask()`.

        Parameters
        ----------
        signals : numpy.ndarray or xarray.DataArray
            Array with shape ``(..., voxels)`` where ``...`` can be any number of
            dimensions. The last dimension must correspond to masked voxels.
        mask : xarray.DataArray
            Boolean mask used for the original extraction. Provides spatial dimensions
            and coordinates for reconstruction.
        new_dims : list of str, optional
            Names for leading dimensions when `signals` is a Numpy array. Must match the
            number of leading dimensions ``(ndim - 1)``. If not provided, uses ``["dim_0",
            "dim_1", ...]``. Ignored if `signals` is a DataArray.
        new_dims_coords : dict[str, numpy.ndarray], optional
            Coordinates for leading dimensions when `signals` is a Numpy array. Keys must
            match dimension names in `new_dims`. If not provided, uses integer indices for
            all dimensions. Ignored if `signals` is a DataArray.
        attrs : dict, optional
            Attributes to attach to the output DataArray.
        fill_value : float, default: 0.0
            Value to fill in non-masked voxels.

        Returns
        -------
        xr.DataArray
            Reconstructed DataArray with shape ``(..., z, y, x)`` where spatial
            coordinates come from the mask.

        Examples
        --------
        >>> signals = data.fusi.extract.with_mask(mask)
        >>> # Process signals with sklearn
        >>> from sklearn.decomposition import PCA
        >>> pca = PCA(n_components=5)
        >>> components = pca.fit_transform(signals.values)
        >>>
        >>> # Unmask spatial components
        >>> spatial_pca = signals.fusi.extract.unmask(
        ...     components.T,  # (5, n_voxels)
        ...     mask=mask,
        ...     new_dims=["component"],
        ...     attrs=data.attrs,
        ... )
        >>> spatial_pca.dims
        ("component", "z", "y", "x")

        """
        from confusius.extract.unmask import unmask

        return unmask(signals, mask, new_dims, new_dims_coords, attrs, fill_value)
