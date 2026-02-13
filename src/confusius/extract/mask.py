"""Implementation of `with_mask` extraction function."""

import numpy as np
import xarray as xr


def with_mask(
    data: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    """Extract signals from fUSI data using a boolean mask.

    This function flattens the spatial dimensions specified by the mask into a
    single ``voxels`` dimension, while preserving all other dimensions (e.g., time,
    pose).

    Parameters
    ----------
    data : xarray.DataArray
        Input array with spatial dimensions matching the mask. Can have any number
        of non-spatial dimensions (e.g., ``time``, ``pose``). The spatial dimensions
        must match those in the mask.
    mask : xarray.DataArray
        Boolean mask defining which voxels to extract. Its dimensions define the
        spatial dimensions that will be flattened. Must have identical coordinates
        to the data's spatial dimensions.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions flattened into a ``voxels`` dimension.
        All non-spatial dimensions are preserved. The ``voxels`` dimension has a
        MultiIndex storing spatial coordinates.

        For example:

        - ``(time, z, y, x)`` → ``(time, voxels)``
        - ``(time, pose, z, y, x)`` → ``(time, pose, voxels)``
        - ``(z, y, x)`` → ``(voxels,)``

        For simple round-trip reconstruction, use ``.unstack("voxels")`` which
        re-creates the original DataArray using the smallest bounding box containing the
        masked voxels. For full mask shape reconstruction, use
        `confusius.extract.unmask`.

    Raises
    ------
    ValueError
        If `mask` dimensions don't match `data`'s spatial dimensions, or if `data` has
        fewer than 2 spatial dimensions.
    TypeError
        If `mask` is not boolean dtype.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius import extract
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
    >>> signals = extract.with_mask(data, mask)
    >>> signals.dims
    ("time", "voxels")
    >>>
    >>> # 3D+t data with extra dim: (time, pose, z, y, x)
    >>> pose_data = xr.DataArray(
    ...     np.random.randn(100, 5, 10, 20, 30),
    ...     dims=["time", "pose", "z", "y", "x"],
    ... )
    >>> pose_signals = extract.with_mask(pose_data, mask)
    >>> pose_signals.dims
    ("time", "pose", "voxels")

    """
    spatial_dims = list(mask.dims)

    if len(spatial_dims) < 2:
        raise ValueError(
            f"Mask must have at least 2 spatial dimensions, got {len(spatial_dims)}: "
            f"{spatial_dims}"
        )

    if mask.dtype != bool:
        raise TypeError(f"Mask must be boolean dtype, got {mask.dtype}")

    if not set(spatial_dims).issubset(set(data.dims)):
        missing_dims = set(spatial_dims) - set(data.dims)
        raise ValueError(
            f"Data is missing spatial dimensions from mask: {missing_dims}. "
            f"Data dims: {data.dims}, Mask dims: {spatial_dims}"
        )

    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]
    if non_spatial_dims:
        sel_dict = {d: 0 for d in non_spatial_dims}
        template = data.isel(sel_dict)
    else:
        template = data

    for dim in spatial_dims:
        if dim in mask.coords and dim in template.coords:
            if not np.array_equal(mask.coords[dim].values, template.coords[dim].values):
                raise ValueError(
                    f"Mask coordinates for dimension '{dim}' do not match data coordinates.\n"
                    f"  Data {dim}: {template.coords[dim].values}\n"
                    f"  Mask {dim}: {mask.coords[dim].values}\n"
                    f"Coordinates must be identical for masking to work correctly."
                )

    mask_aligned = mask.reindex_like(template)

    data_flat = data.stack(space=spatial_dims)

    mask_flat = mask_aligned.values.flatten()
    signals = data_flat.isel(space=mask_flat)

    signals = signals.rename({"space": "voxels"})

    return signals
