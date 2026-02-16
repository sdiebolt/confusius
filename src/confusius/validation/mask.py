"""Mask validation utilities."""

import numpy as np
import numpy.typing as npt
import xarray as xr


def validate_mask(
    mask: xr.DataArray,
    data: xr.DataArray,
    mask_name: str = "mask",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> npt.NDArray:
    """Validate that a mask matches data spatial dimensions and coordinates.

    Parameters
    ----------
    mask : xarray.DataArray
        Mask to validate. Must have boolean dtype and coordinates must match data.
    data : xarray.DataArray
        Data array to validate mask against.
    mask_name : str, default: "mask"
        Name of the mask parameter (used in error messages).
    rtol : float, default: 1e-5
        Relative tolerance for coordinate comparison.
    atol : float, default: 1e-8
        Absolute tolerance for coordinate comparison.

    Returns
    -------
    numpy.ndarray
        Validated mask as numpy array.

    Raises
    ------
    TypeError
        If mask is not boolean dtype or not a DataArray.
    ValueError
        If mask dimensions don't match data or if coordinates don't match.

    Notes
    -----
    Uses ``np.allclose`` (with rtol/atol) for coordinate comparison, which is more
    appropriate for floating-point coordinates than exact equality.
    """
    if not isinstance(mask, xr.DataArray):
        raise TypeError(
            f"{mask_name} must be an xarray.DataArray, got {type(mask).__name__}."
        )

    if mask.dtype != bool:
        raise TypeError(f"{mask_name} must be boolean dtype, got {mask.dtype}.")

    spatial_dims = list(mask.dims)

    if not set(spatial_dims).issubset(set(data.dims)):
        missing_dims = set(spatial_dims) - set(data.dims)
        raise ValueError(
            f"Data is missing spatial dimensions from {mask_name}: {missing_dims}. "
            f"Data dims: {data.dims}, {mask_name} dims: {spatial_dims}."
        )

    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]
    if non_spatial_dims:
        sel_dict = {d: 0 for d in non_spatial_dims}
        template = data.isel(sel_dict)
    else:
        template = data

    for dim in spatial_dims:
        data_has_coord = dim in template.coords
        mask_has_coord = dim in mask.coords

        if data_has_coord and not mask_has_coord:
            raise ValueError(
                f"{mask_name} is missing coordinate '{dim}' that exists in data."
            )

        if mask_has_coord and not data_has_coord:
            raise ValueError(
                f"{mask_name} has coordinate '{dim}' but data does not."
            )

        if data_has_coord and mask_has_coord:
            mask_coord_vals = mask.coords[dim].values
            data_coord_vals = template.coords[dim].values

            if mask_coord_vals.shape != data_coord_vals.shape:
                raise ValueError(
                    f"{mask_name} coordinates for dimension '{dim}' do not match "
                    f"data coordinates (different shapes).\n"
                    f"  Data {dim} shape: {data_coord_vals.shape}\n"
                    f"  {mask_name} {dim} shape: {mask_coord_vals.shape}"
                )

            if not np.allclose(mask_coord_vals, data_coord_vals, rtol=rtol, atol=atol):
                raise ValueError(
                    f"{mask_name} coordinates for dimension '{dim}' do not match "
                    f"data coordinates (within rtol={rtol}, atol={atol}).\n"
                    f"  Data {dim}: {data_coord_vals}\n"
                    f"  {mask_name} {dim}: {mask_coord_vals}"
                )

    return mask.values
