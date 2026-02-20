"""Xarray accessor for IO operations."""

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from confusius.io.nifti import NiftiVersion

__all__ = ["FUSIIOAccessor"]


class FUSIIOAccessor:
    """Accessor for IO operations on fUSI data.

    This accessor provides methods to save fUSI data to various formats
    including NIfTI.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> data = xr.DataArray(np.random.rand(10, 32, 1, 64),
    ...                       dims=["time", "z", "y", "x"])
    >>> data.fusi.io.to_nifti("output.nii.gz")
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def to_nifti(
        self,
        path: str | Path,
        nifti_version: "NiftiVersion" = 1,
    ) -> None:
        """Save DataArray to NIfTI format.

        Saves the DataArray to a NIfTI file and always writes a BIDS-style JSON sidecar
        alongside it. The data is transposed to NIfTI convention ``(x, y, z, time)``
        before saving.

        Parameters
        ----------
        path : str or pathlib.Path
            Output path for the NIfTI file, with ``.nii`` or ``.nii.gz`` extension. If
            ``.nii.gz`` is used, the file will be saved in compressed format.
        nifti_version : {1, 2}, default: 1
            NIfTI format version to use. Version 2 is a simple extension to support
            larger files and arrays with dimension sizes greater than 32,767.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> data = xr.DataArray(np.random.rand(10, 32, 1, 64),
        ...                       dims=["time", "z", "y", "x"])
        >>> data.fusi.io.to_nifti("recording.nii.gz")
        """
        from confusius.io.nifti import save_nifti

        save_nifti(self._obj, path=path, nifti_version=nifti_version)
