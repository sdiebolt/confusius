"""Volumewise registration for fUSI data."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius.registration.motion import create_motion_dataframe

if TYPE_CHECKING:
    import SimpleITK as sitk


def register_volumewise(
    data: xr.DataArray,
    *,
    reference_time: int = 0,
    n_jobs: int = -1,
    allow_rotation: bool = True,
    rotation_penalty: float = 100.0,
) -> xr.DataArray:
    """Register all volumes in a fUSI recording to a reference volume.

    For 2D+time data, performs 2D registration on each frame. For 3D+time data, performs
    3D volume-to-volume registration. Uses normalized correlation metric with optional
    rotation.

    Parameters
    ----------
    data : xarray.DataArray
        Input data to register.
    reference_time : int, default: 0
        Index of the time point to use as registration target.
    n_jobs : int, default: -1
        Number of parallel jobs. -1 uses all available CPUs. Use 1 for serial
        processing.
    allow_rotation : bool, default: True
        Whether to allow rotation in addition to translation.
        Uses conservative optimizer settings to prevent spurious rotations.
    rotation_penalty : float, default: 100.0
        Penalty factor for rotation relative to translation (only used if
        allow_rotation=True). Higher values more strongly penalize rotation.
        The optimizer updates rotation `rotation_penalty` times slower than translation.

    Returns
    -------
    xarray.DataArray
        Registered data with same coordinates and attributes as input.
    """
    import SimpleITK as sitk

    if "time" not in data.dims:
        raise ValueError("Time dimension 'time' not found in data")

    data_moved = data.transpose("time", ...)

    # Singleton dimensions but be squeezed to avoid issues with SimpleITK: Gaussian
    # filters require at least 4 pixels per dimension.
    squeeze_dims = [
        d for d in data_moved.dims if d != "time" and data_moved.sizes[d] == 1
    ]
    squeezed_coords = {}
    if squeeze_dims:
        # Store coordinates of squeezed dimensions to restore later.
        for dim in squeeze_dims:
            squeezed_coords[dim] = data_moved.coords[dim]
        data_moved = data_moved.squeeze(squeeze_dims)

    arr = data_moved.values

    if arr.ndim == 3:
        # Dimensions are (time, dim0, dim1).
        dim0, dim1 = data_moved.dims[1], data_moved.dims[2]
        spacing_2d: tuple[float, float] | None = None
        if dim0 in data_moved.coords and dim1 in data_moved.coords:
            coord0 = data_moved.coords[dim0].values
            coord1 = data_moved.coords[dim1].values
            spacing0 = float(np.abs(coord0[1] - coord0[0])) if len(coord0) > 1 else 1.0
            spacing1 = float(np.abs(coord1[1] - coord1[0])) if len(coord1) > 1 else 1.0
            # SimpleITK expects (width, height) = (dim1, dim0) spacing.
            spacing_2d = (spacing1, spacing0)

        registered, transforms = _register_2dt(
            arr,
            reference_time=reference_time,
            n_jobs=n_jobs,
            spacing=spacing_2d,
            allow_rotation=allow_rotation,
            rotation_penalty=rotation_penalty,
        )

        ref_img = sitk.GetImageFromArray(arr[reference_time])
        if spacing_2d is not None:
            ref_img.SetSpacing(spacing_2d)
        time_coords = (
            data_moved.coords["time"].values if "time" in data_moved.coords else None
        )
        motion_df = create_motion_dataframe(
            transforms=transforms, reference_image=ref_img, time_coords=time_coords
        )

    elif arr.ndim == 4:
        # Dimensions are (time, dim0, dim1, dim2).
        dim0, dim1, dim2 = data_moved.dims[1], data_moved.dims[2], data_moved.dims[3]
        spacing_3d: tuple[float, float, float] | None = None
        if (
            dim0 in data_moved.coords
            and dim1 in data_moved.coords
            and dim2 in data_moved.coords
        ):
            coord0 = data_moved.coords[dim0].values
            coord1 = data_moved.coords[dim1].values
            coord2 = data_moved.coords[dim2].values
            spacing0 = float(np.abs(coord0[1] - coord0[0])) if len(coord0) > 1 else 1.0
            spacing1 = float(np.abs(coord1[1] - coord1[0])) if len(coord1) > 1 else 1.0
            spacing2 = float(np.abs(coord2[1] - coord2[0])) if len(coord2) > 1 else 1.0
            # SimpleITK expects (width, height, depth) = (dim2, dim1, dim0) spacing.
            spacing_3d = (spacing2, spacing1, spacing0)

        registered, transforms = _register_3dt(
            arr,
            reference_time=reference_time,
            n_jobs=n_jobs,
            spacing=spacing_3d,
            allow_rotation=allow_rotation,
            rotation_penalty=rotation_penalty,
        )

        # Use full 3D volume as reference for motion parameters.
        ref_img = sitk.GetImageFromArray(arr[reference_time])
        if spacing_3d is not None:
            ref_img.SetSpacing(spacing_3d)
        time_coords = (
            data_moved.coords["time"].values if "time" in data_moved.coords else None
        )
        motion_df = create_motion_dataframe(
            transforms, ref_img, time_coords=time_coords
        )
    else:
        raise ValueError(f"Expected 3D or 4D data, got {arr.ndim}D")

    result = xr.DataArray(
        registered,
        coords=data_moved.coords,
        dims=data_moved.dims,
        attrs=data.attrs.copy(),
    )

    result.attrs["registration"] = "volumewise"
    result.attrs["reference_time"] = reference_time
    result.attrs["motion_params"] = motion_df

    if squeeze_dims:
        for dim in squeeze_dims:
            result = result.expand_dims({dim: squeezed_coords[dim]})

    return result.transpose(*data.dims)


def _register_2d_frame(
    fixed_image: "sitk.Image",
    moving_image: "sitk.Image",
    allow_rotation: bool = False,
    rotation_penalty: float = 100.0,
) -> tuple["sitk.Image", "sitk.Transform"]:
    """Register a single 2D frame to a reference."""
    import SimpleITK as sitk

    registration = sitk.ImageRegistrationMethod()

    # Use normalized correlation - more stable for noisy data.
    registration.SetMetricAsCorrelation()
    registration.SetInterpolator(sitk.sitkLinear)

    if allow_rotation:
        # Conservative settings to avoid spurious rotations.
        # Based on RABIES register_slice function parameters.
        registration.SetOptimizerAsGradientDescent(
            learningRate=0.1,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        # Use scales to penalize rotation relative to translation.
        # Rotation parameter gets higher scale = smaller steps.
        registration.SetOptimizerScales(
            [rotation_penalty, 1.0, 1.0]
        )  # [rotation, tx, ty]
    else:
        # Translation-only settings (same as RABIES register_slice).
        registration.SetOptimizerAsGradientDescent(
            learningRate=0.5,
            numberOfIterations=100,
            convergenceMinimumValue=1e-7,
            convergenceWindowSize=10,
        )

    # Adjust multi-resolution settings based on image size.
    # ITK's Gaussian smoothing requires at least 4 pixels per dimension.
    min_size = min(fixed_image.GetSize())
    if min_size < 8:
        # Too small for multi-resolution - use single level with no smoothing.
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    elif min_size < 20:
        # Can only use small shrink factors and minimal smoothing.
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    else:
        # Full multi-resolution with smoothing.
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    if allow_rotation:
        # Use Euler2DTransform for translation + rotation.
        initial_transform = sitk.Euler2DTransform()
        # Set center of rotation to image center.
        center = fixed_image.TransformContinuousIndexToPhysicalPoint(
            [(sz - 1) / 2.0 for sz in fixed_image.GetSize()]
        )
        initial_transform.SetCenter(center)
    else:
        # Use TranslationTransform for translation only.
        initial_transform = sitk.TranslationTransform(2)

    registration.SetInitialTransform(initial_transform, inPlace=True)

    final_transform = registration.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )

    resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    return resampled, final_transform


def _register_3d_volume(
    fixed_image: "sitk.Image",
    moving_image: "sitk.Image",
    allow_rotation: bool = False,
    rotation_penalty: float = 100.0,
) -> tuple["sitk.Image", "sitk.Transform"]:
    """Register a single 3D volume to a reference."""
    import SimpleITK as sitk

    registration = sitk.ImageRegistrationMethod()

    # Use normalized correlation - more stable for noisy data.
    registration.SetMetricAsCorrelation()
    registration.SetInterpolator(sitk.sitkLinear)

    if allow_rotation:
        # Conservative settings to avoid spurious rotations.
        registration.SetOptimizerAsGradientDescent(
            learningRate=0.1,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        # Use scales to penalize rotation relative to translation.
        # For Euler3DTransform: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z].
        registration.SetOptimizerScales(
            [rotation_penalty, rotation_penalty, rotation_penalty, 1.0, 1.0, 1.0]
        )
    else:
        # Translation-only settings.
        registration.SetOptimizerAsGradientDescent(
            learningRate=0.5,
            numberOfIterations=100,
            convergenceMinimumValue=1e-7,
            convergenceWindowSize=10,
        )

    # Adjust multi-resolution settings based on image size.
    min_size = min(fixed_image.GetSize())
    if min_size < 8:
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    elif min_size < 20:
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    else:
        # Full multi-resolution with smoothing.
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    if allow_rotation:
        # Use Euler3DTransform for translation + rotation.
        initial_transform = sitk.Euler3DTransform()
        # Set center of rotation to volume center.
        center = fixed_image.TransformContinuousIndexToPhysicalPoint(
            [(sz - 1) / 2.0 for sz in fixed_image.GetSize()]
        )
        initial_transform.SetCenter(center)
    else:
        # Use TranslationTransform for translation only.
        initial_transform = sitk.TranslationTransform(3)

    registration.SetInitialTransform(initial_transform, inPlace=True)

    final_transform = registration.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )

    resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    return resampled, final_transform


def _register_single_frame(
    frame_data: np.ndarray,
    ref_data: np.ndarray,
    spacing: tuple[float, float] | None = None,
    allow_rotation: bool = False,
    rotation_penalty: float = 100.0,
) -> tuple[np.ndarray, "sitk.Transform"]:
    """Register a single frame (for parallel processing)."""
    import SimpleITK as sitk

    ref_img = sitk.GetImageFromArray(ref_data)
    if spacing is not None:
        ref_img.SetSpacing(spacing)

    moving = sitk.GetImageFromArray(frame_data)
    moving.CopyInformation(ref_img)
    registered, transform = _register_2d_frame(
        ref_img, moving, allow_rotation, rotation_penalty
    )

    return (sitk.GetArrayFromImage(registered), transform)


def _register_2dt(
    arr: np.ndarray,
    reference_time: int,
    n_jobs: int,
    spacing: tuple[float, float] | None = None,
    allow_rotation: bool = False,
    rotation_penalty: float = 100.0,
) -> tuple[np.ndarray, list["sitk.Transform"]]:
    """Register a 2D+t recording."""
    from joblib import Parallel, delayed
    from joblib_progress import joblib_progress

    output = np.zeros_like(arr)
    ref_data = arr[reference_time]
    n_frames = len(arr)

    with joblib_progress("Registering frames...", total=n_frames):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_register_single_frame)(
                frame,
                ref_data,
                spacing,
                allow_rotation,
                rotation_penalty,
            )
            for frame in arr
        )

    transforms: list["sitk.Transform"] = []
    for t, (registered_frame, transform) in enumerate(results):
        output[t] = registered_frame
        transforms.append(transform)

    return output, transforms


def _register_single_volume(
    volume_data: np.ndarray,
    ref_data: np.ndarray,
    spacing: tuple[float, float, float] | None = None,
    allow_rotation: bool = False,
    rotation_penalty: float = 100.0,
) -> tuple[np.ndarray, "sitk.Transform"]:
    """Register a single 3D volume (for parallel processing)."""
    import SimpleITK as sitk

    ref_img = sitk.GetImageFromArray(ref_data)
    if spacing is not None:
        ref_img.SetSpacing(spacing)

    moving = sitk.GetImageFromArray(volume_data)
    moving.CopyInformation(ref_img)
    registered, transform = _register_3d_volume(
        ref_img, moving, allow_rotation, rotation_penalty
    )

    return (sitk.GetArrayFromImage(registered), transform)


def _register_3dt(
    arr: np.ndarray,
    reference_time: int,
    n_jobs: int,
    spacing: tuple[float, float, float] | None = None,
    allow_rotation: bool = False,
    rotation_penalty: float = 100.0,
) -> tuple[np.ndarray, list["sitk.Transform"]]:
    """Register a 3D+t recording."""
    from joblib import Parallel, delayed
    from joblib_progress import joblib_progress

    output = np.zeros_like(arr)
    ref_data = arr[reference_time]
    n_frames = len(arr)

    # Use joblib for parallel/serial processing.
    with joblib_progress("Registering volumes...", total=n_frames):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_register_single_volume)(
                volume,
                ref_data,
                spacing,
                allow_rotation,
                rotation_penalty,
            )
            for volume in arr
        )

    transforms: list["sitk.Transform"] = []
    for t, (registered_volume, transform) in enumerate(results):
        output[t] = registered_volume
        transforms.append(transform)

    return output, transforms
