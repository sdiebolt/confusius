"""Motion parameter estimation and framewise displacement computation."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    import SimpleITK as sitk

__all__ = [
    "extract_motion_parameters",
    "compute_framewise_displacement",
    "create_motion_dataframe",
]


def extract_motion_parameters(
    transforms: "list[sitk.Transform]",
) -> NDArray[np.floating]:
    """Extract motion parameters from SimpleITK transforms.

    For 2D rigid transforms, extracts: ``[rotation, translation_x, translation_y]``.
    For 3D rigid transforms, extracts:
    ``[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]``.

    Parameters
    ----------
    transforms : list[SimpleITK.Transform]
        List of rigid transforms from registration.

    Returns
    -------
    (n_frames, n_params) numpy.ndarray
        Motion parameters array.

        - For 2D: ``n_params = 3 (rotation, tx, ty)``.
        - For 3D: ``n_params = 6 (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)``.
    """
    import SimpleITK as sitk

    params_list = []

    for transform in transforms:
        if transform.GetName() == "CompositeTransform":
            # We need to cast the transform to access composite-specific methods.
            composite = sitk.CompositeTransform(transform)
            if composite.GetNumberOfTransforms() > 0:
                # Get the first (and usually only) transform.
                actual_transform = composite.GetNthTransform(0)
                params = list(actual_transform.GetParameters())
            else:
                params = []
        else:
            params = list(transform.GetParameters())

        if len(params) == 0 or transform.GetName() == "IdentityTransform":
            dimension = transform.GetDimension()
            if dimension == 2:
                params = [0.0, 0.0, 0.0]  # [rotation, tx, ty]
            else:  # dimension == 3
                params = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]  # [rot_x, rot_y, rot_z, tx, ty, tz]
        elif len(params) == 2:
            # 2D TranslationTransform: [tx, ty] -> add rotation=0.
            params = [0.0, float(params[0]), float(params[1])]
        elif len(params) == 3 and transform.GetDimension() == 2:
            # 2D Euler2DTransform: [rotation, tx, ty] - already correct.
            params = [float(p) for p in params]
        elif len(params) == 3 and transform.GetDimension() == 3:
            # 3D TranslationTransform: [tx, ty, tz] -> add rotations=0.
            params = [
                0.0,
                0.0,
                0.0,
                float(params[0]),
                float(params[1]),
                float(params[2]),
            ]
        elif len(params) == 6:
            # 3D Euler3DTransform: [rot_x, rot_y, rot_z, tx, ty, tz] - already correct.
            params = [float(p) for p in params]
        else:
            msg = f"Unexpected parameter count {len(params)} for {transform.GetName()}"
            raise ValueError(msg)

        params_list.append(params)

    return np.array(params_list)


def compute_framewise_displacement(
    transforms: "list[sitk.Transform]",
    reference_image: "sitk.Image",
    mask: NDArray[np.bool_] | None = None,
) -> dict[str, NDArray[np.floating]]:
    """Compute framewise displacement from transforms.

    Framewise displacement measures how much voxels move between consecutive
    frames after registration. For each voxel, we compute the Euclidean distance
    between its position at frame t and frame t+1 after applying transforms.

    Parameters
    ----------
    transforms : list[SimpleITK.Transform]
        List of transforms, one per frame.
    reference_image : SimpleITK.Image
        Reference image defining the physical space.
    mask : numpy.ndarray, optional
        Boolean mask indicating which voxels to include. If not provided, uses all
        voxels.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"mean_fd"``: Mean framewise displacement per frame.
        - ``"max_fd"``: Maximum framewise displacement per frame.
        - ``"rms_fd"``: RMS framewise displacement per frame.
    """
    n_frames = len(transforms)
    spacing = reference_image.GetSpacing()
    size = reference_image.GetSize()

    if len(size) == 2:
        x_coords = np.arange(size[0]) * spacing[0]
        y_coords = np.arange(size[1]) * spacing[1]
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    elif len(size) == 3:
        x_coords = np.arange(size[0]) * spacing[0]
        y_coords = np.arange(size[1]) * spacing[1]
        z_coords = np.arange(size[2]) * spacing[2]
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    else:
        raise ValueError(f"Only 2D and 3D images supported, got {len(size)}D")

    if mask is not None:
        mask_flat = mask.ravel()
        points = points[mask_flat]

    mean_fd = np.zeros(n_frames)
    max_fd = np.zeros(n_frames)
    rms_fd = np.zeros(n_frames)

    for t in range(n_frames - 1):
        points_t = np.array([transforms[t].TransformPoint(tuple(p)) for p in points])
        points_t1 = np.array(
            [transforms[t + 1].TransformPoint(tuple(p)) for p in points]
        )

        # Displacements are computed using the Euclidean distance between points at t
        # and t+1.
        displacements = np.linalg.norm(points_t1 - points_t, axis=1)

        mean_fd[t] = np.mean(displacements)
        max_fd[t] = np.max(displacements)
        rms_fd[t] = np.sqrt(np.mean(displacements**2))

    # Last frame has no next frame, so FD is 0.
    mean_fd[-1] = 0.0
    max_fd[-1] = 0.0
    rms_fd[-1] = 0.0

    return {
        "mean_fd": mean_fd,
        "max_fd": max_fd,
        "rms_fd": rms_fd,
    }


def create_motion_dataframe(
    transforms: "list[sitk.Transform]",
    reference_image: "sitk.Image",
    mask: NDArray[np.bool_] | None = None,
    time_coords: NDArray[np.floating] | None = None,
) -> "pd.DataFrame":
    """Create a DataFrame with motion parameters and framewise displacement.

    Parameters
    ----------
    transforms : list[SimpleITK.Transform]
        List of transforms from registration.
    reference_image : SimpleITK.Image
        Reference image defining physical space.
    mask : numpy.ndarray, optional
        Boolean mask for FD computation.
    time_coords : numpy.ndarray, optional
        Time coordinates for each frame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        For 2D:

        - ``rotation``: Rotation angle in radians.
        - ``trans_x``: Translation in x (mm).
        - ``trans_y``: Translation in y (mm).

        For 3D:

        - ``rot_x, rot_y, rot_z``: Rotation angles in radians.
        - ``trans_x, trans_y, trans_z``: Translations (mm).

        Both:
        - ``mean_fd``: Mean framewise displacement (mm).
        - ``max_fd``: Maximum framewise displacement (mm).
        - ``rms_fd``: RMS framewise displacement (mm).
    """
    import pandas as pd

    params = extract_motion_parameters(transforms)

    fd_dict = compute_framewise_displacement(transforms, reference_image, mask)

    if params.shape[1] == 3:
        df = pd.DataFrame(
            {
                "rotation": params[:, 0],
                "trans_x": params[:, 1],
                "trans_y": params[:, 2],
                "mean_fd": fd_dict["mean_fd"],
                "max_fd": fd_dict["max_fd"],
                "rms_fd": fd_dict["rms_fd"],
            }
        )
    else:
        df = pd.DataFrame(
            {
                "rot_x": params[:, 0],
                "rot_y": params[:, 1],
                "rot_z": params[:, 2],
                "trans_x": params[:, 3],
                "trans_y": params[:, 4],
                "trans_z": params[:, 5],
                "mean_fd": fd_dict["mean_fd"],
                "max_fd": fd_dict["max_fd"],
                "rms_fd": fd_dict["rms_fd"],
            }
        )

    if time_coords is not None:
        df.index = time_coords
        df.index.name = "time"
    else:
        df.index.name = "frame"

    return df
