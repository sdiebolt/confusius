"""Unit tests for motion parameter estimation functions."""

import numpy as np
import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose, assert_array_equal

from confusius.registration.motion import (
    compute_framewise_displacement,
    create_motion_dataframe,
    extract_motion_parameters,
)


class TestExtractMotionParameters:
    """Tests for extract_motion_parameters function."""

    def test_2d_translation_only(self, translation_transform_2d):
        """2D translation transform extracts [0, tx, ty]."""
        transforms = [translation_transform_2d]
        params = extract_motion_parameters(transforms)

        assert params.shape == (1, 3)
        # Translation transform: rotation=0, then tx, ty.
        assert_allclose(params[0], [0.0, 2.0, 3.0])

    def test_2d_euler_transform(self, euler_transform_2d):
        """2D Euler transform extracts [rotation, tx, ty]."""
        transforms = [euler_transform_2d]
        params = extract_motion_parameters(transforms)

        assert params.shape == (1, 3)
        assert_allclose(params[0], [0.1, 1.5, 2.5])

    def test_3d_translation_only(self, translation_transform_3d):
        """3D translation transform extracts [0, 0, 0, tx, ty, tz]."""
        transforms = [translation_transform_3d]
        params = extract_motion_parameters(transforms)

        assert params.shape == (1, 6)
        assert_allclose(params[0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0])

    def test_3d_euler_transform(self, euler_transform_3d):
        """3D Euler transform extracts [rot_x, rot_y, rot_z, tx, ty, tz]."""
        transforms = [euler_transform_3d]
        params = extract_motion_parameters(transforms)

        assert params.shape == (1, 6)
        assert_allclose(params[0], [0.05, 0.1, 0.15, 1.0, 2.0, 3.0])

    def test_multiple_transforms(self, identity_transform_2d, translation_transform_2d):
        """Multiple transforms return stacked parameters."""
        transforms = [identity_transform_2d, translation_transform_2d]
        params = extract_motion_parameters(transforms)

        assert params.shape == (2, 3)
        assert_allclose(params[0], [0.0, 0.0, 0.0])
        assert_allclose(params[1], [0.0, 2.0, 3.0])

    @pytest.mark.parametrize(
        ("dim", "expected_shape", "expected_params"),
        [
            (2, (1, 3), [0.0, 0.0, 0.0]),
            (3, (1, 6), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
    )
    def test_identity_transform(self, dim, expected_shape, expected_params):
        """Identity transform returns zeros for 2D and 3D."""
        t = sitk.Transform(dim, sitk.sitkIdentity)
        params = extract_motion_parameters([t])

        assert params.shape == expected_shape
        assert_allclose(params[0], expected_params)

    def test_composite_transform_unwraps(self, translation_transform_2d):
        """Composite transform is unwrapped to get inner transform parameters."""
        composite = sitk.CompositeTransform(2)
        composite.AddTransform(translation_transform_2d)

        params = extract_motion_parameters([composite])
        assert params.shape == (1, 3)
        assert_allclose(params[0], [0.0, 2.0, 3.0])


class TestComputeFramewiseDisplacement:
    """Tests for compute_framewise_displacement function."""

    def test_identical_transforms_zero_fd(self, reference_image_2d):
        """Identical transforms produce zero framewise displacement."""
        t1 = sitk.TranslationTransform(2)
        t2 = sitk.TranslationTransform(2)
        transforms = [t1, t2]

        fd = compute_framewise_displacement(transforms, reference_image_2d)

        assert_allclose(fd["mean_fd"], [0.0, 0.0])
        assert_allclose(fd["max_fd"], [0.0, 0.0])
        assert_allclose(fd["rms_fd"], [0.0, 0.0])

    def test_known_translation_displacement_2d(self, reference_image_2d):
        """Known translation produces correct FD."""
        t1 = sitk.TranslationTransform(2)
        t2 = sitk.TranslationTransform(2)
        t2.SetOffset((3.0, 4.0))  # Distance = 5.0
        transforms = [t1, t2]

        fd = compute_framewise_displacement(transforms, reference_image_2d)

        # All voxels move by same amount for pure translation.
        assert_allclose(fd["mean_fd"][0], 5.0)
        assert_allclose(fd["max_fd"][0], 5.0)
        assert_allclose(fd["rms_fd"][0], 5.0)
        # Last frame FD is always 0.
        assert fd["mean_fd"][-1] == 0.0

    def test_known_translation_displacement_3d(self, reference_image_3d):
        """3D translation produces correct FD."""
        t1 = sitk.TranslationTransform(3)
        t2 = sitk.TranslationTransform(3)
        t2.SetOffset((1.0, 2.0, 2.0))  # Distance = 3.0
        transforms = [t1, t2]

        fd = compute_framewise_displacement(transforms, reference_image_3d)

        assert_allclose(fd["mean_fd"][0], 3.0)
        assert_allclose(fd["max_fd"][0], 3.0)

    def test_with_mask(self, reference_image_2d):
        """Mask restricts FD computation to masked voxels."""
        t1 = sitk.TranslationTransform(2)
        t2 = sitk.TranslationTransform(2)
        t2.SetOffset((3.0, 4.0))
        transforms = [t1, t2]

        # Create mask for subset of voxels.
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True

        fd = compute_framewise_displacement(transforms, reference_image_2d, mask=mask)

        # Pure translation: same displacement regardless of mask.
        assert_allclose(fd["mean_fd"][0], 5.0)


class TestCreateMotionDataframe:
    """Tests for create_motion_dataframe function."""

    def test_2d_dataframe_columns(
        self, translation_transform_2d, identity_transform_2d, reference_image_2d
    ):
        """2D transforms produce DataFrame with correct columns."""
        transforms = [identity_transform_2d, translation_transform_2d]
        df = create_motion_dataframe(transforms, reference_image_2d)

        expected_cols = [
            "rotation",
            "trans_x",
            "trans_y",
            "mean_fd",
            "max_fd",
            "rms_fd",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 2

    def test_3d_dataframe_columns(self, translation_transform_3d, reference_image_3d):
        """3D transforms produce DataFrame with correct columns."""
        t_identity = sitk.TranslationTransform(3)
        transforms = [t_identity, translation_transform_3d]
        df = create_motion_dataframe(transforms, reference_image_3d)

        expected_cols = [
            "rot_x",
            "rot_y",
            "rot_z",
            "trans_x",
            "trans_y",
            "trans_z",
            "mean_fd",
            "max_fd",
            "rms_fd",
        ]
        assert list(df.columns) == expected_cols

    def test_time_coords_as_index(self, identity_transform_2d, reference_image_2d):
        """Time coordinates are used as DataFrame index."""
        transforms = [identity_transform_2d, identity_transform_2d]
        time_coords = np.array([0.0, 0.5])

        df = create_motion_dataframe(
            transforms, reference_image_2d, time_coords=time_coords
        )

        assert df.index.name == "time"
        assert_array_equal(df.index, time_coords)

    def test_no_time_coords_uses_frame_index(
        self, identity_transform_2d, reference_image_2d
    ):
        """Without time coords, index is named 'frame'."""
        transforms = [identity_transform_2d]
        df = create_motion_dataframe(transforms, reference_image_2d)

        assert df.index.name == "frame"

    def test_motion_values_correct(
        self, translation_transform_2d, identity_transform_2d, reference_image_2d
    ):
        """Motion parameter values are correctly populated."""
        transforms = [identity_transform_2d, translation_transform_2d]
        df = create_motion_dataframe(transforms, reference_image_2d)

        assert_allclose(df.loc[0, "rotation"], 0.0)
        assert_allclose(df.loc[0, "trans_x"], 0.0)
        assert_allclose(df.loc[0, "trans_y"], 0.0)

        assert_allclose(df.loc[1, "rotation"], 0.0)
        assert_allclose(df.loc[1, "trans_x"], 2.0)
        assert_allclose(df.loc[1, "trans_y"], 3.0)
