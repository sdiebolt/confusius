"""Shared fixtures for registration tests."""

import numpy as np
import pytest
import SimpleITK as sitk
import xarray as xr


@pytest.fixture
def sample_2d_image():
    """Create a simple 2D test image with a recognizable pattern."""
    # Create a 32x32 image with a bright square in the center.
    img = np.zeros((32, 32), dtype=np.float32)
    img[12:20, 12:20] = 100.0
    return img


@pytest.fixture
def sample_3d_volume():
    """Create a simple 3D test volume with a recognizable pattern."""
    # Create a 16x16x16 volume with a bright cube in the center.
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[6:10, 6:10, 6:10] = 100.0
    return vol


@pytest.fixture
def sample_2dt_dataarray(sample_2d_image):
    """Create a 2D+time xarray DataArray for registration tests."""
    n_frames = 5
    data = np.stack([sample_2d_image] * n_frames, axis=0)
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": np.arange(n_frames) * 0.1,
            "y": np.arange(32) * 0.1,
            "x": np.arange(32) * 0.1,
        },
    )


@pytest.fixture
def sample_3dt_dataarray(sample_3d_volume):
    """Create a 3D+time xarray DataArray for registration tests."""
    n_frames = 3
    data = np.stack([sample_3d_volume] * n_frames, axis=0)
    return xr.DataArray(
        data,
        dims=("time", "z", "y", "x"),
        coords={
            "time": np.arange(n_frames) * 0.1,
            "z": np.arange(16) * 0.2,
            "y": np.arange(16) * 0.1,
            "x": np.arange(16) * 0.1,
        },
    )


@pytest.fixture
def identity_transform_2d():
    """Create a 2D identity transform."""
    return sitk.TranslationTransform(2)


@pytest.fixture
def translation_transform_2d():
    """Create a 2D translation transform with known offset."""
    t = sitk.TranslationTransform(2)
    t.SetOffset((2.0, 3.0))  # tx=2, ty=3
    return t


@pytest.fixture
def euler_transform_2d():
    """Create a 2D Euler transform with known rotation and translation."""
    t = sitk.Euler2DTransform()
    t.SetAngle(0.1)  # ~5.7 degrees
    t.SetTranslation((1.5, 2.5))
    return t


@pytest.fixture
def translation_transform_3d():
    """Create a 3D translation transform with known offset."""
    t = sitk.TranslationTransform(3)
    t.SetOffset((1.0, 2.0, 3.0))
    return t


@pytest.fixture
def euler_transform_3d():
    """Create a 3D Euler transform with known rotation and translation."""
    t = sitk.Euler3DTransform()
    t.SetRotation(0.05, 0.1, 0.15)  # rot_x, rot_y, rot_z
    t.SetTranslation((1.0, 2.0, 3.0))
    return t


@pytest.fixture
def reference_image_2d():
    """Create a 2D reference image for FD computation."""
    img = sitk.Image(10, 10, sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0))
    return img


@pytest.fixture
def reference_image_3d():
    """Create a 3D reference image for FD computation."""
    img = sitk.Image(8, 8, 8, sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img
