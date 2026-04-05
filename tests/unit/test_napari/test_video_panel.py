"""Tests for the VideoPanel widget.

_VideoArray and _compute_spatial_params are tested as pure-logic units.
Integration tests for dimension reorder, frame step, and time overlay use
real napari viewers via `make_napari_viewer`.

See https://napari.org/dev/plugins/testing_and_publishing/test.html
"""

from __future__ import annotations

import numpy as np
import pytest

from confusius._napari._video._video_panel import VideoPanel, _VideoArray
from confusius.plotting import plot_napari

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeVideo:
    """Minimal VideoReaderNP stub."""

    def __init__(self, n_frames=20, h=48, w=64, rgb=False):
        c = (3,) if rgb else ()
        self.shape = (n_frames, h, w, *c)
        self.number_of_frames = n_frames
        self.frame_rate = 30.0
        self.frame_shape = (h, w, 3) if rgb else (h, w)
        self._frames = np.random.default_rng(0).integers(
            0, 255, size=self.shape, dtype=np.uint8
        )

    def __getitem__(self, idx):
        return self._frames[idx]


def _inject_fake_video(panel, fv, *, frame_step=1):
    """Inject a _FakeVideo into an already-constructed VideoPanel.

    Sets all internal video state so that ``_rebuild_video_layer`` can
    run without any actual file I/O.
    """
    panel._video = fv
    panel._frame_dtype = np.uint8
    panel._frame_shape = fv.frame_shape
    panel._video_h = fv.frame_shape[0]
    panel._video_w = fv.frame_shape[1]
    panel._is_rgb = len(fv.frame_shape) == 3 and fv.frame_shape[2] in (3, 4)
    panel._fps = fv.frame_rate
    panel._video_name = "Video: test"
    panel._step_spin.setValue(frame_step)


# ---------------------------------------------------------------------------
# _VideoArray
# ---------------------------------------------------------------------------


class TestVideoArray:
    def test_integer_index(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        np.testing.assert_array_equal(arr[2], fv[2])

    def test_slice_index(self):
        fv = _FakeVideo(n_frames=10, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        frames = arr[2:5]
        assert frames.shape == (3, 4, 6)
        np.testing.assert_array_equal(frames[0], fv[2])

    def test_tuple_index_with_spatial_slice(self):
        fv = _FakeVideo(n_frames=5, h=8, w=10)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(8, 10))
        result = arr[1, :4, :5]
        np.testing.assert_array_equal(result, fv[1][:4, :5])

    def test_shape_dtype_len(self):
        fv = _FakeVideo(n_frames=8, h=16, w=32)
        arr = _VideoArray(fv, dtype=np.float32, frame_shape=(16, 32))
        assert arr.shape == (8, 16, 32)
        assert arr.dtype == np.float32
        assert len(arr) == 8
        assert arr.ndim == 3

    def test_unsupported_index_raises(self):
        fv = _FakeVideo(n_frames=3, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        with pytest.raises(IndexError, match="Unsupported time index type"):
            arr[[0, 1]]

    def test_dimension_padding(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6), n_pad=1)
        assert arr.shape == (5, 1, 4, 6)
        # Integer index on pad dim should return the frame.
        np.testing.assert_array_equal(arr[2, 0], fv[2])

    def test_rgb_shape(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6, rgb=True)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6, 3))
        assert arr.shape == (5, 4, 6, 3)
        np.testing.assert_array_equal(arr[1], fv[1])

    def test_rgb_with_padding(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6, rgb=True)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6, 3), n_pad=1)
        assert arr.shape == (5, 1, 4, 6, 3)
        np.testing.assert_array_equal(arr[2, 0], fv[2])

    def test_empty_slice(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        result = arr[3:3]
        assert result.shape == (0, 4, 6)

    def test_explicit_h_before_w(self):
        """VideoArray with H at dim 1, W at dim 3 (e.g. displaying z x)."""
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(
            fv,
            dtype=np.uint8,
            frame_shape=(4, 6),
            n_pad=1,
            time_dim=0,
            h_dim=1,
            w_dim=3,
        )
        assert arr.shape == (5, 4, 1, 6)
        result = arr[0, :, 0, :]
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(result, fv[0])

    def test_explicit_w_before_h(self):
        """VideoArray with W at dim 1, H at dim 3 (e.g. displaying x z)."""
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(
            fv,
            dtype=np.uint8,
            frame_shape=(4, 6),
            n_pad=1,
            time_dim=0,
            h_dim=3,
            w_dim=1,
        )
        assert arr.shape == (5, 6, 1, 4)
        result = arr[0, :, 0, :]
        assert result.shape == (6, 4)
        np.testing.assert_array_equal(result, fv[0].T)

    def test_explicit_dims_rgb_with_transpose(self):
        """RGB video with W before H in layout."""
        fv = _FakeVideo(n_frames=5, h=4, w=6, rgb=True)
        arr = _VideoArray(
            fv,
            dtype=np.uint8,
            frame_shape=(4, 6, 3),
            n_pad=1,
            time_dim=0,
            h_dim=3,
            w_dim=1,
        )
        assert arr.shape == (5, 6, 1, 4, 3)
        result = arr[0, :, 0, :]
        assert result.shape == (6, 4, 3)
        np.testing.assert_array_equal(result, np.swapaxes(fv[0], 0, 1))


# ---------------------------------------------------------------------------
# _compute_spatial_params
# ---------------------------------------------------------------------------


class TestComputeSpatialParams:
    def test_returns_defaults_without_ref_layer(self, panel):
        panel._ref_layer = None
        assert panel._compute_spatial_params(2) == (1.0, 0.0)

    def test_video_scaled_to_scan_height(self, panel, sample_4d_volume):
        # vertical_dim=2 -> "y" coord.
        scale, ty = panel._compute_spatial_params(2)

        y_coords = sample_4d_volume.coords["y"].values.astype(np.float64)
        y_min, y_max = y_coords.min(), y_coords.max()
        y_step = float(np.median(np.diff(y_coords)))
        expected_scale = (y_max - y_min + abs(y_step)) / 48
        assert scale == pytest.approx(expected_scale)

        center_y = (y_min + y_max) / 2
        expected_ty = center_y - scale * (48 - 1) / 2
        assert ty == pytest.approx(expected_ty)

    def test_returns_defaults_when_coords_missing(self, panel):
        panel._axis_labels = ("time", "z", "foo", "bar")
        assert panel._compute_spatial_params(2) == (1.0, 0.0)

    def test_spatial_params_z_vertical(self, panel, sample_4d_volume):
        """When z is the vertical axis, scale uses z-coordinates."""
        scale, tz = panel._compute_spatial_params(1)  # dim 1 = "z".

        z_coords = sample_4d_volume.coords["z"].values.astype(np.float64)
        z_min, z_max = z_coords.min(), z_coords.max()
        z_step = float(np.median(np.diff(z_coords)))
        expected_scale = (z_max - z_min + abs(z_step)) / 48
        assert scale == pytest.approx(expected_scale)


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def viewer(make_napari_viewer):
    """Real napari viewer for integration tests."""
    return make_napari_viewer()


@pytest.fixture
def fusi_layer(viewer, sample_4d_volume):
    """Add the sample 4D volume to the viewer as a real image layer.

    Returns the napari layer, which carries axis_labels, scale, translate,
    and xarray metadata -- the same state a real fUSI scan would have.
    """
    _, layer = plot_napari(
        sample_4d_volume,
        viewer=viewer,
        show_colorbar=False,
        show_scale_bar=False,
    )
    return layer


@pytest.fixture
def panel(viewer, fusi_layer, sample_4d_volume):
    """VideoPanel wired to a real viewer that already has a fUSI layer."""
    p = VideoPanel(viewer)

    # Set reference layer and scan metadata (normally done by _load).
    p._ref_layer = fusi_layer
    p._axis_labels = tuple(sample_4d_volume.dims)
    p._units = list(getattr(fusi_layer, "units", [None] * fusi_layer.ndim))
    p._n_pad = max(fusi_layer.ndim - 3, 0)
    p._fusi_time_idx = list(sample_4d_volume.dims).index("time")
    p._video_h = 48
    order = viewer.dims.order
    p._displayed_dims = (order[-2], order[-1])

    return p


@pytest.fixture
def loaded_panel(panel, viewer):
    """Panel with a fake RGB video injected and the video layer built.

    This is the closest to "user clicked Load" without actual file I/O.
    """
    fv = _FakeVideo(n_frames=10, h=48, w=64, rgb=True)
    _inject_fake_video(panel, fv)
    panel._rebuild_video_layer()

    # Connect the dimension order guard (normally done at end of _load).
    viewer.dims.events.order.connect(panel._on_dim_order_changed)
    panel._guarding_order = True

    return panel


# ---------------------------------------------------------------------------
# Dimension reorder integration (real viewer)
# ---------------------------------------------------------------------------


class TestDimReorder:
    def test_rebuild_on_displayed_dims_change(self, loaded_panel, viewer):
        """Changing displayed dims removes old layer and adds new one."""
        old_name = loaded_panel._video_layer.name
        assert old_name in [ly.name for ly in viewer.layers]

        # Change displayed dims: swap y and z in the display axes.
        # Default order is (0, 1, 2, 3); displayed are last 2: dims 2, 3.
        # New order (0, 2, 3, 1): displayed dims 3, 1.
        viewer.dims.order = (0, 2, 3, 1)

        # Video layer should still exist (rebuilt), but displayed_dims updated.
        assert loaded_panel._displayed_dims == (3, 1)
        assert loaded_panel._video_layer is not None
        assert loaded_panel._video_layer.name in [ly.name for ly in viewer.layers]

    def test_no_rebuild_when_displayed_dims_unchanged(self, loaded_panel, viewer):
        """Reordering only slider dims (not displayed) keeps the same layer."""
        old_layer = loaded_panel._video_layer

        # Swap slider dims 0 and 1, keep displayed dims 2, 3 the same.
        viewer.dims.order = (1, 0, 2, 3)

        # Same layer object, no rebuild.
        assert loaded_panel._video_layer is old_layer
        assert loaded_panel._displayed_dims == (2, 3)

    def test_time_in_display_gets_corrected(self, loaded_panel, viewer):
        """If time ends up in displayed dims, the order is fixed."""
        # Try to put time (dim 0) in a displayed position.
        viewer.dims.order = (1, 2, 3, 0)

        # The guard should have corrected the order so time is not displayed.
        # Time (dim 0) should be back in a slider position.
        order = viewer.dims.order
        displayed = (order[-2], order[-1])
        assert 0 not in displayed, (
            f"Time dim 0 should not be displayed, but order is {order}"
        )

    def test_rebuild_creates_correct_video_array_shape(self, loaded_panel, viewer):
        """After dim change, video layer has H and W at new positions."""
        # Change to display dims 3, 1 (x, z).
        viewer.dims.order = (0, 2, 3, 1)

        layer = loaded_panel._video_layer
        # Shape should be (10, 64, 1, 48, 3): time, W, pad, H, C.
        assert layer.data.shape == (10, 64, 1, 48, 3)


# ---------------------------------------------------------------------------
# Frame step integration (real viewer)
# ---------------------------------------------------------------------------


class TestFrameStep:
    def test_frame_step_reduces_time_shape(self):
        """With step=3, time dimension is len(range(0, n_frames, step))."""
        fv = _FakeVideo(n_frames=12, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6), step=3)
        # 12 frames with step 3: indices 0, 3, 6, 9 -> 4 logical frames.
        assert arr.shape == (4, 4, 6)
        # Logical frame 0 -> physical frame 0.
        np.testing.assert_array_equal(arr[0], fv[0])
        # Logical frame 2 -> physical frame 6.
        np.testing.assert_array_equal(arr[2], fv[6])

    def test_rebuild_time_scale_reflects_frame_step(self, loaded_panel, viewer):
        """Time scale must be frame_step / fps."""
        loaded_panel._step_spin.setValue(3)
        loaded_panel._rebuild_video_layer()

        layer = loaded_panel._video_layer
        time_idx = loaded_panel._fusi_time_idx
        expected = 3.0 / loaded_panel._fps
        np.testing.assert_allclose(layer.scale[time_idx], expected)

    def test_frame_step_change_rebuilds_with_smaller_time_dim(
        self, loaded_panel, viewer
    ):
        """Changing frame step rebuilds the video layer with fewer time frames."""
        old_layer = loaded_panel._video_layer

        # Trigger frame step change.
        loaded_panel._step_spin.setValue(3)
        loaded_panel._on_frame_step_changed(3)

        new_layer = loaded_panel._video_layer
        # Layer was rebuilt (different object or at least different data).
        assert new_layer is not old_layer or new_layer.data is not old_layer.data

        # New VideoArray has reduced time dimension.
        n_frames = loaded_panel._video.number_of_frames  # 10
        expected_time = len(range(0, n_frames, 3))  # 4
        time_idx = loaded_panel._fusi_time_idx
        assert new_layer.data.shape[time_idx] == expected_time

    def test_frame_step_change_preserves_time_position(self, loaded_panel, viewer):
        """Changing frame step restores the slider to the closest time."""
        time_idx = loaded_panel._fusi_time_idx

        # Move the slider to a specific world time.
        viewer.dims.set_point(time_idx, 0.1)
        world_time_before = float(viewer.dims.point[time_idx])

        loaded_panel._step_spin.setValue(3)
        loaded_panel._on_frame_step_changed(3)

        # Should restore world time; napari snaps to nearest valid step.
        world_time_after = float(viewer.dims.point[time_idx])
        np.testing.assert_allclose(world_time_after, world_time_before, atol=0.2)


# ---------------------------------------------------------------------------
# Time overlay syncs with video fps (real viewer)
# ---------------------------------------------------------------------------


class TestTimeOverlayVideoSync:
    def test_overlay_uses_dims_point_for_video_layer(self, viewer, fusi_layer):
        """For a video layer (no xarray), the overlay uses dims.point which
        is correct because the layer's time scale encodes the frame step.

        With scale = frame_step / fps, napari's world coordinate at
        ``dims.point[time_idx]`` is already in physical seconds.
        """
        from confusius._napari._time_overlay import _TimeOverlay

        overlay = _TimeOverlay(viewer)
        overlay.check()
        assert overlay._active

        # Add a video-like layer with fps metadata but no xarray.
        video_data = np.zeros((10, 4, 6, 8), dtype=np.uint8)
        fps = 50.0
        time_scale = 1.0 / fps
        video_layer = viewer.add_image(
            video_data,
            name="Video: test",
            scale=(time_scale, 1.0, 1.0, 1.0),
            axis_labels=("time", "z", "y", "x"),
            metadata={"fps": fps, "time_units": "s"},
        )

        # Select the video layer so the overlay uses it as reference.
        viewer.layers.selection = {video_layer}
        assert overlay._ref_layer is video_layer

        # Move to step 3; world time = 3 * time_scale = 0.06s.
        time_idx = overlay._time_idx
        viewer.dims.set_current_step(time_idx, 3)

        # The overlay should fall back to dims.point (no xarray).
        expected_time = float(viewer.dims.point[time_idx])
        expected_text = f"{expected_time:.2f} s"
        assert viewer.text_overlay.text == expected_text
