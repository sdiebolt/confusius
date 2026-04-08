"""Generate documentation images for the Visualization user guide.

Data is fetched automatically from the Nunez-Elizalde et al. (2022) fUSI-BIDS
dataset on OSF (https://osf.io/43skw/) via `confusius.datasets`.  The first run
downloads ~30 MB; subsequent runs use the local cache.

The two atlas overlay images (`napari-labels.png` and `volume-with-contours-*.png`)
require an external labelled atlas mask aligned to the recording.  Set
`ATLAS_MASK_PATH` to a Zarr or NIfTI file containing an integer-labelled mask to
generate those images; leave it as `None` to skip them.

Usage
-----
Run from the project root::

    uv run docs/images/visualization/generate.py

All images are saved to docs/images/visualization/.

Notes
-----
- napari screenshots are taken programmatically. Review them after the script
  finishes and retake manually (File > Save Screenshot in napari) if the canvas
  renders poorly (e.g., all-black).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
from napari.qt import get_qapp
from qtpy.QtCore import Qt

import confusius as cf  # noqa: F401  # Register xarray accessors.
from confusius.datasets import fetch_nunez_elizalde_2022

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Optional: path to an integer-labelled atlas mask (Zarr or NIfTI) aligned to
# the recording.  Set to a valid path to generate atlas overlay images; leave
# as None to skip those images.
# ---------------------------------------------------------------------------

ATLAS_MASK_PATH: str | None = None

# ---------------------------------------------------------------------------
# Fetch dataset and load data
# ---------------------------------------------------------------------------

print("Fetching Nunez-Elizalde 2022 dataset …")
bids_root = fetch_nunez_elizalde_2022(
    subjects=["CR020"],
    sessions=["20191121"],
    tasks=["spontaneous"],
)

_FUSI_PATH = (
    bids_root
    / "sub-CR020/ses-20191121/fusi"
    / "sub-CR020_ses-20191121_task-spontaneous_acq-slice01_pwd.nii.gz"
)
_ANGIO_PATH = (
    bids_root / "sub-CR020/ses-20191121/angio" / "sub-CR020_ses-20191121_pwd.nii.gz"
)

print("Loading power Doppler data …")
pwd = cf.load(_FUSI_PATH)
print(f"  {pwd.dims}, shape {dict(pwd.sizes)}")

print("Computing mean volume …")
mean_vol = pwd.mean("time").compute()

# Brain mask: voxels above the 40th percentile of the mean image.
# Used for the carpet plot voxel selection.
brain_mask = mean_vol > float(mean_vol.quantile(0.4))

print("Loading angiography (3D volume) …")
vol_3d = cf.load(_ANGIO_PATH).compute()
print(f"  {vol_3d.dims}, shape {dict(vol_3d.sizes)}")

# ---------------------------------------------------------------------------
# Optionally load atlas mask
# ---------------------------------------------------------------------------

atlas_mask = None
id_to_rgb: dict = {}

if ATLAS_MASK_PATH is not None:
    print("Loading atlas mask …")
    atlas_mask = cf.load(ATLAS_MASK_PATH).compute()
    n_atlas_regions = int(np.sum(np.unique(atlas_mask.values) != 0))
    print(f"  {atlas_mask.dims}, {n_atlas_regions} region(s)")

    try:
        from brainglobe_atlasapi import BrainGlobeAtlas

        atlas = BrainGlobeAtlas("allen_mouse_25um")
        id_to_rgb = {
            s["id"]: tuple(c / 255 for c in s["rgb_triplet"])
            for s in atlas.structures_list
        }
        print(f"  Built Allen atlas colormap ({len(id_to_rgb)} entries).")
    except Exception as exc:
        print(f"  Could not load Allen atlas colormap: {exc}")


def _napari_screenshot(viewer: "napari.Viewer", path: str) -> None:
    """Take a full-window napari screenshot without displaying the window.

    QWidget.grab() (used by canvas_only=False) requires the widget to have
    been shown at least once for its layout to be initialised. Setting
    WA_DontShowOnScreen before show() runs the full Qt layout pipeline
    without actually mapping the window on screen, so the tiling WM never
    sees or resizes it.
    """
    win = viewer.window._qt_window
    win.setAttribute(Qt.WA_DontShowOnScreen)
    win.show()
    win.resize(1100, 750)
    get_qapp().processEvents()
    viewer.camera.zoom *= 1.2
    get_qapp().processEvents()
    viewer.screenshot(path=path, canvas_only=False)


_SAVEFIG_KWARGS = {"dpi": 150, "bbox_inches": "tight", "transparent": True}


def _save_light_dark(plotter, stem: str) -> None:
    """Save light and dark variants of a VolumePlotter figure."""
    plotter.savefig(str(HERE / f"{stem}-light.png"), **_SAVEFIG_KWARGS)
    plotter.close()


# ---------------------------------------------------------------------------
# 1. napari — overview (mean volume)
# ---------------------------------------------------------------------------

print("\n── napari images ─────────────────────────────────────────────────────")
print("Generating napari-overview.png …")

try:
    viewer = napari.Viewer(show=False)
    mean_vol.fusi.scale.db().fusi.plot.napari(
        viewer=viewer,
        contrast_limits=(-15, 0),
        colormap="gray",
        name="Power Doppler (dB)",
    )
    _napari_screenshot(viewer, str(HERE / "napari-overview.png"))
    viewer.close()
    print("  Saved napari-overview.png")
except Exception as exc:
    print(f"  napari-overview.png failed: {exc}")

# ---------------------------------------------------------------------------
# 2. napari — power Doppler + Allen atlas Labels layer (atlas mask required)
# ---------------------------------------------------------------------------

if atlas_mask is not None:
    print("Generating napari-labels.png …")
    try:
        from napari.utils.colormaps import DirectLabelColormap

        viewer2 = napari.Viewer(show=False)
        viewer2 = mean_vol.fusi.scale.db().fusi.plot.napari(
            viewer=viewer2,
            contrast_limits=(-15, 0),
            colormap="gray",
            name="Power Doppler (dB)",
        )
        label_colormap = DirectLabelColormap(
            color_dict={
                0: np.zeros(4),
                None: np.zeros(4),
                **{
                    lbl: np.array(list(rgb) + [0.7])
                    for lbl, rgb in id_to_rgb.items()
                    if lbl > 0
                },
            }
        )
        viewer2 = atlas_mask.fusi.plot.napari(
            viewer=viewer2,
            layer_type="labels",
            colormap=label_colormap,
            name="Allen atlas",
            opacity=0.6,
        )
        _napari_screenshot(viewer2, str(HERE / "napari-labels.png"))
        viewer2.close()
        print("  Saved napari-labels.png")
    except Exception as exc:
        print(f"  napari-labels.png failed: {exc}")
else:
    print("Skipping napari-labels.png (ATLAS_MASK_PATH not set).")

# ---------------------------------------------------------------------------
# 3. napari 3D orbit GIF
# ---------------------------------------------------------------------------

print("\n── napari 3D orbit GIF ────────────────────────────────────────────────")

try:
    print("Generating napari-3d-orbit.gif …")

    viewer_orbit = napari.Viewer(show=False)
    viewer_orbit = vol_3d.fusi.plot.napari(
        viewer=viewer_orbit,
        colormap="gray",
        name="Angiography",
        dim_order=("z", "x", "y"),
    )
    viewer_orbit.dims.ndisplay = 3

    win = viewer_orbit.window._qt_window
    win.setAttribute(Qt.WA_DontShowOnScreen)
    win.show()
    win.resize(1100, 750)
    get_qapp().processEvents()

    viewer_orbit.camera.angles = (-90.0, -15.0, 270.0)
    viewer_orbit.camera.zoom *= 0.8
    get_qapp().processEvents()

    N_FRAMES = 36  # One frame every 10°, full 360° orbit.
    GIF_WIDTH = 1100
    from PIL import Image

    frames_pil = []
    for i in range(N_FRAMES):
        azimuth = 270.0 + i * (360.0 / N_FRAMES)
        viewer_orbit.camera.angles = (-90.0, -15.0, azimuth)
        get_qapp().processEvents()
        raw = viewer_orbit.screenshot(canvas_only=False)[..., :3]
        h, w = raw.shape[:2]
        scale = GIF_WIDTH / w
        frames_pil.append(
            Image.fromarray(raw).resize((GIF_WIDTH, int(h * scale)), Image.LANCZOS)
        )

    viewer_orbit.close()

    palette_src = frames_pil[0].quantize(colors=256, dither=0)
    quantized = [frame.quantize(palette=palette_src, dither=0) for frame in frames_pil]

    out_path = str(HERE / "napari-3d-orbit.gif")
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=1000 // 8,
        loop=0,
    )
    print("  Saved napari-3d-orbit.gif")
except Exception as exc:
    print(f"  napari 3D orbit GIF failed: {exc}")

# ---------------------------------------------------------------------------
# 4. Volume grid — mean volume (matplotlib)
# ---------------------------------------------------------------------------

print("\n── Matplotlib images ─────────────────────────────────────────────────")
print("Generating plot-volume-grid-light/dark.png …")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = mean_vol.fusi.scale.db().fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        vmin=-15,
        vmax=0,
        cbar_label="Power Doppler (dB)",
        black_bg=black_bg,
    )
    plotter.savefig(str(HERE / f"plot-volume-grid-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()
print("  Saved plot-volume-grid-light.png and plot-volume-grid-dark.png")

# ---------------------------------------------------------------------------
# 5. Volume grid — true 3D volume (matplotlib)
# ---------------------------------------------------------------------------

print("Generating plot-volume-3d-light/dark.png …")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter_3d = vol_3d.fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        show_colorbar=False,
        black_bg=black_bg,
    )
    plotter_3d.savefig(str(HERE / f"plot-volume-3d-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter_3d.close()
print("  Saved plot-volume-3d-light.png and plot-volume-3d-dark.png")

# ---------------------------------------------------------------------------
# 6. Volume grid — sliced 3D volume (matplotlib)
# ---------------------------------------------------------------------------

print("Generating plot-sliced-volume-3d-light/dark.png …")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter_3d = vol_3d.fusi.plot.volume(
        nrows=1,
        slice_mode="z",
        slice_coords=(1.6, 2.4, 3.2),
        cmap="inferno",
        show_colorbar=False,
        black_bg=black_bg,
    )
    plotter_3d.savefig(
        str(HERE / f"plot-sliced-volume-3d-{suffix}.png"), **_SAVEFIG_KWARGS
    )
    plotter_3d.close()
print("  Saved plot-sliced-volume-3d-light.png and plot-sliced-volume-3d-dark.png")

# ---------------------------------------------------------------------------
# 7. Volume with atlas contours overlaid (atlas mask required)
# ---------------------------------------------------------------------------

if atlas_mask is not None:
    print("Generating volume-with-contours-light/dark.png …")
    overlay_colors = id_to_rgb if id_to_rgb else "white"
    for black_bg, suffix in [(False, "light"), (True, "dark")]:
        plotter_overlay = mean_vol.fusi.scale.db().fusi.plot.volume(
            slice_mode="z",
            cmap="gray",
            vmin=-15,
            vmax=0,
            cbar_label="Power Doppler (dB)",
            black_bg=black_bg,
        )
        plotter_overlay.add_contours(atlas_mask, colors=overlay_colors)
        plotter_overlay.savefig(
            str(HERE / f"volume-with-contours-{suffix}.png"), **_SAVEFIG_KWARGS
        )
        plotter_overlay.close()
    print("  Saved volume-with-contours-light.png and volume-with-contours-dark.png")
else:
    print("Skipping volume-with-contours-*.png (ATLAS_MASK_PATH not set).")

# ---------------------------------------------------------------------------
# 8. draw_napari_labels — interactive ROI drawing
# ---------------------------------------------------------------------------

print("\n── draw_napari_labels ────────────────────────────────────────────────")
print("Generating napari-draw-labels.png …")

try:
    viewer_draw, labels_layer = cf.plotting.draw_napari_labels(
        mean_vol.fusi.scale.db(),
        contrast_limits=(-15, 0),
        colormap="gray",
    )

    ny, nx = labels_layer.data.shape[-2], labels_layer.data.shape[-1]
    y_idx, x_idx = np.ogrid[:ny, :nx]

    def _paint_blob(
        data: np.ndarray,
        cy: int,
        cx: int,
        radius: int,
        label: int,
        rng: np.random.Generator,
    ) -> None:
        """Paint an irregular blob by scattering overlapping discs."""
        n_strokes = 18
        for _ in range(n_strokes):
            dy = int(rng.integers(-radius // 2, radius // 2 + 1))
            dx = int(rng.integers(-radius // 2, radius // 2 + 1))
            r = int(rng.integers(radius // 3, radius // 2 + 1))
            cy_ = int(np.clip(cy + dy, 0, data.shape[-2] - 1))
            cx_ = int(np.clip(cx + dx, 0, data.shape[-1] - 1))
            mask = (y_idx - cy_) ** 2 + (x_idx - cx_) ** 2 <= r**2
            data[..., mask] = label

    rng = np.random.default_rng(0)
    _paint_blob(labels_layer.data, ny // 3, nx // 4, ny // 16, 1, rng)
    _paint_blob(labels_layer.data, ny // 3, 3 * nx // 4, ny // 16, 2, rng)
    labels_layer.refresh()

    _napari_screenshot(viewer_draw, str(HERE / "napari-draw-labels.png"))
    viewer_draw.close()
    print("  Saved napari-draw-labels.png")
except Exception as exc:
    print(f"  draw_napari_labels screenshot failed: {exc}")

# ---------------------------------------------------------------------------
# 9. Carpet plot (matplotlib)
# ---------------------------------------------------------------------------

print("Generating carpet-plot-light/dark.png …")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    fig, _ax = pwd.fusi.plot.carpet(
        mask=brain_mask,
        detrend_order=1,
        standardize=True,
        cmap="gray",
        figsize=(12, 4),
        black_bg=black_bg,
    )
    fig.savefig(str(HERE / f"carpet-plot-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig)
print("  Saved carpet-plot-light.png and carpet-plot-dark.png")

# ---------------------------------------------------------------------------

print("\nDone! Rebuild the docs with `just docs` to see the images in place.")
