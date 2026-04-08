"""Generate documentation images for the Quality Control user guide.

Data is fetched automatically from the Nunez-Elizalde et al. (2022) fUSI-BIDS
dataset on OSF (https://osf.io/43skw/) via `confusius.datasets`.  The first run
downloads ~30 MB; subsequent runs use the local cache.

Usage
-----
Run from the project root::

    uv run docs/images/qc/generate.py

All images are saved to docs/images/qc/.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022
from confusius.plotting import plot_carpet
from confusius.qc import compute_cv, compute_dvars, compute_tsnr

HERE = Path(__file__).parent

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

print("Loading fUSI data …")
pwd = cf.load(_FUSI_PATH)
print(f"  {pwd.dims}, shape {dict(pwd.sizes)}")

# Brain mask: voxels above the 40th percentile of the mean image.
# This is a simple proxy for the brain region in the absence of a dedicated mask.
brain_mask = pwd.mean("time") > float(pwd.mean("time").quantile(0.4))

print("Extracting brain signals …")
signals = pwd.fusi.extract.with_mask(brain_mask).compute()

_SAVEFIG_KWARGS = {"dpi": 150, "bbox_inches": "tight", "transparent": True}

# ---------------------------------------------------------------------------
# 1. DVARS line plot
# ---------------------------------------------------------------------------

print("\n── DVARS ─────────────────────────────────────────────────────────────")
print("Computing DVARS …")
dvars = compute_dvars(signals)
threshold = 3

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    bg_color = "#1a1a1a" if black_bg else "white"
    fg_color = "white" if black_bg else "black"
    line_color = "#4c9be8" if black_bg else "#1f6fbf"
    threshold_color = "#e05c5c" if black_bg else "#c0392b"

    fig, ax = plt.subplots(figsize=(10, 3), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    flagged = dvars > threshold
    dvars.plot(ax=ax, color=line_color, linewidth=0.8, label=dvars.attrs["long_name"])
    dvars[flagged].plot(
        ax=ax,
        marker="o",
        linestyle="",
        color=threshold_color,
        ms=3,
        zorder=3,
        label="Flagged frames",
    )
    ax.axhline(
        threshold,
        color=threshold_color,
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold ({threshold})",
    )

    ax.xaxis.label.set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)
    ax.tick_params(colors=fg_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg_color)
    ax.legend(facecolor=bg_color, labelcolor=fg_color, framealpha=0.5, fontsize=9)

    fig.tight_layout()
    fig.savefig(str(HERE / f"qc-dvars-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig)

print("  Saved qc-dvars-light.png and qc-dvars-dark.png")

# ---------------------------------------------------------------------------
# 2. Carpet plot
# ---------------------------------------------------------------------------

print("\n── Carpet plot ───────────────────────────────────────────────────────")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    fig, ax = plot_carpet(pwd, mask=brain_mask, black_bg=black_bg)
    fig.savefig(str(HERE / f"qc-carpet-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig)

print("  Saved qc-carpet-light.png and qc-carpet-dark.png")

# ---------------------------------------------------------------------------
# 3. Mean power Doppler (reference for CV and tSNR sections)
# ---------------------------------------------------------------------------

print("\n── Mean power Doppler ────────────────────────────────────────────────")
print("Computing mean (dB scale) …")
mean_pwd = pwd.mean("time").fusi.scale.db()

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = mean_pwd.fusi.plot.volume(slice_mode="z", black_bg=black_bg)
    plotter.savefig(str(HERE / f"qc-mean-pwd-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

print("  Saved qc-mean-pwd-light.png and qc-mean-pwd-dark.png")

# ---------------------------------------------------------------------------
# 4. CV spatial map
# ---------------------------------------------------------------------------

print("\n── CV map ────────────────────────────────────────────────────────────")
print("Computing CV …")
cv = compute_cv(pwd)

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = cv.fusi.plot.volume(slice_mode="z", black_bg=black_bg)
    plotter.savefig(str(HERE / f"qc-cv-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

print("  Saved qc-cv-light.png and qc-cv-dark.png")

# ---------------------------------------------------------------------------
# 5. tSNR spatial map
# ---------------------------------------------------------------------------

print("\n── tSNR map ──────────────────────────────────────────────────────────")
print("Computing tSNR …")
tsnr = compute_tsnr(pwd)

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = tsnr.fusi.plot.volume(slice_mode="z", black_bg=black_bg)
    plotter.savefig(str(HERE / f"qc-tsnr-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

print("  Saved qc-tsnr-light.png and qc-tsnr-dark.png")

# ---------------------------------------------------------------------------

print("\nDone! Rebuild the docs with `just docs` to see the images in place.")
