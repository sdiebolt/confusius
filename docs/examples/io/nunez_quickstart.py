# %% [markdown]
# # Load and plot a Nunez-Elizalde recording
#
# Fetch a small subset of the Nunez-Elizalde 2022 dataset, load a power Doppler time
# series, and visualize the mean volume in dB.

# %%
from pathlib import Path

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022

# Download dataset (cached after the first run, ~30 MB).
bids_root = fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201011",
    tasks="spontaneous",
    acqs="slice03",
)

# Load power Doppler time series.
pwd_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201011"
    / "fusi"
    / "sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd.nii.gz"
)
data = cf.load(pwd_path)
data

# %% [markdown]
# Average over time and convert to dB scale for a quick static preview.

# %% tags=["thumbnail"]
import matplotlib.pyplot as plt

mean_db = data.mean("time").fusi.scale.db()

# Plot all z-slices.
plotter = mean_db.fusi.plot.volume(
    cmap="gray",
    cbar_label="Power Doppler (dB)",
)
plt.show()
