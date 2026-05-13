# %% [markdown]
# # Local patch SVD denoising
#
# This example implements a local patch denoising workflow inspired by Veraart et al.
# and the MRtrix `dwidenoise` command. Unlike the global denoising notebook, the signal
# matrix is built independently around each voxel from a small spatial neighborhood.
#
# We use the original Exp1-style estimator from Veraart et al. 2016, applied to each
# local patch, and reconstruct only the center voxel of each patch.

# %% [markdown]
# ## Load one recording

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022
from confusius.signal import standardize

# Keep figure backgrounds transparent in docs and standalone notebooks.
bg_color = "none"
# Match text and axes styling to the active Matplotlib theme.
fg_color = mpl.colors.to_hex(plt.rcParams["text.color"])
# Keep notebook output compact for large DataArray displays.
xr.set_options(display_expand_data=False)

bids_root = fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201011",
    tasks="spontaneous",
    acqs="slice03",
)

pwd_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201011"
    / "fusi"
    / "sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd.nii.gz"
)
data = cf.load(pwd_path).compute().astype(np.float64)
data

# %% [markdown]
# ## Patchwise MRtrix-style estimator
#
# The estimator loops over spatial positions, forms a `(time, patch_voxels)` matrix,
# estimates how many of the smallest singular modes are noise, and reconstructs the
# center voxel from the remaining components.

# %%
def _select_noise_components_mrtrix(eigenvalues: np.ndarray, m: int, n: int) -> tuple[int, float]:
    """Return the number of noise components and the estimated noise variance."""

    r = min(m, n)
    q = max(m, n)
    lam_r = max(eigenvalues[0], 0.0) / q

    cumulative_lambda = 0.0
    sigma2 = 0.0
    cutoff_p = 0

    for p in range(r):
        lam = max(eigenvalues[p], 0.0) / q
        cumulative_lambda += lam
        gamma = (p + 1) / q
        sigma2_mean = cumulative_lambda / (p + 1)
        sigma2_width = (lam - lam_r) / (4.0 * np.sqrt(gamma))

        if sigma2_width < sigma2_mean:
            sigma2 = sigma2_mean
            cutoff_p = p + 1

    return cutoff_p, sigma2


def local_svd_denoise(
    data: xr.DataArray,
    *,
    patch_shape: tuple[int, int, int] = (1, 5, 5),
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Denoise one `(time, z, y, x)` recording with local patch SVD."""

    if any(size % 2 == 0 for size in patch_shape):
        raise ValueError("patch_shape must contain odd sizes.")

    time_size = data.sizes["time"]
    spatial_shape = tuple(data.sizes[dim] for dim in ("z", "y", "x"))
    half_widths = tuple(size // 2 for size in patch_shape)
    patch_voxels = int(np.prod(patch_shape))

    values = np.asarray(data.transpose("time", "z", "y", "x").values)
    padded = np.pad(
        values,
        ((0, 0),) + tuple((half, half) for half in half_widths),
        mode="reflect",
    )

    denoised = np.empty_like(values)
    sigma_map = np.empty(spatial_shape, dtype=np.float64)
    kept_rank_map = np.empty(spatial_shape, dtype=np.intp)

    for z, y, x in np.ndindex(spatial_shape):
        patch = padded[
            :,
            z : z + patch_shape[0],
            y : y + patch_shape[1],
            x : x + patch_shape[2],
        ].reshape(time_size, patch_voxels)

        if time_size <= patch_voxels:
            gram = patch @ patch.T
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            center_index = patch_voxels // 2
            cutoff_p, sigma2 = _select_noise_components_mrtrix(eigenvalues, time_size, patch_voxels)
            keep = np.ones(time_size, dtype=np.float64)
            keep[:cutoff_p] = 0.0
            projection = eigenvectors @ np.diag(keep) @ eigenvectors.T
            denoised[:, z, y, x] = (projection @ patch)[:, center_index]
            kept_rank_map[z, y, x] = time_size - cutoff_p
        else:
            gram = patch.T @ patch
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            cutoff_p, sigma2 = _select_noise_components_mrtrix(eigenvalues, time_size, patch_voxels)
            keep = np.ones(patch_voxels, dtype=np.float64)
            keep[:cutoff_p] = 0.0
            projection = eigenvectors @ np.diag(keep) @ eigenvectors.T
            center_index = patch_voxels // 2
            denoised[:, z, y, x] = (patch @ projection)[:, center_index]
            kept_rank_map[z, y, x] = patch_voxels - cutoff_p

        sigma_map[z, y, x] = np.sqrt(max(sigma2, 0.0))

    denoised_da = xr.DataArray(
        denoised,
        dims=("time", "z", "y", "x"),
        coords=data.coords,
    )
    sigma_da = xr.DataArray(
        sigma_map,
        dims=("z", "y", "x"),
        coords={dim: data.coords[dim] for dim in ("z", "y", "x")},
    )
    rank_da = xr.DataArray(
        kept_rank_map,
        dims=("z", "y", "x"),
        coords={dim: data.coords[dim] for dim in ("z", "y", "x")},
    )
    return denoised_da, sigma_da, rank_da


patch_shape = (1, 5, 5)
denoised, sigma_map, kept_rank_map = local_svd_denoise(data, patch_shape=patch_shape)

summary = {
    "patch_shape": patch_shape,
    "mean_sigma": float(sigma_map.mean()),
    "mean_kept_rank": float(kept_rank_map.mean()),
    "min_kept_rank": int(kept_rank_map.min()),
    "max_kept_rank": int(kept_rank_map.max()),
}
summary

# %% [markdown]
# ## Compare one standardized frame
#
# The denoising itself is performed on the original signal scale. For visualization, we
# standardize the reconstructed volumes afterward so small frame-level differences are
# easier to see.

# %%
data_view = standardize(data)
denoised_view = standardize(denoised)
residual_view = denoised_view - data_view

frame_idx = 64

fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True, facecolor="none")

for ax, (title, vol, cmap) in zip(
    axes[0],
    [
        ("Original", data_view, "viridis"),
        (
            f"Local patch SVD\n(mean kept rank={kept_rank_map.mean().item():.1f})",
            denoised_view,
            "viridis",
        ),
        ("Difference\n(denoised - original)", residual_view, "coolwarm"),
    ],
):
    cf.plotting.plot_volume(
        vol.isel(time=[frame_idx]),
        slice_mode="time",
        cmap=cmap,
        bg_color=bg_color,
        fg_color=fg_color,
        show_titles=False,
        show_colorbar=False,
        axes=ax,
    )
    ax.set_title(title, fontsize=9)

for ax, (title, vol, cmap) in zip(
    axes[1],
    [
        ("Estimated sigma", sigma_map.expand_dims(time=[0]), "magma"),
        ("Kept rank", kept_rank_map.expand_dims(time=[0]), "viridis"),
        ("Reference\n(zero difference)", (data_view * 0).isel(time=[frame_idx]), "coolwarm"),
    ],
):
    cf.plotting.plot_volume(
        vol,
        slice_mode="time",
        cmap=cmap,
        bg_color=bg_color,
        fg_color=fg_color,
        show_titles=False,
        show_colorbar=False,
        axes=ax,
    )
    ax.set_title(title, fontsize=9)
