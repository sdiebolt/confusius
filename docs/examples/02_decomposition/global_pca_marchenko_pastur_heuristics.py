# %% [markdown]
# # Global SVD denoising heuristics
#
# This example compares three global low-rank denoising strategies on one fUSI
# recording:
#
# - a simple 99 % energy cutoff,
# - a global adaptation of the MRtrix `dwidenoise` / Veraart MP estimator, and
# - a log-scree linear-tail heuristic inspired by Song et al. 2017,
#   [Ultrasound Small Vessel Imaging With Block-Wise Adaptive Local Clutter
#   Filtering](https://doi.org/10.1109/TMI.2016.2605819).
#
# Unlike the plain PCA example, this notebook works on the original signal scale and uses
# a direct singular value decomposition of the `(time, voxels)` matrix.

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
# ## Build one global `(time, voxels)` matrix
#
# We keep the native voxel scaling for denoising. The matrix decomposition below uses the
# same raw signal scale for every method.

# %%
spatial_dims = tuple(dim for dim in data.dims if dim != "time")
stacked = data.stack(feature=spatial_dims).transpose("time", "feature")
X = np.asarray(stacked.values)

U, singular_values, Vh = np.linalg.svd(X, full_matrices=False)
eigenvalues = singular_values**2
component_ids = np.arange(1, singular_values.size + 1)
energy_ratio = eigenvalues / eigenvalues.sum()
cum_energy = np.cumsum(energy_ratio)

# Trim numerical zero modes from the tail fits.
nonzero_mask = singular_values > np.finfo(singular_values.dtype).eps * singular_values[0]
singular_values_fit = singular_values[nonzero_mask]
eigenvalues_fit = eigenvalues[nonzero_mask]
component_ids_fit = component_ids[nonzero_mask]

n_samples, n_features = X.shape
r = min(n_samples, n_features)
q = max(n_samples, n_features)


def reconstruct_with_rank(
    left_vectors: np.ndarray,
    singular_values: np.ndarray,
    right_vectors: np.ndarray,
    rank: int,
    template: xr.DataArray,
) -> xr.DataArray:
    """Reconstruct the data matrix using the leading `rank` singular values."""

    filtered = np.zeros_like(singular_values)
    filtered[:rank] = singular_values[:rank]
    matrix = (left_vectors * filtered) @ right_vectors
    return xr.DataArray(
        matrix,
        dims=stacked.dims,
        coords=stacked.coords,
    ).unstack("feature").transpose(*template.dims)


def select_rank_energy(cumulative_energy: np.ndarray, threshold: float = 0.99) -> dict[str, float]:
    """Return the smallest rank whose cumulative matrix energy exceeds `threshold`."""

    rank = int(np.searchsorted(cumulative_energy, threshold)) + 1
    return {
        "rank": float(rank),
        "energy": float(cumulative_energy[rank - 1]),
    }


def select_rank_veraart_mrtrix(
    eigenvalues: np.ndarray,
    r: int,
    q: int,
    *,
    exp1: bool = True,
) -> dict[str, float]:
    """Return the global MRtrix-style Veraart cutoff.

    This follows the logic in MRtrix `dwidenoise.cpp`, but applies it once to the full
    `(time, voxels)` matrix instead of local patches.
    """

    lam_r = max(eigenvalues[-1], 0.0) / q
    cumulative_lambda = 0.0
    sigma2 = 0.0
    cutoff_p = 0

    for p in range(r):
        lam = max(eigenvalues[-(p + 1)], 0.0) / q
        cumulative_lambda += lam
        gamma = (p + 1) / (q if exp1 else q - (r - p - 1))
        sigma2_mean = cumulative_lambda / (p + 1)
        sigma2_width = (lam - lam_r) / (4.0 * np.sqrt(gamma))

        if sigma2_width < sigma2_mean:
            sigma2 = sigma2_mean
            cutoff_p = p + 1

    rank = max(r - cutoff_p, 1)
    return {
        "rank": float(rank),
        "noise_components": float(cutoff_p),
        "sigma2": float(sigma2),
        "energy": float(cum_energy[rank - 1]),
    }


def select_rank_log_linear_tail(
    singular_values: np.ndarray,
    min_tail: int = 20,
    min_r2: float = 0.995,
    deviation_std: float = 3.0,
) -> dict[str, np.ndarray | float]:
    """Return a log-scree linear-tail cutoff inspired by Song et al. 2017.

    Since we do not have the paper's Doppler-frequency pre-cutoff, we search for the
    deepest tail that still looks approximately linear in log scale.
    """

    m = len(singular_values)
    if m < min_tail + 2:
        raise ValueError("Not enough singular values for a tail fit.")

    x = np.arange(1, m + 1, dtype=float)
    y = np.log10(np.maximum(singular_values, np.finfo(float).tiny))

    best_start: int | None = None
    coeffs: np.ndarray | None = None
    best_rmse: float | None = None
    found_linear_tail = False

    for start in range(m - min_tail, -1, -1):
        trial_coeffs = np.polyfit(x[start:], y[start:], deg=1)
        trial_fit = np.polyval(trial_coeffs, x[start:])
        trial_rmse = np.sqrt(np.mean((y[start:] - trial_fit) ** 2))
        ss_tot = np.sum((y[start:] - y[start:].mean()) ** 2)
        trial_r2 = 1.0 - np.sum((y[start:] - trial_fit) ** 2) / ss_tot

        if trial_r2 >= min_r2:
            found_linear_tail = True
            best_start = start
            coeffs = trial_coeffs
            best_rmse = trial_rmse
        elif found_linear_tail:
            break

    if best_start is None or coeffs is None or best_rmse is None:
        best_start = m - min_tail
        coeffs = np.polyfit(x[best_start:], y[best_start:], deg=1)
        best_rmse = np.sqrt(
            np.mean((y[best_start:] - np.polyval(coeffs, x[best_start:])) ** 2)
        )
        found_linear_tail = False

    y_fit = np.polyval(coeffs, x)
    tail_residuals = y[best_start:] - y_fit[best_start:]
    tolerance = max(deviation_std * tail_residuals.std(ddof=0), 0.02)

    if found_linear_tail:
        rank = 1
        for idx in range(best_start, -1, -1):
            if y[idx] > y_fit[idx] + tolerance:
                rank = idx + 1
                break
    else:
        rank = best_start + 1

    return {
        "rank": float(rank),
        "found_linear_tail": float(found_linear_tail),
        "pre_cutoff": float(best_start + 1),
        "tolerance": float(tolerance),
        "rmse": float(best_rmse),
        "energy": float(cum_energy[rank - 1]),
        "x": x,
        "log_singular_values": y,
        "log_fit": y_fit,
    }


energy_result = select_rank_energy(cum_energy)
veraart_result = select_rank_veraart_mrtrix(eigenvalues_fit, r=len(eigenvalues_fit), q=q)
linear_result = select_rank_log_linear_tail(singular_values_fit)

k_90 = int(energy_result["rank"])
k_veraart = int(veraart_result["rank"])
k_linear = int(linear_result["rank"])

selection_summary = {
    "k_90": k_90,
    "k_veraart": k_veraart,
    "k_linear": k_linear,
    "var_90": float(energy_result["energy"]),
    "var_veraart": float(veraart_result["energy"]),
    "var_linear": float(linear_result["energy"]),
    "veraart_sigma2": float(veraart_result["sigma2"]),
    "veraart_noise_components": int(veraart_result["noise_components"]),
    "linear_pre_cutoff": int(linear_result["pre_cutoff"]),
}
selection_summary

# %% [markdown]
# ## Compare the rank-selection rules

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

axes[0].plot(component_ids, np.log10(singular_values), marker="o", ms=2, lw=1)
axes[0].axvline(k_90, color="tab:red", lw=1.5, ls="--", label=f"99 % (k={k_90})")
axes[0].axvline(
    k_veraart,
    color="tab:orange",
    lw=1.5,
    ls="--",
    label=f"MRtrix-style Veraart (k={k_veraart})",
)
axes[0].axvline(k_linear, color="tab:purple", lw=1.5, ls=":", label=f"Linear tail (k={k_linear})")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("log10 singular value")
axes[0].set_title("Global singular value spectrum")
axes[0].legend()

axes[1].plot(component_ids, cum_energy * 100, marker="o", ms=2, lw=1)
axes[1].axhline(99, color="tab:red", lw=1, ls="--", label="99 % energy")
axes[1].axvline(k_90, color="tab:red", lw=1.5, ls="--")
axes[1].axvline(k_veraart, color="tab:orange", lw=1.5, ls="--", label="MRtrix-style Veraart")
axes[1].axvline(k_linear, color="tab:purple", lw=1.5, ls=":", label="Linear tail")
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative matrix energy (%)")
axes[1].set_title("Energy retained by each cutoff")
axes[1].legend()

# %% [markdown]
# ## Compare reconstructed frames
#
# We reconstruct three denoised versions from the same global SVD, then standardize the
# reconstructed volumes only for display so that frame differences are easier to see.

# %%
denoised_90 = reconstruct_with_rank(U, singular_values, Vh, k_90, data)
denoised_veraart = reconstruct_with_rank(U, singular_values, Vh, k_veraart, data)
denoised_linear = reconstruct_with_rank(U, singular_values, Vh, k_linear, data)

data_view = standardize(data)
denoised_90_view = standardize(denoised_90)
denoised_veraart_view = standardize(denoised_veraart)
denoised_linear_view = standardize(denoised_linear)
residual_90_view = denoised_90_view - data_view
residual_veraart_view = denoised_veraart_view - data_view
residual_linear_view = denoised_linear_view - data_view
top_vmin = float(
    min(
        data_view.min().item(),
        denoised_90_view.min().item(),
        denoised_veraart_view.min().item(),
        denoised_linear_view.min().item(),
    )
)
top_vmax = float(
    max(
        data_view.max().item(),
        denoised_90_view.max().item(),
        denoised_veraart_view.max().item(),
        denoised_linear_view.max().item(),
    )
)
residual_limit = float(
    max(
        np.abs(residual_90_view).max().item(),
        np.abs(residual_veraart_view).max().item(),
        np.abs(residual_linear_view).max().item(),
    )
)

# %%
frame_idx = 64

fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True, facecolor="none")

for ax, (title, vol, cmap) in zip(
    axes[0],
    [
        ("Original", data_view, "viridis"),
        (
            f"99 % energy\n(k={k_90}, {energy_result['energy'] * 100:.1f} %)",
            denoised_90_view,
            "viridis",
        ),
        (
            f"MRtrix-style Veraart\n(k={k_veraart}, {veraart_result['energy'] * 100:.1f} %)",
            denoised_veraart_view,
            "viridis",
        ),
        (
            (
                f"Linear tail fit\n(k={k_linear}, {linear_result['energy'] * 100:.1f} %)"
                if linear_result["found_linear_tail"]
                else f"Linear tail fallback\n(k={k_linear}, {linear_result['energy'] * 100:.1f} %)"
            ),
            denoised_linear_view,
            "viridis",
        ),
    ],
):
    cf.plotting.plot_volume(
        vol.isel(time=[frame_idx]),
        slice_mode="time",
        cmap=cmap,
        vmin=top_vmin,
        vmax=top_vmax,
        bg_color=bg_color,
        fg_color=fg_color,
        show_titles=False,
        show_colorbar=True,
        axes=ax,
    )
    ax.set_title(title, fontsize=9)

for ax, (title, vol, cmap) in zip(
    axes[1],
    [
        ("Reference\n(zero difference)", data_view * 0, "coolwarm"),
        ("99 % energy\nminus original", residual_90_view, "coolwarm"),
        ("MRtrix-style Veraart\nminus original", residual_veraart_view, "coolwarm"),
        (
            "Linear tail fit\nminus original"
            if linear_result["found_linear_tail"]
            else "Linear tail fallback\nminus original",
            residual_linear_view,
            "coolwarm",
        ),
    ],
):
    cf.plotting.plot_volume(
        vol.isel(time=[frame_idx]),
        slice_mode="time",
        cmap=cmap,
        vmin=-residual_limit,
        vmax=residual_limit,
        bg_color=bg_color,
        fg_color=fg_color,
        show_titles=False,
        show_colorbar=True,
        axes=ax,
    )
    ax.set_title(title, fontsize=9)

top_mappable = axes[0, 0].images[0] if axes[0, 0].images else axes[0, 0].collections[0]
bottom_mappable = (
    axes[1, 0].images[0] if axes[1, 0].images else axes[1, 0].collections[0]
)

fig.colorbar(top_mappable, ax=axes[0, :], location="right", fraction=0.03, pad=0.02)
fig.colorbar(
    bottom_mappable,
    ax=axes[1, :],
    location="right",
    fraction=0.03,
    pad=0.02,
)
