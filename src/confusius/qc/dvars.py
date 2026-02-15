"""DVARS computation for quality control.

Portions of this file are derived from Nipype, which is licensed under the Apache
License 2.0. See ``NOTICE`` file for details.
"""

import numpy as np
import numpy.typing as npt


def _ar1_yule_walker(signal: npt.NDArray) -> float:
    """Estimate AR(1) coefficient using Yule-Walker equations.

    The Yule-Walker equations estimate the autoregressive coefficients of a process by
    solving a system of linear equations involving the autocorrelation sequence.

    For an AR(1) process ``x(n) = a(1) * x(n-1) + e(n)``, the Yule-Walker equation
    is: ``R(1) = a(1) * R(0)``, where ``R(k)`` is the autocorrelation at lag ``k``.

    Parameters
    ----------
    signal : (t,) numpy.ndarray
        Array of signal values.

    Returns
    -------
    float
        The AR(1) coefficient ``a(1)``.

    Notes
    -----
    For AR(1), we only need autocorrelation at lags 0 and 1, so we use direct
    computation (``O(n)``) rather than FFT-based methods (``O(n log n)``).
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()

    n = len(signal)

    # Direct computation of autocorrelation at lags 0 and 1.
    # For AR(1), we only need these two values, so full FFT is unnecessary.
    r0 = np.dot(signal, signal) / n  # lag 0
    r1 = np.dot(signal[:-1], signal[1:]) / n  # lag 1

    # AR(1) coefficient: R(1) / R(0).
    # Add small regularization to avoid division by zero.
    if r0 < 1e-15:
        return 0.0

    return np.clip(r1 / r0, -1.0, 1.0)


def compute_dvars(
    signals: npt.NDArray,
    standardize: bool = True,
    normalization_factor: float | None = 1000,
    remove_zero_variance: bool = True,
    variance_tolerance: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute the DVARS metric.

    DVARS (D referring to temporal derivative, VARS to RMS variance) measures how much
    the intensity at one time point changes relative to the previous time point. The
    standardized version accounts for the expected variance given the temporal
    autocorrelation structure of the data.

    This function was adapted from `nipype.algorithms.confounds.compute_dvars`.

    Parameters
    ----------
    signals : (n_samples, n_signals) numpy.ndarray
        Signals to compute DVARS from. Each column is a separate signal (e.g., voxel
        time series).
    standardize : bool, default: True
        Whether to computed the standardized DVARS. When ``True``, the DVARS values are
        standardized by the expected standard deviation given the AR(1) structure of the
        data. When ``False``, return non-standardized DVARS (raw RMS of temporal
        differences).
    normalization_factor : float or None, default: 1000
        Normalization factor applied to the median signal intensity. If ``None``, no
        normalization is performed. Note that standardized DVARS is invariant to
        intensity normalization.
    remove_zero_variance : bool, default: True
        Whether to exclude signals with near-zero variance. Zero-variance signals can
        artificially reduce the median during intensity normalization and should
        typically be removed.
    variance_tolerance : float, default: 0.0
        Tolerance for identifying zero-variance signals. Signals with robust standard
        deviation (``IQR/1.349``) less than or equal to this value are considered to
        have zero variance.

    Returns
    -------
    (n_samples,) numpy.ndarray
        DVARS values. When `standardized` is ``True``, returns standardized DVARS. The
        first element is set to the minimum DVARS value (following FSL convention).

    Notes
    -----
    The standardized DVARS metric was introduced by Nichols (2013) to provide a
    normalized measure of DVARS that accounts for the temporal autocorrelation in fMRI
    data.

    The computation involves:

    1. Robust standard deviation estimation using IQR (following FSL's "lower"
       interpolation method).
    2. Optional removal of zero-variance signals.
    3. Optional intensity normalization by median scaling.
    4. AR(1) coefficient estimation via Yule-Walker equations (standardized only).
    5. Computation of expected variance for temporal differences given AR(1)
       structure (standardized only).
    6. Standardization by dividing observed DVARS by expected DVARS (standardized
       only).

    References
    ----------
    [^1]:
        Smyser, Christopher D., et al. “Functional Connectivity MRI in Infants:
        Exploration of the Functional Organization of the Developing Brain.” NeuroImage,
        vol. 56, no. 3, June 2011, pp. 1437–52. DOI.org (Crossref),
        https://doi.org/10.1016/j.neuroimage.2011.02.073.

    [^2]:
        Power, Jonathan D., et al. “Spurious but Systematic Correlations in Functional
        Connectivity MRI Networks Arise from Subject Motion.” NeuroImage, vol. 59, no.
        3, Feb. 2012, pp. 2142–54. DOI.org (Crossref),
        https://doi.org/10.1016/j.neuroimage.2011.10.018.

    [^3]:
        Nichols, Thomas E. “Notes on Creating a Standardized Version of DVARS.” arXiv,
        2017. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.1704.01469.

    [^4]:
        Afyouni, Soroosh, and Thomas E. Nichols. “Insight and Inference for DVARS.”
        NeuroImage, vol. 172, May 2018, pp. 291–312. DOI.org (Crossref),
        https://doi.org/10.1016/j.neuroimage.2017.12.098.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.qc import compute_dvars
    >>> # Generate synthetic signals: 100 time points, 50 voxels
    >>> rng = np.random.default_rng(42)
    >>> signals = rng.standard_normal((100, 50))
    >>> # Add AR(1) structure
    >>> for t in range(1, 100):
    ...     signals[t] = 0.5 * signals[t - 1] + 0.5 * signals[t]
    >>> # Compute standardized DVARS (default)
    >>> dvars_stdz = compute_dvars(signals)
    >>> dvars_stdz.shape
    (100,)
    >>> # Compute non-standardized DVARS
    >>> dvars_nstd = compute_dvars(signals, standardized=False)
    >>> dvars_nstd.shape
    (100,)
    """
    from scipy import stats as sp_stats

    if signals.ndim != 2:
        raise ValueError(f"signals must be 2D, got {signals.ndim}D")

    # Robust standard deviation using IQR with "lower" interpolation (FSL convention).
    # IQR / 1.349 approximates the standard deviation for normal distributions.
    signals_sd = sp_stats.iqr(signals, axis=0, scale=1.0, interpolation="lower")
    signals_sd = signals_sd / 1.349

    if remove_zero_variance:
        nonzero_mask = signals_sd > variance_tolerance
        if not np.any(nonzero_mask):
            raise ValueError(
                "All signals have variance below tolerance. Check input data or adjust "
                "'variance_tol'."
            )
        signals = signals[:, nonzero_mask]
        signals_sd = signals_sd[nonzero_mask]

    # Intensity normalization does not affect standardized DVARS.
    if not standardize and normalization_factor is not None:
        signals *= normalization_factor / np.median(signals)
        # The standard deviation will be impacted by the same scaling factor.
        signals_sd *= normalization_factor / np.median(signals)

    signals_diff = np.diff(signals, axis=0)

    # Non-standardized DVARS: RMS of differences across signals at each time point.
    dvars_nstd = np.sqrt(np.mean(signals_diff**2, axis=1))

    # FSL convention: first element is the minimum DVARS.
    dvars_nstd = np.insert(dvars_nstd, 0, dvars_nstd.min())

    if not standardize:
        return dvars_nstd

    ar1 = np.array([_ar1_yule_walker(signals[:, i]) for i in range(signals.shape[1])])

    # For an AR(1) process, the variance of temporal differences is:
    # Var(diff) = Var(x) * (2 * (1 - rho)) = 2 * (1 - rho) * sigma^2
    # where rho is the AR(1) coefficient and sigma^2 is the process variance. See
    # Nicholas (2013).
    diff_sdhat = np.sqrt(2 * (1 - ar1)) * signals_sd

    # Standardized DVARS: observed DVARS divided by mean expected DVARS.
    dvars_stdz = dvars_nstd / np.mean(diff_sdhat)

    return dvars_stdz
