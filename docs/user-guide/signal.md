---
icon: lucide/brush-cleaning
---

# Signal Processing

!!! info "Coming soon"
    This page is currently under construction. The signal processing modules provide:

    **Extraction:**
 
    - [`extract_with_mask`][confusius.extract.extract_with_mask]: Flatten spatial dimensions into a voxel array using a boolean mask.
    - [`extract_with_labels`][confusius.extract.extract_with_labels]: Extract region-averaged signals using an integer label map.
    - [`unmask`][confusius.extract.unmask]: Reconstruct a full spatial volume from a flat voxel array.

    **Preprocessing:**
 
    - [`censor_samples`][confusius.signal.censor_samples] / [`interpolate_samples`][confusius.signal.interpolate_samples]: Mark or interpolate corrupted samples (e.g., motion outliers).
    - [`regress_confounds`][confusius.signal.regress_confounds]: Remove nuisance signals via least-squares regression.
    - [`compute_compcor_confounds`][confusius.signal.compute_compcor_confounds]: Compute CompCor confounds from high-variance voxels.
    - [`filter_butterworth`][confusius.signal.filter_butterworth]: Apply low-pass, high-pass, or band-pass Butterworth filters.
    - [`standardize`][confusius.signal.standardize]: Scale signals to unit variance or percent signal change.
    - [`detrend`][confusius.signal.detrend]: Remove linear or polynomial trends from time series.
    - [`clean`][confusius.signal.clean]: Apply full preprocessing pipeline in one step.

    Please refer to the [API Reference](../api/signal.md) and [Roadmap](../roadmap.md) for more information.
