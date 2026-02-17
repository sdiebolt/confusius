# Roadmap

Alpha release checklist for ConfUSIus.

- [ ] [`io` module](api/io.md): input/output support.
    - [x] AUTC beamformed IQ.
    - [x] EchoFrame beamformed IQ.
    - [x] NIfTI.
    - [x] Zarr.
    - [ ] Iconeus SCAN (HDF5; see <https://github.com/XunMa1214/fUS-toolbox>).
- [x] [`iq` module](api/iq.md): beamformed IQ processing.
    - [x] [`process_iq_blocks`][confusius.iq.process_iq_blocks] (sliding-window processing).
    - [x] [`process_iq_to_power_doppler`][confusius.iq.process_iq_to_power_doppler].
    - [x] [`process_iq_to_axial_velocity`][confusius.iq.process_iq_to_axial_velocity].
- [x] [`extract` module](api/extract.md): signal extraction.
    - [x] [`with_mask`][confusius.extract.with_mask].
    - [ ] `with_labels`.
    - [x] [`unmask`][confusius.extract.unmask].
- [x] [`signal` module](api/signal.md): preprocessing (see `nilearn.signal.clean`).
    - [x] [`censor_samples`][confusius.signal.censor_samples] /
      [`interpolate_samples`][confusius.signal.interpolate_samples].
    - [x] [`regress_confounds`][confusius.signal.regress_confounds].
    - [x] [`filter_butterworth`][confusius.signal.filter_butterworth].
    - [x] [`standardize`][confusius.signal.standardize].
    - [x] [`detrend`][confusius.signal.detrend].
- [ ] [`plotting` module](api/plotting.md): visualization.
    - [x] [`plot_napari`][confusius.plotting.plot_napari] (interactive).
    - [x] [`plot_carpet`][confusius.plotting.plot_carpet].
    - [ ] `plot_volume` (2D slices via `matplotlib.pyplot.pcolormesh`).
    - [ ] `plot_roi` (2D contour plots).
- [ ] [`registration` module](api/registration.md): volume alignment.
    - [x] [`register_volumewise`][confusius.registration.register_volumewise].
    - [x] [`compute_framewise_displacement`][confusius.registration.compute_framewise_displacement].
    - [ ] `register_volumes`.
- [ ] `atlas` module for atlas fetching and resampling.
    - [ ] [BrainGlobe Atlas
      API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html) wrapper
      using Xarray Datasets?
- [ ] `connectivity` module: functional connectivity.
- [ ] `glm` module: general linear models.
