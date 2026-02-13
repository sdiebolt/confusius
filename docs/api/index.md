# API Reference

Complete reference documentation for the ConfUSIus API. Please refer to the [user
guide](../user_guide/index.md) for more detailed explanations and usage examples.

## Core Modules

### Data I/O

- **[I/O](io.md)**: Load and save fUSI data in various formats, including those from
  AUTC and EchoFrame systems, as well as general-purpose formats like Zarr and NIfTI.

### Beamformed In-phase/Quadrature Signals

- **[Beamformed in-phase/quadrature (IQ) signals](iq.md)**: Processing of beamformed IQ
  signals, including clutter filtering and computation of derived measures such as power
  Doppler and axial velocity.

### Signal Extraction

- **[Signal extraction](extract.md)**: Extract and reconstruct signals from fUSI data
  using spatial masks. Flatten spatial dimensions for analysis and unmask processed
  signals back to volumetric space.

### Visualization

- **[Plotting](plotting.md)**: Rich visualization utilities for fUSI data including
  static plots and interactive napari viewers.

### Registration

- **[Registration](registration.md)**: Motion correction and spatial alignment tools for
  aligning fUSI data across time points or with anatomical references.

### Xarray Integration

- **[Xarray](xarray.md)**: Xarray integration providing DataArray accessors for fUSI
  data.
