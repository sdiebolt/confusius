---
hide:
    - navigation
    - toc
---

# Welcome to ConfUSIus

ConfUSIus is a Python package for handling, visualization, preprocessing, and
statistical analysis of functional ultrasound imaging (fUSI) data.

!!! warning "Pre-Alpha Status"
    ConfUSIus is currently in **pre-alpha** and under **active development**. 
    The API is subject to change, and features may be incomplete or unstable.
    Use at your own risk.

## Quick Links

<div class="grid cards" markdown>

-   [:material-book-open-variant:{ .lg .middle } **User Guide**](user-guide/index.md)

    ---

    Learn how to use ConfUSIus for fUSI data analysis

-   [:material-code-braces:{ .lg .middle } **API Reference**](api/index.md)

    ---

    Comprehensive reference of all modules, classes, and functions

-   [:material-lightbulb:{ .lg .middle } **Examples**](examples/index.md)

    ---

    Example workflows and use cases

</div>

## Features

- **I/O Operations**: Load and save fUSI data in various formats (AUTC, EchoFrame,
  NIfTI, Zarr, etc.)
- **Beamformed IQ Processing**: Process raw beamformed IQ signals into power Doppler,
  velocity, and other derived metrics.
- **Registration**: Motion correction and spatial alignment tools.
- **Visualization**: Rich plotting utilities for fUSI data exploration.
- **Xarray Integration**: Seamless integration with Xarray for labeled multi-dimensional
  arrays.

## Installation

```bash
pip install confusius
```

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv add confusius
```

See the [Installation Guide](installation.md) for detailed instructions.

## Quick Start

```python
import confusius as cf

# Load fUSI data.
data = cf.io.load_nifti("path/to/data.nii.gz")

# Perform motion correction.
corrected_data = data.fusi.register.volumewise()

# Visualize with Napari.
correted_data.fusi.plot()
```
