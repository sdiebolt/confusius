"""Python package for analysis and visualization of functional ultrasound imaging data."""

__all__ = ["io", "iq", "plotting", "qc", "registration", "xarray", "__version__"]

from importlib import metadata

__version__ = metadata.version("confusius")

from confusius import io, iq, plotting, qc, registration, xarray
