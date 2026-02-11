"""Tests for quality control module initialization."""

from confusius import qc


def test_qc_module_import():
    """Test that qc module can be imported."""
    assert hasattr(qc, "compute_dvars")


def test_qc_all_exports():
    """Test that qc.__all__ contains expected exports."""
    assert "compute_dvars" in qc.__all__
