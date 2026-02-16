"""Global fixtures shared across all tests."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
