import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_seeds() -> None:
    """Seed numpy globally for each test."""
    np.random.seed(42)
