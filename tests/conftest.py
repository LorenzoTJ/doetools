"""Shared test fixtures and configuration for doetools tests.

This module provides common fixtures used across unit and integration tests.
"""

import numpy as np
import pytest
from doetools.utils.factors import ContinuousFactor, CategoricalFactor, MixtureFactor


# ==============================================================================
# Factor Fixtures
# ==============================================================================
# Fixtures for various factor configurations used in tests.
# These include continuous, categorical, mixture, and mixed factor sets.
# Both constrained and unconstrained scenarios are covered.
# Each fixture returns a dictionary of factor names to factor instances.
# =============================================================================
    
@pytest.fixture
def sample_continuous_factors():
    """Create a dictionary of sample continuous factors for testing."""
    return {
        'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80),
        'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5),
        'pH': ContinuousFactor(n_levels=3, lower_bound=3, upper_bound=9)
    }


@pytest.fixture
def sample_categorical_factors():
    """Create a dictionary of sample categorical factors for testing."""
    return {
        'Catalyst': CategoricalFactor(levels=['A', 'B', 'C']),
        'Solvent': CategoricalFactor(levels=['Water', 'Ethanol']),
        'Additive': CategoricalFactor(levels=['None', 'Salt', 'Surfactant', 'Polymer'])
    }


@pytest.fixture
def sample_mixture_factors():
    """Create a dictionary of sample mixture factors for testing."""
    return {
        'Component_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Component_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Component_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        "Component_D": MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }


@pytest.fixture
def constrained_mixture_factors():
    """Create mixture factors with upper bound constraints."""
    return {
        'Component_A': MixtureFactor(lower_bound=0.1, upper_bound=0.6),
        'Component_B': MixtureFactor(lower_bound=0.1, upper_bound=0.8),
        'Component_C': MixtureFactor(lower_bound=0.1, upper_bound=0.8)
    }


@pytest.fixture
def mixed_factors():
    """Create a dictionary with both process and mixture factors."""
    return {
        'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80),
        'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5),
        "Catalyst" : CategoricalFactor(levels=['A', 'B']),
        'Component_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Component_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }

@pytest.fixture
def process_factors():
    """Create a dictionary of only process factors for testing."""
    return {
        'Temperature': ContinuousFactor(n_levels=4, lower_bound=10, upper_bound=100),
        'Pressure': ContinuousFactor(n_levels=4, lower_bound=1, upper_bound=10),
        'pH': ContinuousFactor(n_levels=4, lower_bound=2, upper_bound=12),
        "Catalyst": CategoricalFactor(levels=['X', 'Y', 'Z'])
    }   

@pytest.fixture
def screening_process_factors():
    """Create a dictionary of only process factors for testing."""
    return {
        'Temperature': ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=100),
        'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=10),
        'pH': ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=12),
        "Catalyst": CategoricalFactor(levels=['X', 'Y', 'Z']),
        "Solvent": CategoricalFactor(levels=['A', 'B', "C"])
    }   
    
@pytest.fixture
def sample_pb_factors():
    """Create a dictionary of factors suitable for Plackett-Burman design."""
    return {
        'Factor_1': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10),
        'Factor_2': ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15),
        'Factor_3': ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=20),
        'Factor_4': ContinuousFactor(n_levels=2, lower_bound=15, upper_bound=25),
        'Factor_5': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=30)
    }


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_design_matrix():
    """Create a small design matrix for testing."""
    return np.array([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1]
    ])
