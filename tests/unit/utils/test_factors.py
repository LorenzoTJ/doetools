"""
Unit tests for the factors module.

Tests cover three factor types used in Design of Experiments:
- CategoricalFactor: Discrete non-numeric levels
- ContinuousFactor: Numeric factors with defined ranges
- MixtureFactor: Mixture components with proportions
"""

import pytest
import numpy as np

from doetools import ContinuousFactor, MixtureFactor, CategoricalFactor


# ==============================================================================
#                          CategoricalFactor Tests
# ==============================================================================

@pytest.mark.parametrize("levels,expected_codes", [
    (["A", "B"], [-1.0, 1.0]),
    (["A", "B", "C"], [-1.0, 0.0, 1.0]),
    (["Low", "Medium", "High", "Very High"], [-1.0, -1/3, 1/3, 1.0]),
    (["L1", "L2", "L3", "L4", "L5"], [-1.0, -0.5, 0.0, 0.5, 1.0]),
])
def test_categorical_factor_coding(levels, expected_codes):
    """Test that categorical factors are correctly coded to [-1, 1] range."""
    factor = CategoricalFactor(levels=levels)
    
    assert factor.type == "cat"
    assert factor.levels == levels
    assert len(factor.coded_levels) == len(levels)
    assert np.allclose(factor.coded_levels, expected_codes)
    assert factor.reference_level == levels[0]
    assert factor.reference_code == expected_codes[0]


def test_categorical_factor_explicit_reference_preserves_coding():
    """An explicit reference changes defaults without reordering level codes."""
    levels = ["A", "B", "C"]
    factor = CategoricalFactor(levels=levels, reference_level="B")

    assert factor.levels == levels
    assert np.array_equal(factor.coded_levels, [-1.0, 0.0, 1.0])
    assert factor.reference_level == "B"
    assert factor.reference_code == 0.0


def test_categorical_factor_rejects_unknown_reference():
    """The reference must identify one of the declared levels."""
    with pytest.raises(ValueError, match="reference_level.*must be one of"):
        CategoricalFactor(levels=["A", "B"], reference_level="C")


def test_categorical_factor_two_levels():
    """Test the common case of a two-level categorical factor."""
    factor = CategoricalFactor(levels=["Low", "High"])
    
    assert factor.type == "cat"
    assert len(factor.levels) == 2
    assert np.array_equal(factor.coded_levels, [-1.0, 1.0])


def test_categorical_factor_properties():
    """Test that categorical factors have correct properties."""
    factor = CategoricalFactor(levels=["A", "B", "C", "D"])
    
    # Coded levels should span exactly [-1, 1]
    assert factor.coded_levels[0] == -1.0
    assert factor.coded_levels[-1] == 1.0
    
    # Coded levels should be monotonically increasing
    assert np.all(np.diff(factor.coded_levels) > 0)
    
    # Coded levels should be evenly spaced
    diffs = np.diff(factor.coded_levels)
    assert np.allclose(diffs, diffs[0], atol = 0.01)

# ==============================================================================
#                          ContinuousFactor Tests
# ==============================================================================

def test_continuous_factor():
    """Test basic continuous factor creation and level generation."""
    factor1 = ContinuousFactor(n_levels=5, lower_bound=0.0, upper_bound=10.0, decimals=1)
    factor2 = ContinuousFactor(n_levels=3, lower_bound=-5.0, upper_bound=5.0, decimals=0)
    
    # Factor 1: decimal levels
    assert factor1.type == "cont"
    assert len(factor1.levels) == 5
    assert len(factor1.coded_levels) == 5
    assert factor1.lower_bound == 0.0
    assert factor1.upper_bound == 10.0
    assert factor1.n_levels == 5
    assert np.allclose(factor1.levels, [0.0, 2.5, 5.0, 7.5, 10.0])
    assert np.allclose(factor1.coded_levels, [-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Factor 2: integer levels
    assert factor2.type == "cont"
    assert len(factor2.levels) == 3
    assert len(factor2.coded_levels) == 3
    assert np.array_equal(factor2.levels, [-5.0, 0.0, 5.0])
    assert np.allclose(factor2.coded_levels, [-1.0, 0.0, 1.0])


def test_continuous_factor_properties():
    """Test that continuous factors have correct mathematical properties."""
    factor = ContinuousFactor(n_levels=7, lower_bound=1.0, upper_bound=9.0, decimals=2)
    
    # Levels should span exactly [lower_bound, upper_bound]
    assert factor.levels[0] == factor.lower_bound
    assert factor.levels[-1] == factor.upper_bound
    
    # Coded levels should span exactly [-1, 1]
    assert factor.coded_levels[0] == -1.0
    assert factor.coded_levels[-1] == 1.0
    
    # Levels should be monotonically increasing
    assert np.all(np.diff(factor.levels) > 0)
    assert np.all(np.diff(factor.coded_levels) > 0)
    
    # Levels should be evenly spaced
    level_diffs = np.diff(factor.levels)
    assert np.allclose(level_diffs, level_diffs[0], atol=0.01)


def test_continuous_factor_two_levels():
    """Test the common case of a two-level continuous factor."""
    factor = ContinuousFactor(n_levels=2, lower_bound=20.0, upper_bound=80.0, decimals=0)
    
    assert np.array_equal(factor.levels, [20.0, 80.0])
    assert np.array_equal(factor.coded_levels, [-1.0, 1.0])


def test_continuous_factor_fixed_level():
    """A constant imported factor has a finite center code."""
    factor = ContinuousFactor(n_levels=1, lower_bound=5.0, upper_bound=5.0)

    assert np.array_equal(factor.levels, [5.0])
    assert np.array_equal(factor.coded_levels, [0.0])


def test_continuous_factor_small_range():
    """Test continuous factor with a very small range."""
    factor = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=0.1, decimals=2)
    
    assert factor.type == "cont"
    assert np.allclose(factor.levels, [0.0, 0.05, 0.1])
    assert np.allclose(factor.coded_levels, [-1.0, 0.0, 1.0])


def test_continuous_factor_rounding():
    """Test that rounding is applied correctly to levels."""
    factor = ContinuousFactor(n_levels=5, lower_bound=0.0, upper_bound=1.0, decimals=3)
    
    # All levels should be rounded to 3 decimals
    for level in factor.levels:
        assert level == round(level, 3)


def test_continuous_factor_errors():
    """Test error handling for invalid continuous factor configurations."""
    
    # Lower bound greater than upper bound
    with pytest.raises(ValueError, match="lower_bound needs to be smaller"):
        ContinuousFactor(n_levels=5, lower_bound=10.0, upper_bound=0.0)
    
    # Negative decimals
    with pytest.raises(ValueError, match="decimals must be >= 0"):
        ContinuousFactor(n_levels=5, lower_bound=0.0, upper_bound=10.0, decimals=-1)

    with pytest.raises(ValueError, match="n_levels must be >= 1"):
        ContinuousFactor(n_levels=0, lower_bound=0.0, upper_bound=10.0)

    with pytest.raises(ValueError, match="Equal bounds describe one fixed level"):
        ContinuousFactor(n_levels=2, lower_bound=5.0, upper_bound=5.0)
    
    # Bounds collapse after rounding
    with pytest.raises(ValueError, match="Rounding collapsed some levels into duplicates."):
        ContinuousFactor(n_levels=5, lower_bound=0.0, upper_bound=0.01, decimals=2)
    
    # Integer levels that cannot be evenly distributed
    with pytest.raises(ValueError, match="It is not possible to find"):
        ContinuousFactor(n_levels=4, lower_bound=0.0, upper_bound=1.0, decimals=0)
    
    # Rounding creates duplicate levels
    with pytest.raises(ValueError, match="Rounding collapsed some levels into duplicates."):
        ContinuousFactor(n_levels=10, lower_bound=0.0, upper_bound=0.05, decimals=2)

# ==============================================================================
#                          MixtureFactor Tests
# ==============================================================================

def test_mixture_factor():
    """Test basic mixture factor creation."""
    factor1 = MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    factor2 = MixtureFactor(lower_bound=0.2, upper_bound=0.8)
    
    assert factor1.type == "mix"
    assert factor1.lower_bound == 0.0
    assert factor1.upper_bound == 1.0
    assert factor1.decimals == 2

    assert factor2.type == "mix"
    assert factor2.lower_bound == 0.2
    assert factor2.upper_bound == 0.8
    assert factor2.decimals == 2

# ==============================================================================
#                          Integration Tests
# ==============================================================================

def test_all_factor_types_together():
    """Test that all three factor types can coexist with proper type identifiers."""
    cat_factor = CategoricalFactor(levels=["Low", "High"])
    cont_factor = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=10.0)
    mix_factor = MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    
    # Each should have unique type identifier
    assert cat_factor.type == "cat"
    assert cont_factor.type == "cont"
    assert mix_factor.type == "mix"
    
    # All types should be distinct
    types = {cat_factor.type, cont_factor.type, mix_factor.type}
    assert len(types) == 3
    factor1 = ContinuousFactor(n_levels=5, lower_bound=0.0, upper_bound=10.0, decimals=1)
    factor2 = ContinuousFactor(n_levels=3, lower_bound=-5.0, upper_bound=5.0, decimals=0)
    
    assert factor1.type == "cont"
    assert len(factor1.levels) == 5
    assert len(factor1.coded_levels) == 5
    assert factor1.lower_bound == 0.0
    assert factor1.upper_bound == 10.0
    assert factor1.n_levels == 5
    assert np.isclose(factor1.levels[0], 0.0)
    assert np.isclose(factor1.levels[1], 2.5)
    assert np.isclose(factor1.levels[2], 5.0)
    assert np.isclose(factor1.levels[3], 7.5)
    assert np.isclose(factor1.levels[4], 10.0)
    assert np.isclose(factor1.coded_levels[0], -1)
    assert np.isclose(factor1.coded_levels[1], -0.5)
    assert np.isclose(factor1.coded_levels[2], 0.0)
    assert np.isclose(factor1.coded_levels[3], 0.5)
    assert np.isclose(factor1.coded_levels[4], 1)
    
    assert factor2.type == "cont"
    assert len(factor2.levels) == 3
    assert len(factor2.coded_levels) == 3
    assert np.isclose(factor2.levels[0], -5)
    assert np.isclose(factor2.levels[1], 0)
    assert np.isclose(factor2.levels[2], 5)
    assert np.isclose(factor2.coded_levels[0], -1)
    assert np.isclose(factor2.coded_levels[1], 0)
    assert np.isclose(factor2.coded_levels[2], 1)
    assert factor2.lower_bound == -5.0
    assert factor2.upper_bound == 5.0
    assert factor2.n_levels == 3


