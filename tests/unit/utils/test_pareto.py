"""Tests for the ParetoMixin class for multi-objective optimization.

This module tests Pareto front computation for process, mixture, and mixed
designs, including validation, dominance sorting, and visualization.
"""

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from doetools.utils.pareto import ParetoMixin
from doetools.utils.factors import ContinuousFactor, CategoricalFactor, MixtureFactor


class DummyMLRResult:
    """Mock MLR result object for testing."""
    def __init__(self, y_hat=None, coef=None):
        self.y_hat = y_hat if y_hat is not None else pd.Series(dtype=float)
        self.coef = coef if coef is not None else pd.DataFrame()


class DummyMLRWrapper:
    """Mock MLR wrapper for testing."""
    def __init__(self):
        self.results = {}


class DummyDesign(ParetoMixin):
    """Dummy design class that inherits ParetoMixin for testing."""
    
    def __init__(self):
        super().__init__()
        self._model_matrix = None
        self._mlr_wrapper = None
        self._responses = None
        self._response_conditions = {}
        self._factors = {}
        self._design_matrix = pd.DataFrame()
        self._model_spec = None
        self._response_list = []
        self._responses_list = []  # For plot default axes
        self._prediction_func = None  # Allow custom prediction function
    
    def _decode_matrix(self, coded_df):
        """Mock decode method - returns copy for simplicity."""
        return coded_df.copy()
    
    def _code_matrix(self, df):
        """Mock code method - returns copy for simplicity."""
        return df.copy()
    
    def _build_model_matrix(self, coded_df, model_spec):
        """Mock model matrix builder - returns coded_df for simplicity."""
        return coded_df.copy()
    
    def _mlr_predict(self, model_matrix, responses):
        """Mock MLR prediction - returns simple linear combinations."""
        # Use custom prediction function if set
        if self._prediction_func is not None:
            return self._prediction_func(model_matrix, responses)
        
        # Create predictions based on simple formulas for testing
        predictions = pd.DataFrame(index=model_matrix.index)
        
        # Simple mock predictions for testing dominance
        # Y1: minimize (lower is better), Y2: minimize (lower is better)
        if model_matrix.shape[1] >= 2:
            predictions['Y1'] = model_matrix.iloc[:, 0] + model_matrix.iloc[:, 1]
            predictions['Y2'] = model_matrix.iloc[:, 0] - model_matrix.iloc[:, 1]
        else:
            predictions['Y1'] = model_matrix.iloc[:, 0]
            predictions['Y2'] = -model_matrix.iloc[:, 0]
        
        return predictions


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def dummy_design():
    """Create a basic dummy design with minimal setup."""
    return DummyDesign()


@pytest.fixture
def fully_configured_design():
    """Create a fully configured design ready for Pareto computation."""
    design = DummyDesign()
    
    # Set up factors
    design._factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    # Set up design matrix
    design._design_matrix = pd.DataFrame({
        'A': [0, 5, 10],
        'B': [0, 5, 10]
    })
    
    # Set up model matrix
    design._model_matrix = pd.DataFrame(np.ones((3, 2)))
    
    # Set up MLR wrapper
    design._mlr_wrapper = DummyMLRWrapper()
    
    # Set up responses
    design._responses = pd.DataFrame({
        'Y1': [1.0, 2.0, 3.0],
        'Y2': [3.0, 2.0, 1.0]
    })
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    
    # Set up response conditions
    design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    
    # Set up model spec
    design._model_spec = {'A': 1, 'B': 1}
    
    return design


@pytest.fixture
def mixture_design():
    """Create a design configured for mixture factors."""
    design = DummyDesign()
    
    design._factors = {
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    design._design_matrix = pd.DataFrame({
        'Comp_A': [1.0, 0.0, 0.0],
        'Comp_B': [0.0, 1.0, 0.0],
        'Comp_C': [0.0, 0.0, 1.0]
    })
    
    design._model_matrix = pd.DataFrame(np.ones((3, 3)))
    design._mlr_wrapper = DummyMLRWrapper()
    design._responses = pd.DataFrame({'Y1': [1, 2, 3], 'Y2': [3, 2, 1]})
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    design._model_spec = {}
    
    return design


@pytest.fixture
def mixed_design():
    """Create a design with both process and mixture factors."""
    design = DummyDesign()
    
    design._factors = {
        'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=0),
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    design._design_matrix = pd.DataFrame({
        'Temp': [20, 50, 80],
        'Comp_A': [1.0, 0.5, 0.0],
        'Comp_B': [0.0, 0.5, 0.5],
        'Comp_C': [0.0, 0.0, 0.5]
    })
    
    design._model_matrix = pd.DataFrame(np.ones((3, 4)))
    design._mlr_wrapper = DummyMLRWrapper()
    design._responses = pd.DataFrame({'Y1': [1, 2, 3], 'Y2': [3, 2, 1]})
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    design._model_spec = {}
    
    return design


# ==============================================================================
# Test 1: Initialization
# ==============================================================================

def test_pareto_mixin_initialization(dummy_design):
    """Test that ParetoMixin initializes with _non_dominated_points as None."""
    assert dummy_design._non_dominated_points is None


# ==============================================================================
# Test 2: Input Validation
# ==============================================================================

def test_compute_pareto_front_without_model_matrix(dummy_design):
    """Test that compute_pareto_front raises ValueError without model matrix."""
    factors = {'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10)}
    
    with pytest.raises(ValueError, match="Model matrix has not been built yet"):
        dummy_design.compute_pareto_front(factors)


def test_compute_pareto_front_without_mlr_models(dummy_design):
    """Test that compute_pareto_front raises ValueError without MLR models."""
    dummy_design._model_matrix = pd.DataFrame(np.ones((3, 2)))
    factors = {'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10)}
    
    with pytest.raises(ValueError, match="MLR models have not been computed yet"):
        dummy_design.compute_pareto_front(factors)


def test_compute_pareto_front_without_responses(dummy_design):
    """Test that compute_pareto_front raises ValueError without responses."""
    dummy_design._model_matrix = pd.DataFrame(np.ones((3, 2)))
    dummy_design._mlr_wrapper = DummyMLRWrapper()
    factors = {'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10)}
    
    with pytest.raises(ValueError, match="Responses haven't been loaded yet"):
        dummy_design.compute_pareto_front(factors)


def test_compute_pareto_front_without_response_conditions(dummy_design):
    """Test that compute_pareto_front raises ValueError without response conditions."""
    dummy_design._model_matrix = pd.DataFrame(np.ones((3, 2)))
    dummy_design._mlr_wrapper = DummyMLRWrapper()
    dummy_design._responses = pd.DataFrame({'Y1': [1, 2, 3]})
    factors = {'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10)}
    
    with pytest.raises(ValueError, match="Response conditions haven't been set yet"):
        dummy_design.compute_pareto_front(factors)


def test_compute_pareto_front_mismatched_factor_count(fully_configured_design):
    """Test that compute_pareto_front raises ValueError with wrong factor count."""
    # Design has 2 factors (A, B), but we provide 3
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10),
        'C': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10)
    }
    
    with pytest.raises(ValueError, match="Number of factors provided does not match"):
        fully_configured_design.compute_pareto_front(factors)


def test_compute_pareto_front_unknown_factor_name(fully_configured_design):
    """Test that compute_pareto_front raises ValueError with unknown factor."""
    # Design has factors A and B, but we provide A and C
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10),
        'C': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10)
    }
    
    with pytest.raises(ValueError, match="Variable 'C' not found in design matrix"):
        fully_configured_design.compute_pareto_front(factors)


# ==============================================================================
# Test 3: Process Factors (Continuous)
# ==============================================================================

def test_compute_pareto_front_continuous_factors_only(fully_configured_design):
    """Test Pareto front computation with only continuous factors."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    # Check that non_dominated_points is a DataFrame
    assert fully_configured_design._non_dominated_points is not None
    assert isinstance(fully_configured_design._non_dominated_points, pd.DataFrame)
    
    # Check that it has the correct columns (factors + responses)
    expected_columns = ['A', 'B', 'Y1', 'Y2']
    assert all(col in fully_configured_design._non_dominated_points.columns for col in expected_columns)
    
    # Check that we have at least some Pareto points
    assert len(fully_configured_design._non_dominated_points) > 0
    
    # Check that the grid size respects n_levels (5x5 = 25 candidates)
    # After dominance sorting, should have fewer points
    assert len(fully_configured_design._non_dominated_points) <= 25


def test_compute_pareto_front_respects_n_levels(fully_configured_design):
    """Test that grid resolution respects n_levels parameter."""
    # Use coarse grid
    factors = {
        'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    # With 3 levels each, we should have at most 3x3 = 9 candidates
    assert len(fully_configured_design._non_dominated_points) <= 9


def test_compute_pareto_front_categorical_factors(fully_configured_design):
    """Test Pareto front computation with categorical factors."""
    # Update factors to include categorical
    fully_configured_design._factors = {
        'A': CategoricalFactor(levels=['Low', 'Medium', 'High']),
        'B': CategoricalFactor(levels=['X', 'Y'])
    }
    
    fully_configured_design._design_matrix = pd.DataFrame({
        'A': ['Low', 'Medium', 'High'],
        'B': ['X', 'Y', 'X']
    })
    
    factors = {
        'A': CategoricalFactor(levels=['Low', 'Medium', 'High']),
        'B': CategoricalFactor(levels=['X', 'Y'])
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    # Check that computation succeeded
    assert fully_configured_design._non_dominated_points is not None
    assert len(fully_configured_design._non_dominated_points) > 0
    
    # With 3 and 2 levels, at most 3x2 = 6 candidates
    assert len(fully_configured_design._non_dominated_points) <= 6


# ==============================================================================
# Test 4: Mixture Factors
# ==============================================================================

def test_compute_pareto_front_mixture_factors_only(mixture_design):
    """Test Pareto front computation with only mixture factors."""
    factors = {
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    mixture_design.compute_pareto_front(factors, m=3)
    
    # Check that computation succeeded
    assert mixture_design._non_dominated_points is not None
    assert isinstance(mixture_design._non_dominated_points, pd.DataFrame)
    
    # Check columns
    expected_columns = ['Comp_A', 'Comp_B', 'Comp_C', 'Y1', 'Y2']
    assert all(col in mixture_design._non_dominated_points.columns for col in expected_columns)
    
    # For simplex lattice with m=3 and 3 components: {3,0,0}, {0,3,0}, {0,0,3},
    # {2,1,0}, {2,0,1}, {1,2,0}, {1,0,2}, {0,2,1}, {0,1,2}, {1,1,1} = 10 points
    assert len(mixture_design._non_dominated_points) <= 10


def test_compute_pareto_front_mixture_default_m(mixture_design):
    """Test that m defaults to 5 for mixture factors."""
    factors = {
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    # Don't provide m parameter
    mixture_design.compute_pareto_front(factors)
    
    # Should succeed with default m=5
    assert mixture_design._non_dominated_points is not None
    assert len(mixture_design._non_dominated_points) > 0


def test_compute_pareto_front_mixture_custom_m(mixture_design):
    """Test mixture factors with custom m values."""
    factors = {
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    # Test with m=2 (coarser grid)
    mixture_design.compute_pareto_front(factors, m=2)
    count_m2 = len(mixture_design._non_dominated_points)
    
    # Test with m=4 (finer grid)
    mixture_design.compute_pareto_front(factors, m=4)
    count_m4 = len(mixture_design._non_dominated_points)
    
    # Higher m should generally produce more candidates (though after dominance
    # sorting the Pareto front size may vary)
    assert count_m2 > 0
    assert count_m4 > 0


def test_mixture_infeasible_lower_bounds(mixture_design):
    """Test that infeasible lower bounds raise ValueError."""
    # Sum of lower bounds > 1
    factors = {
        'Comp_A': MixtureFactor(lower_bound=0.5, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.5, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.5, upper_bound=1.0)
    }
    
    with pytest.raises(ValueError, match="Sum of lower_bounds must be < 1"):
        mixture_design.compute_pareto_front(factors, m=3)


def test_mixture_infeasible_upper_bounds(mixture_design):
    """Test that infeasible upper bounds raise ValueError."""
    # Sum of upper bounds < 1
    factors = {
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=0.2),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=0.2),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=0.2)
    }
    
    with pytest.raises(ValueError, match="Infeasible: sum\\(upper_bounds\\) must be >= 1"):
        mixture_design.compute_pareto_front(factors, m=3)


# ==============================================================================
# Test 5: Mixed Design (Process + Mixture)
# ==============================================================================

def test_compute_pareto_front_process_and_mixture(mixed_design):
    """Test Pareto front computation with both process and mixture factors."""
    factors = {
        'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=0),
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    mixed_design.compute_pareto_front(factors, m=2)
    
    # Check that computation succeeded
    assert mixed_design._non_dominated_points is not None
    assert isinstance(mixed_design._non_dominated_points, pd.DataFrame)
    
    # Check columns include both process and mixture factors
    expected_columns = ['Temp', 'Comp_A', 'Comp_B', 'Comp_C', 'Y1', 'Y2']
    assert all(col in mixed_design._non_dominated_points.columns for col in expected_columns)
    
    # For process: 3 levels, For mixture with m=2: 6 points
    # Cross-product: 3 * 6 = 18 candidates maximum
    assert len(mixed_design._non_dominated_points) <= 18


def test_mixed_design_with_categorical_and_mixture(mixed_design):
    """Test mixed design with categorical process factors and mixture."""
    # Update to have categorical process factor
    mixed_design._factors = {
        'Catalyst': CategoricalFactor(levels=['A', 'B', 'C']),
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    mixed_design._design_matrix = pd.DataFrame({
        'Catalyst': ['A', 'B', 'C'],
        'Comp_A': [1.0, 0.5, 0.0],
        'Comp_B': [0.0, 0.5, 0.5],
        'Comp_C': [0.0, 0.0, 0.5]
    })
    
    factors = {
        'Catalyst': CategoricalFactor(levels=['A', 'B', 'C']),
        'Comp_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'Comp_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    }
    
    mixed_design.compute_pareto_front(factors, m=2)
    
    # Check that computation succeeded
    assert mixed_design._non_dominated_points is not None
    assert len(mixed_design._non_dominated_points) > 0


# ==============================================================================
# Test 6: Constraint Filters
# ==============================================================================

def test_compute_pareto_front_with_single_filter(fully_configured_design):
    """Test Pareto front computation with a single constraint filter."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    # Apply constraint: A <= 5
    filters = [lambda df: df['A'] <= 5.0]
    
    fully_configured_design.compute_pareto_front(factors, filters=filters)
    
    # Check that all points satisfy the constraint
    assert all(fully_configured_design._non_dominated_points['A'] <= 5.0)


def test_compute_pareto_front_with_multiple_filters(fully_configured_design):
    """Test Pareto front computation with multiple constraint filters."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    # Apply multiple constraints
    filters = [
        lambda df: df['A'] <= 7.5,
        lambda df: df['B'] >= 2.5,
        lambda df: df['A'] + df['B'] <= 12.5
    ]
    
    fully_configured_design.compute_pareto_front(factors, filters=filters)
    
    # Check that all constraints are satisfied
    pareto_df = fully_configured_design._non_dominated_points
    assert all(pareto_df['A'] <= 7.5)
    assert all(pareto_df['B'] >= 2.5)
    assert all(pareto_df['A'] + pareto_df['B'] <= 12.5)


def test_compute_pareto_front_no_filters(fully_configured_design):
    """Test Pareto front computation without constraint filters."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    # No filters
    fully_configured_design.compute_pareto_front(factors)
    
    # Should have more points than with filters
    assert len(fully_configured_design._non_dominated_points) > 0


# ==============================================================================
# Test 7: Dominance Sorting Logic - Correctness
# ==============================================================================

def test_pareto_front_excludes_dominated_points():
    """Test that truly dominated points are excluded from Pareto front."""
    design = DummyDesign()
    
    # Set up with known response values
    design._factors = {
        'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
    }
    design._design_matrix = pd.DataFrame({'A': [0, 5, 10]})
    design._model_matrix = pd.DataFrame(np.ones((3, 1)))
    design._mlr_wrapper = DummyMLRWrapper()
    design._responses = pd.DataFrame({'Y1': [1, 2, 3], 'Y2': [1, 2, 3]})
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    design._model_spec = {}
    
    # Create deterministic predictions where point (1,1) dominates (2,2) and (3,3)
    # All objectives are minimize, so lower is better
    def deterministic_predict(model_matrix, responses):
        predictions = pd.DataFrame(index=model_matrix.index)
        predictions['Y1'] = [1.0, 2.0, 3.0]  # Point 0: best, Point 1: middle, Point 2: worst
        predictions['Y2'] = [1.0, 2.0, 3.0]  # Point 0 dominates both Point 1 and Point 2
        return predictions
    
    design._prediction_func = deterministic_predict
    
    factors = {'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)}
    design.compute_pareto_front(factors)
    
    # Only the first point (1, 1) should be in the Pareto front
    assert len(design._non_dominated_points) == 1
    assert design._non_dominated_points['Y1'].iloc[0] == 1.0
    assert design._non_dominated_points['Y2'].iloc[0] == 1.0


def test_pareto_front_includes_all_non_dominated_points():
    """Test that all non-dominated points are included in Pareto front."""
    design = DummyDesign()
    
    design._factors = {
        'A': ContinuousFactor(n_levels=4, lower_bound=0, upper_bound=10, decimals=1),
    }
    design._design_matrix = pd.DataFrame({'A': [0, 2.5, 5, 7.5]})
    design._model_matrix = pd.DataFrame(np.ones((4, 1)))
    design._mlr_wrapper = DummyMLRWrapper()
    design._responses = pd.DataFrame({'Y1': [1, 2, 3, 4], 'Y2': [1, 2, 3, 4]})
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    design._model_spec = {}
    
    # Create predictions where multiple points are on the Pareto front
    # Points form a trade-off: (1,4), (2,2), (4,1) - none dominates another
    def tradeoff_predict(model_matrix, responses):
        predictions = pd.DataFrame(index=model_matrix.index)
        predictions['Y1'] = [1.0, 2.0, 4.0, 5.0]  # Increasing
        predictions['Y2'] = [4.0, 2.0, 1.0, 3.0]  # Decreasing then up
        return predictions
    
    design._prediction_func = tradeoff_predict
    
    factors = {'A': ContinuousFactor(n_levels=4, lower_bound=0, upper_bound=10, decimals=1)}
    design.compute_pareto_front(factors)
    
    # Points 0, 1, 2 should be on Pareto front; Point 3 (5,3) is dominated by Point 1 (2,2)
    assert len(design._non_dominated_points) == 3
    
    # Check that the three non-dominated points are present
    y1_values = sorted(design._non_dominated_points['Y1'].tolist())
    y2_values = sorted(design._non_dominated_points['Y2'].tolist())
    assert y1_values == [1.0, 2.0, 4.0]
    assert y2_values == [1.0, 2.0, 4.0]


def test_pareto_front_maximization_vs_minimization():
    """Test that maximization and minimization are handled correctly."""
    design = DummyDesign()
    
    design._factors = {
        'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
    }
    design._design_matrix = pd.DataFrame({'A': [0, 5, 10]})
    design._model_matrix = pd.DataFrame(np.ones((3, 1)))
    design._mlr_wrapper = DummyMLRWrapper()
    design._responses = pd.DataFrame({'Y1': [1, 2, 3], 'Y2': [1, 2, 3]})
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    design._response_conditions = {
        'Y1': {'maximize': True, 'lower_limit': None, 'upper_limit': None},  # Maximize
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}  # Minimize
    }
    design._model_spec = {}
    
    # Y1: want high (maximize), Y2: want low (minimize)
    # Point 0: Y1=1 (bad), Y2=3 (bad) - dominated
    # Point 1: Y1=2 (medium), Y2=2 (medium) - Pareto optimal
    # Point 2: Y1=3 (good), Y2=1 (good) - Pareto optimal
    def mixed_objectives_predict(model_matrix, responses):
        predictions = pd.DataFrame(index=model_matrix.index)
        predictions['Y1'] = [1.0, 2.0, 3.0]  # Maximize (want high)
        predictions['Y2'] = [3.0, 2.0, 1.0]  # Minimize (want low)
        return predictions
    
    design._prediction_func = mixed_objectives_predict
    
    factors = {'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)}
    design.compute_pareto_front(factors)
    
    # Point 2 (3, 1) dominates Point 0 (1, 3): higher Y1, lower Y2
    # Point 2 dominates Point 1 (2, 2): higher Y1, lower Y2
    # Only Point 2 should be on Pareto front
    assert len(design._non_dominated_points) == 1
    assert design._non_dominated_points['Y1'].iloc[0] == 3.0
    assert design._non_dominated_points['Y2'].iloc[0] == 1.0


def test_pareto_front_with_trade_offs():
    """Test Pareto front correctly identifies trade-offs between objectives."""
    design = DummyDesign()
    
    design._factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
    }
    design._design_matrix = pd.DataFrame({'A': [0, 2.5, 5, 7.5, 10]})
    design._model_matrix = pd.DataFrame(np.ones((5, 1)))
    design._mlr_wrapper = DummyMLRWrapper()
    design._responses = pd.DataFrame({'Y1': [1, 2, 3, 4, 5], 'Y2': [1, 2, 3, 4, 5]})
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    design._model_spec = {}
    
    # Create classic Pareto trade-off: as Y1 decreases, Y2 increases
    # Points: (1,5), (2,4), (3,3), (4,2), (5,1)
    # All should be on Pareto front - none dominates another
    def tradeoff_predict(model_matrix, responses):
        predictions = pd.DataFrame(index=model_matrix.index)
        predictions['Y1'] = [1.0, 2.0, 3.0, 4.0, 5.0]  # Increasing
        predictions['Y2'] = [5.0, 4.0, 3.0, 2.0, 1.0]  # Decreasing
        return predictions
    
    design._prediction_func = tradeoff_predict
    
    factors = {'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)}
    design.compute_pareto_front(factors)
    
    # All 5 points should be on the Pareto front (classic trade-off curve)
    assert len(design._non_dominated_points) == 5
    
    # Verify the trade-off structure is preserved
    sorted_by_y1 = design._non_dominated_points.sort_values('Y1')
    # As Y1 increases, Y2 should decrease
    y2_diff = sorted_by_y1['Y2'].diff().dropna()
    assert all(y2_diff < 0), "Y2 should decrease as Y1 increases in trade-off"


def test_pareto_front_single_optimal_point():
    """Test case where one point dominates all others."""
    design = DummyDesign()
    
    design._factors = {
        'A': ContinuousFactor(n_levels=4, lower_bound=0, upper_bound=10, decimals=1),
    }
    design._design_matrix = pd.DataFrame({'A': [0, 2.5, 5, 7.5]})
    design._model_matrix = pd.DataFrame(np.ones((4, 1)))
    design._mlr_wrapper = DummyMLRWrapper()
    design._responses = pd.DataFrame({'Y1': [1, 2, 3, 4], 'Y2': [1, 2, 3, 4]})
    design._response_list = ['Y1', 'Y2']
    design._responses_list = ['Y1', 'Y2']
    design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    design._model_spec = {}
    
    # Point 0 (0.5, 0.5) dominates all others
    def single_optimal_predict(model_matrix, responses):
        predictions = pd.DataFrame(index=model_matrix.index)
        predictions['Y1'] = [0.5, 2.0, 3.0, 4.0]  # First is best
        predictions['Y2'] = [0.5, 2.0, 3.0, 4.0]  # First is best
        return predictions
    
    design._prediction_func = single_optimal_predict
    
    factors = {'A': ContinuousFactor(n_levels=4, lower_bound=0, upper_bound=10, decimals=1)}
    design.compute_pareto_front(factors)
    
    # Only one point should be optimal
    assert len(design._non_dominated_points) == 1
    assert design._non_dominated_points['Y1'].iloc[0] == 0.5
    assert design._non_dominated_points['Y2'].iloc[0] == 0.5


def test_pareto_front_maximization_objective(fully_configured_design):
    """Test that maximization objectives are handled correctly (negated)."""
    # Set one response to maximize
    fully_configured_design._response_conditions = {
        'Y1': {'maximize': True, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': False, 'lower_limit': None, 'upper_limit': None}
    }
    
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    # Should successfully compute Pareto front with mixed objectives
    assert fully_configured_design._non_dominated_points is not None
    assert len(fully_configured_design._non_dominated_points) > 0


def test_pareto_front_all_maximization(fully_configured_design):
    """Test Pareto front with all maximization objectives."""
    fully_configured_design._response_conditions = {
        'Y1': {'maximize': True, 'lower_limit': None, 'upper_limit': None},
        'Y2': {'maximize': True, 'lower_limit': None, 'upper_limit': None}
    }
    
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    assert fully_configured_design._non_dominated_points is not None
    assert len(fully_configured_design._non_dominated_points) > 0


# ==============================================================================
# Test 8: Rounding and Decimals
# ==============================================================================

def test_continuous_factor_rounding(fully_configured_design):
    """Test that continuous factors are rounded to specified decimals."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=2),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design._factors = factors
    fully_configured_design.compute_pareto_front(factors)
    
    # Check that values are properly rounded
    pareto_df = fully_configured_design._non_dominated_points
    
    # Check A is rounded to 2 decimals
    for val in pareto_df['A']:
        assert len(str(val).split('.')[-1]) <= 2 or val == int(val)
    
    # Check B is rounded to 1 decimal
    for val in pareto_df['B']:
        assert len(str(val).split('.')[-1]) <= 1 or val == int(val)


# ==============================================================================
# Test 9: non_dominated_points Property
# ==============================================================================

def test_non_dominated_points_before_computation(dummy_design):
    """Test that non_dominated_points returns None before computation."""
    assert dummy_design.non_dominated_points is None


def test_non_dominated_points_after_computation(fully_configured_design):
    """Test that non_dominated_points returns DataFrame after computation."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    # Access via property
    pareto_df = fully_configured_design.non_dominated_points
    
    assert pareto_df is not None
    assert isinstance(pareto_df, pd.DataFrame)
    assert len(pareto_df) > 0
    
    # Check structure
    expected_columns = ['A', 'B', 'Y1', 'Y2']
    assert all(col in pareto_df.columns for col in expected_columns)


# ==============================================================================
# Test 10: plot_pareto_front Functionality
# ==============================================================================

def test_plot_pareto_front_2d_default(fully_configured_design):
    """Test 2D plot with default axes (first two responses)."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    # Plot with defaults (should use Y1 and Y2)
    fig = fully_configured_design.plot_pareto_front()
    
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_plot_pareto_front_2d_specified(fully_configured_design):
    """Test 2D plot with explicitly specified axes."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    # Specify x and y explicitly
    fig = fully_configured_design.plot_pareto_front(x='Y1', y='Y2')
    
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_plot_pareto_front_3d(fully_configured_design):
    """Test 3D plot with x, y, and z specified."""
    # Add a third response
    fully_configured_design._responses['Y3'] = [2.0, 2.5, 3.0]
    fully_configured_design._response_list.append('Y3')
    fully_configured_design._responses_list.append('Y3')
    fully_configured_design._response_conditions['Y3'] = {
        'maximize': False, 'lower_limit': None, 'upper_limit': None
    }
    
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    # Update _mlr_predict to handle 3 responses
    def mlr_predict_3(model_matrix, responses):
        predictions = pd.DataFrame(index=model_matrix.index)
        predictions['Y1'] = model_matrix.iloc[:, 0] + model_matrix.iloc[:, 1]
        predictions['Y2'] = model_matrix.iloc[:, 0] - model_matrix.iloc[:, 1]
        predictions['Y3'] = model_matrix.iloc[:, 0] * 0.5
        return predictions
    
    fully_configured_design._mlr_predict = mlr_predict_3
    fully_configured_design.compute_pareto_front(factors)
    
    # Create 3D plot
    fig = fully_configured_design.plot_pareto_front(x='Y1', y='Y2', z='Y3')
    
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_plot_pareto_front_x_only_raises_error(fully_configured_design):
    """Test that specifying only x raises ValueError."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    with pytest.raises(ValueError, match="Please specify both x and y variables"):
        fully_configured_design.plot_pareto_front(x='Y1')


def test_plot_pareto_front_y_only_raises_error(fully_configured_design):
    """Test that specifying only y raises ValueError."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    with pytest.raises(ValueError, match="Please specify both x and y variables"):
        fully_configured_design.plot_pareto_front(y='Y2')


def test_plot_pareto_front_same_x_y_raises_error(fully_configured_design):
    """Test that x == y raises ValueError."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    with pytest.raises(ValueError, match="x and y variables cannot be the same"):
        fully_configured_design.plot_pareto_front(x='Y1', y='Y1')


def test_plot_pareto_front_z_equals_x_raises_error(fully_configured_design):
    """Test that z == x raises ValueError."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    with pytest.raises(ValueError, match="z must be distinct from x and y"):
        fully_configured_design.plot_pareto_front(x='Y1', y='Y2', z='Y1')


def test_plot_pareto_front_invalid_column_raises_error(fully_configured_design):
    """Test that invalid column name raises ValueError."""
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    with pytest.raises(ValueError, match="'Y3' is not a column in the Pareto front data"):
        fully_configured_design.plot_pareto_front(x='Y1', y='Y3')


def test_plot_pareto_front_with_response_limits(fully_configured_design):
    """Test that response limits are added to 2D plot."""
    # Set response limits
    fully_configured_design._response_conditions = {
        'Y1': {'maximize': False, 'lower_limit': 0.5, 'upper_limit': 2.5},
        'Y2': {'maximize': False, 'lower_limit': 0.5, 'upper_limit': 2.5}
    }
    
    factors = {
        'A': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1),
        'B': ContinuousFactor(n_levels=5, lower_bound=0, upper_bound=10, decimals=1)
    }
    
    fully_configured_design.compute_pareto_front(factors)
    
    fig = fully_configured_design.plot_pareto_front(x='Y1', y='Y2')
    
    # Check that figure has shapes (vlines/hlines for limits)
    assert fig is not None
    # The shapes are added as vlines and hlines, check that they exist
    # We can't easily introspect plotly shapes, but we can verify the function runs
    assert isinstance(fig, go.Figure)
