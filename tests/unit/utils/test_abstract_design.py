import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from doetools.utils.abstract_design import Design
from doetools.utils.factors import ContinuousFactor, CategoricalFactor, MixtureFactor
from doetools.utils.model_spec import ModelTerms


# =============================================================================
#                           Mock Design Class
# =============================================================================

class MockDesign(Design):
    """
    Minimal concrete implementation of Design for testing.
    
    This mock class allows testing of the abstract Design class methods
    without depending on actual design implementations like FullFactorial
    or CentralComposite.
    """
    
    def __init__(self):
        super().__init__()
        self._design_type = "Mock"


# =============================================================================
#                           Test Fixtures
# =============================================================================

@pytest.fixture
def simple_continuous_factors() -> Dict[str, Any]:
    """Create a simple dictionary of continuous factors for testing."""
    return {
        "A": ContinuousFactor(n_levels=3, lower_bound=10.0, upper_bound=20.0, decimals=1),
        "B": ContinuousFactor(n_levels=3, lower_bound=50.0, upper_bound=100.0, decimals=1),
    }
    
@pytest.fixture
def simple_categorical_factors() -> Dict[str, Any]:
    """Create a simple dictionary of categorical factors for testing."""
    return {
        "A": CategoricalFactor(levels=["Low", "Medium", "High"]),
        "B": CategoricalFactor(levels=["Type1", "Type2", "Type3"]),
    }

@pytest.fixture
def frf_2levels_3factors() -> Dict[str, Any]:
    """Create a simple dictionary of continuous factors for fractional factorial testing."""
    return {
        "A": ContinuousFactor(n_levels=2, lower_bound=0.0, upper_bound=1.0, decimals=2),
        "B": ContinuousFactor(n_levels=2, lower_bound=5.0, upper_bound=10.0, decimals=1),
        "C": ContinuousFactor(n_levels=2, lower_bound=100.0, upper_bound=200.0, decimals=0),
    }

@pytest.fixture
def mixed_factors() -> Dict[str, Any]:
    """Create a dictionary with continuous and categorical factors."""
    return {
        "Temp": ContinuousFactor(n_levels=3, lower_bound=100.0, upper_bound=200.0, decimals=0),
        "Pressure": ContinuousFactor(n_levels=2, lower_bound=1.0, upper_bound=5.0, decimals=1),
        "Catalyst": CategoricalFactor(levels=["Type_A", "Type_B", "Type_C"]),
    }
    
@pytest.fixture
def unconstrained_mixture_factors() -> Dict[str, Any]:
    """Create mixture factors without upper bound constraints."""
    return {
        'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
    }


@pytest.fixture
def mixture_factors() -> Dict[str, Any]:
    """Create a dictionary of mixture factors."""
    return {
        "X1": MixtureFactor(lower_bound=0.1, upper_bound=0.6, decimals=2),
        "X2": MixtureFactor(lower_bound=0.2, upper_bound=0.7, decimals=2),
        "X3": MixtureFactor(lower_bound=0.1, upper_bound=0.5, decimals=2),
    }


@pytest.fixture
def ff_2factors_3levels_coded_design_matrix() -> pd.DataFrame:
    """Create a simple coded design matrix for testing."""
    return pd.DataFrame({
        "A": [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "B": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
    })

@pytest.fixture
def frf_3factors_2levels_coded_design_matrix() -> pd.DataFrame:
    """Create a simple fractional factorial coded design matrix for testing."""
    return pd.DataFrame({
        "A": [-1.0, -1.0, 1.0, 1.0],
        "B": [-1.0, 1.0, -1.0, 1.0],
        "C": [-1.0, 1.0, 1.0, -1.0],
    })

@pytest.fixture
def unconstrained_simplex_design_matrix() -> pd.DataFrame:
    """Create a simple unconstrained simplex design matrix for testing."""
    return pd.DataFrame({
        "X1": [0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 1/3],
        "X2": [0.0, 1.0, 0.0, 0.5, 0.5, 0, 1/3],
        "X3": [1.0, 0.0, 0.0, 0, 0.5, 0.5, 1/3],
    })


@pytest.fixture
def simple_design_with_replicates() -> pd.DataFrame:
    """Create a coded design matrix with replicate runs."""
    return pd.DataFrame({
        "A": [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0], # Last three are replicates
        "B": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
    })


@pytest.fixture
def mock_design_ff(simple_continuous_factors, ff_2factors_3levels_coded_design_matrix):
    """Create a basic mock design with continuous factors."""
    design = MockDesign()
    design._factors = simple_continuous_factors
    design._coded_design_matrix = ff_2factors_3levels_coded_design_matrix.copy()
    design._design_matrix = design._decode_matrix(design._coded_design_matrix)
    return design

@pytest.fixture
def mock_design_frf(frf_2levels_3factors, frf_3factors_2levels_coded_design_matrix):
    """Create a basic mock design with continuous factors."""
    design = MockDesign()
    design._factors = frf_2levels_3factors
    design._coded_design_matrix = frf_3factors_2levels_coded_design_matrix.copy()
    design._design_matrix = design._decode_matrix(design._coded_design_matrix)
    return design

@pytest.fixture
def mock_design_ff_with_model(mock_design_ff):
    """Create a mock design with model terms set."""
    design = mock_design_ff
    terms = ModelTerms(
        intercept=True,
        pro_main=["A", "B"],
        pro_int2=[("A", "B")],
        pro_quadratic=[]
    )
    design.set_model_terms(terms)
    return design


@pytest.fixture
def sample_responses() -> pd.DataFrame:
    """Create sample response data."""
    return pd.DataFrame({
        "Yield": [45.2, 78.5, 62.3, 89.1, 55.8, 92.4, 88.0, 76.5, 95.0],
        "Purity": [88.5, 91.2, 85.7, 94.5, 87.3, 95.8, 90.1, 89.4, 93.2],
    })


# =============================================================================
#                           Test: Initialization
# =============================================================================

class TestInitialization:
    """Test the initialization of the Design class."""
    
    def test_mock_design_instantiation(self):
        """Test that MockDesign can be instantiated."""
        design = MockDesign()
        assert design is not None
        assert isinstance(design, Design)
    
    def test_initial_attributes(self):
        """Test that all attributes are initialized correctly."""
        design = MockDesign()
        assert design._factors is None
        assert design._design_type == "Mock"
        assert design._replicates == 0
        assert design._center_points == 0
        assert design._coded_design_matrix is None
        assert design._design_matrix is None
        assert design._model_spec is None
        assert design._model_matrix is None
        assert design._leverages is None
        assert design._response_list is None
        assert design._responses is None
        assert design._response_conditions == {}
        assert design._mlr_wrapper is None


# =============================================================================
#                     Test: Coding/Decoding Methods
# =============================================================================

class TestCodingDecoding:
    """Test the coding and decoding transformation methods."""
    
    def test_decode_continuous_factors(self, simple_continuous_factors):    
        """Test decoding continuous factors from coded to actual values."""
        design = MockDesign()
        design._factors = simple_continuous_factors
        
        coded_matrix = pd.DataFrame({
            "A": [-1.0, 0.0, 1.0],
            "B": [-1.0, 0.0, 1.0],
        })
        
        decoded = design._decode_matrix(coded_matrix)
        
        # A: 10 to 20, center = 15
        assert np.isclose(decoded.loc[0, "A"], 10.0)
        assert np.isclose(decoded.loc[1, "A"], 15.0)
        assert np.isclose(decoded.loc[2, "A"], 20.0)
        
        # B: 50 to 100, center = 75
        assert np.isclose(decoded.loc[0, "B"], 50.0)
        assert np.isclose(decoded.loc[1, "B"], 75.0)
        assert np.isclose(decoded.loc[2, "B"], 100.0)
    
    def test_code_continuous_factors(self, simple_continuous_factors):
        """Test coding continuous factors from actual to coded values."""
        design = MockDesign()
        design._factors = simple_continuous_factors
        
        actual_matrix = pd.DataFrame({
            "A": [10.0, 15.0, 20.0],
            "B": [50.0, 75.0, 100.0],
        })
        
        coded = design._code_matrix(actual_matrix)
        
        assert np.isclose(coded.loc[0, "A"], -1.0)
        assert np.isclose(coded.loc[1, "A"], 0.0)
        assert np.isclose(coded.loc[2, "A"], 1.0)
        
        assert np.isclose(coded.loc[0, "B"], -1.0)
        assert np.isclose(coded.loc[1, "B"], 0.0)
        assert np.isclose(coded.loc[2, "B"], 1.0)
    
    def test_decode_categorical_factors(self):
        """Test decoding categorical factors."""
        design = MockDesign()
        design._factors = {
            "Cat": CategoricalFactor(levels=["Low", "Med", "High"])
        }
        
        coded_matrix = pd.DataFrame({
            "Cat": [-1.0, 0.0, 1.0],
        })
        
        decoded = design._decode_matrix(coded_matrix)
        
        assert decoded.loc[0, "Cat"] == "Low"
        assert decoded.loc[1, "Cat"] == "Med"
        assert decoded.loc[2, "Cat"] == "High"
    
    def test_code_categorical_factors(self):
        """Test coding categorical factors."""
        design = MockDesign()
        design._factors = {
            "Cat": CategoricalFactor(levels=["Low", "Med", "High"])
        }
        
        actual_matrix = pd.DataFrame({
            "Cat": ["Low", "Med", "High"],
        })
        
        coded = design._code_matrix(actual_matrix)
        
        assert np.isclose(coded.loc[0, "Cat"], -1.0)
        assert np.isclose(coded.loc[1, "Cat"], 0.0)
        assert np.isclose(coded.loc[2, "Cat"], 1.0)
    
    def test_decode_mixture_factors(self, mixture_factors, unconstrained_simplex_design_matrix):
        """Test decoding mixture factors."""
        design = MockDesign()
        design._factors = mixture_factors
        
        coded_matrix = unconstrained_simplex_design_matrix
        
        decoded = design._decode_matrix(coded_matrix)
        
        pd.testing.assert_frame_equal(decoded, coded_matrix)
    
    def test_code_mixture_factors(self, mixture_factors, unconstrained_simplex_design_matrix):
        """Test coding mixture factors."""
        design = MockDesign()
        design._factors = mixture_factors
        
        real_matrix = unconstrained_simplex_design_matrix
        
        coded = design._code_matrix(real_matrix)
        
        pd.testing.assert_frame_equal(coded, real_matrix)
    
    def test_code_decode_roundtrip(self, simple_continuous_factors, ff_2factors_3levels_coded_design_matrix):
        """Test that coding and decoding are inverse operations."""
        design = MockDesign()
        design._factors = simple_continuous_factors
        
        decoded = design._decode_matrix(ff_2factors_3levels_coded_design_matrix)
        recoded = design._code_matrix(decoded)
        
        pd.testing.assert_frame_equal(recoded, ff_2factors_3levels_coded_design_matrix)
    
    def test_code_dict_continuous(self, simple_continuous_factors):
        """Test coding a dictionary of continuous factor levels."""
        design = MockDesign()
        design._factors = simple_continuous_factors
        
        levels = {"A": 15.0, "B": 75.0}
        coded_levels = design._code_dict(levels)
        
        assert np.isclose(coded_levels["A"], 0.0)
        assert np.isclose(coded_levels["B"], 0.0)
    
    def test_code_dict_none(self):
        """Test that code_dict returns None when given None."""
        design = MockDesign()
        assert design._code_dict(None) is None


# =============================================================================
#                     Test: Center Points
# =============================================================================

class TestCenterPoints:
    """Test center point related methods."""
    
    def test_add_center_points_continuous(self, simple_continuous_factors):
        """Test adding center points to a design with continuous factors."""
        design = MockDesign()
        design._factors = simple_continuous_factors
        
        initial_matrix = pd.DataFrame({
            "A": [-1.0, 1.0],
            "B": [-1.0, 1.0],
        })
        
        result = design._add_center_points(initial_matrix, center_points=3)
        
        assert result.shape[0] == 5  # 2 original + 3 center points
        
        # Check that last 3 rows are center points
        for i in range(3):
            assert np.isclose(result.iloc[2 + i]["A"], 0.0)
            assert np.isclose(result.iloc[2 + i]["B"], 0.0)
    
    def test_number_of_center_points(self, simple_continuous_factors, simple_categorical_factors, unconstrained_mixture_factors):
        """Test counting center points in a design."""
        design_1 = MockDesign()
        design_1._factors = simple_continuous_factors
        design_2 = MockDesign()
        design_2._factors = simple_categorical_factors
        design_3 = MockDesign()
        design_3._factors = unconstrained_mixture_factors
        
        matrix_1 = pd.DataFrame({
            "A": [-1.0, 0.0, 0.0, 1.0],
            "B": [-1.0, 0.0, 0.0, 1.0],
        })
        
        matrix_2 = pd.DataFrame({
            "A": [-1.0, 1.0, -1.0, 1.0],
            "B": [-1.0, -1.0, 1.0, 1.0],
        })
        
        matrix_3 = pd.DataFrame({
            "X1": [1.0, 0.0, 0.0, 1/3],
            "X2": [0.0, 1.0, 0.0, 1/3],
            "X3": [0.0, 0.0, 1.0, 1/3],
        })
        
        count1 = design_1._number_of_center_points(matrix_1)
        assert count1 == 2
        count2 = design_2._number_of_center_points(matrix_2)
        assert count2 == 1
        count3 = design_3._number_of_center_points(matrix_3)
        assert count3 == 1
    
    def test_center_points_categorical(self, simple_categorical_factors):
        """Test that center points use baseline level for categorical factors."""
        design = MockDesign()
        design._factors = {
            "A": ContinuousFactor(n_levels=3, lower_bound=10.0, upper_bound=20.0),
            "Cat": CategoricalFactor(levels=["Low", "High"])
        }
        
        initial_matrix = pd.DataFrame({
            "A": [-1.0, 1.0],
            "Cat": [-1.0, 1.0],
        })
        
        result = design._add_center_points(initial_matrix, center_points=1)
        
        # Center point should have A=0 and Cat=-1 (baseline)
        assert np.isclose(result.iloc[2]["A"], 0.0)
        assert np.isclose(result.iloc[2]["Cat"], -1.0)

    def test_center_points_use_configured_categorical_reference(self):
        """Adding and counting center points uses each categorical reference."""
        design = MockDesign()
        design._factors = {
            "A": ContinuousFactor(n_levels=3, lower_bound=10.0, upper_bound=20.0),
            "Cat": CategoricalFactor(
                levels=["Low", "Medium", "High"],
                reference_level="High",
            ),
        }
        initial_matrix = pd.DataFrame({"A": [-1.0, 1.0], "Cat": [-1.0, 0.0]})

        result = design._add_center_points(initial_matrix, center_points=2)

        assert np.allclose(result.iloc[-2:]["A"], 0.0)
        assert np.allclose(result.iloc[-2:]["Cat"], 1.0)
        assert design._number_of_center_points(result) == 2
    
    def test_center_points_unconstrained_simplex(self):
        """Test that center points are at the center of the simplex for mixture factors."""
        design = MockDesign()
        design._factors = {
            "A" : MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            "B" : MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            "C" : MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        }
        
        initial_matrix = pd.DataFrame({
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 1.0],
        })
        
        result = design._add_center_points(initial_matrix, center_points=1)
        
        # Center point should have A=0 and Cat=-1 (baseline)
        assert np.isclose(result.iloc[3]["A"], 1/3)
        assert np.isclose(result.iloc[3]["B"], 1/3)
        assert np.isclose(result.iloc[3]["C"], 1/3)
    
    def test_center_points_constrained_simplex(self):
        """Test that center points are at the center of the simplex for mixture factors."""
        design = MockDesign()
        design._factors = {
            "A" : MixtureFactor(lower_bound=0.2, upper_bound=1.0),
            "B" : MixtureFactor(lower_bound=0.2, upper_bound=1.0),
            "C" : MixtureFactor(lower_bound=0.2, upper_bound=1.0),
        }
        
        initial_matrix = pd.DataFrame({
            "A": [0.8, 0.1, 0.1],
            "B": [0.1, 0.8, 0.1],
            "C": [0.1, 0.1, 0.8],
        })
        
        result = design._add_center_points(initial_matrix, center_points=1)
        
        # Center point should have A=0 and Cat=-1 (baseline)
        assert np.isclose(result.iloc[3]["A"], 1/3)
        assert np.isclose(result.iloc[3]["B"], 1/3)
        assert np.isclose(result.iloc[3]["C"], 1/3)



# =============================================================================
#                     Test: Model Matrix Building
# =============================================================================

class TestModelMatrix:
    """Test model matrix construction methods."""
    
    def test_build_model_matrix_main_only(self, mock_design_ff):
        """Test building model matrix with main effects only."""
        design = mock_design_ff
        
        from doetools.utils.model_spec import ModelSpec
        model_spec = ModelSpec(
            intercept=True,
            main=["A", "B"],
            interaction2=[],
            quadratic=[], 
            interaction3=[]
        )
        
        model_matrix = design._build_model_matrix(design._coded_design_matrix, model_spec)
        
        assert "Int" in model_matrix.columns
        assert "A" in model_matrix.columns
        assert "B" in model_matrix.columns
        assert model_matrix.shape[1] == 3  # Int, A, B
        assert all(model_matrix["Int"] == 1)
    
    def test_build_model_matrix_with_interaction(self, mock_design_ff):
        """Test building model matrix with two-way interactions."""
        design = mock_design_ff
        
        from doetools.utils.model_spec import ModelSpec
        model_spec = ModelSpec(
            intercept=True,
            main=["A", "B"],
            interaction2=[("A", "B")],
            quadratic=[],
            interaction3=[]
        )
        
        model_matrix = design._build_model_matrix(design._coded_design_matrix, model_spec)
        
        assert "A:B" in model_matrix.columns
        assert model_matrix.shape[1] == 4  # Int, A, B, A:B
        
        # Verify interaction is computed correctly
        expected_interaction = design._coded_design_matrix["A"] * design._coded_design_matrix["B"]
        pd.testing.assert_series_equal(model_matrix["A:B"], expected_interaction, check_names=False)
    
    def test_build_model_matrix_with_quadratic(self, mock_design_ff):
        """Test building model matrix with quadratic terms."""
        design = mock_design_ff
        
        from doetools.utils.model_spec import ModelSpec
        model_spec = ModelSpec(
            intercept=True,
            main=["A", "B"],
            interaction2=[],
            quadratic=["A", "B"],
            interaction3=[]
        )
        
        model_matrix = design._build_model_matrix(design._coded_design_matrix, model_spec)
        
        assert "A^2" in model_matrix.columns
        assert "B^2" in model_matrix.columns
        assert model_matrix.shape[1] == 5  # Int, A, B, A^2, B^2
        
        # Verify quadratic is computed correctly
        expected_A2 = design._coded_design_matrix["A"] ** 2
        pd.testing.assert_series_equal(model_matrix["A^2"], expected_A2, check_names=False)
        expected_B2 = design._coded_design_matrix["B"] ** 2
        pd.testing.assert_series_equal(model_matrix["B^2"], expected_B2, check_names=False)

    def test_build_model_matrix_with_interaction3(self, mock_design_ff):
        """Test building model matrix with three-way interactions."""
        design = mock_design_ff
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix["C"] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        
        from doetools.utils.model_spec import ModelSpec
        model_spec = ModelSpec(
            intercept=True,
            main=["A", "B", "C"],
            interaction2=[],
            quadratic=[],   
            interaction3=[("A", "B", "A")]
        )
        
        model_matrix = design._build_model_matrix(design._coded_design_matrix, model_spec)
        
        assert "A:B:A" in model_matrix.columns
        assert model_matrix.shape[1] == 5  # Int, A, B, C, A:B:A
        
        # Verify three-way interaction is computed correctly
        expected_interaction3 = design._coded_design_matrix["A"] * design._coded_design_matrix["B"] * design._coded_design_matrix["A"]
        pd.testing.assert_series_equal(model_matrix["A:B:A"], expected_interaction3, check_names=False)
        
# =============================================================================
#                     Test: Replicates
# =============================================================================

class TestReplicates:
    """Test replicate identification and counting methods."""
    
    def test_group_by_replicates(self, simple_design_with_replicates):
        """Test identification of replicate groups."""
        design = MockDesign()
        design._coded_design_matrix = simple_design_with_replicates
        
        groups = design._group_by_replicates()
        
        # Should find 3 groups: [0,9], [4,10], [8,11]
        assert len(groups) == 3
        assert [0, 9] in groups or [9, 0] in groups
        assert [4, 10] in groups or [10, 4] in groups
        assert [8, 11] in groups or [11, 8] in groups
    
    def test_number_of_replicates(self, simple_design_with_replicates):
        """Test counting total number of replicates."""
        design = MockDesign()
        design._coded_design_matrix = simple_design_with_replicates
        
        n_replicates = design._number_of_replicates()
        
        # 3 groups with 2 points each = 3 replicates total (1 per group)
        assert n_replicates == 3
    
    def test_no_replicates(self, ff_2factors_3levels_coded_design_matrix):
        """Test with no replicates present."""
        design = MockDesign()
        design._coded_design_matrix = ff_2factors_3levels_coded_design_matrix
        
        groups = design._group_by_replicates()
        n_replicates = design._number_of_replicates()
        
        assert len(groups) == 0
        assert n_replicates == 0


# =============================================================================
#                     Test: Set Model Terms
# =============================================================================

class TestSetModelTerms:
    """Test the set_model_terms method."""
    
    def test_set_model_terms_basic(self, mock_design_ff):
        
        """Test setting model terms for a basic design."""
        design = mock_design_ff
        
        terms = ModelTerms(
            intercept=True,
            pro_main=["A", "B"],
            pro_int2=[],
            pro_quadratic=[]
        )
        
        design.set_model_terms(terms)
        
        assert design._model_spec is not None
        assert design._model_matrix is not None
        assert design._model_matrix.shape[0] == design._coded_design_matrix.shape[0]
    
    def test_set_model_terms_overdetermined(self, mock_design_frf):
        """Test that error is raised when model has too many terms."""
        design = mock_design_frf
        
        # Try to fit a model with more terms than observations
        # Design has 4 runs, so requesting too many interactions should fail
        terms = ModelTerms(
            intercept=True,
            pro_main=["A", "B", "C"],
            pro_int2=[("A", "B"), ("A", "C"), ("B", "C")],
            pro_quadratic=["A", "B", "C"],
            pro_int3=[("A", "B", "C")]
        )

        try:
            design.set_model_terms(terms)
        except ValueError as e:
            assert "exceeds the number of experiments" in str(e)


# =============================================================================
#                     Test: Add Replicates
# =============================================================================

class TestAddReplicates:
    """Test the add_replicates method."""
    
    def test_add_replicates_all(self, mock_design_ff):
        """Test adding replicates to all design points."""
        design = mock_design_ff
        initial_size = design._coded_design_matrix.shape[0]
        
        design.add_replicates(type="all", n_replicates=2)
        
        # Should triple the design size (original + 2 replicates each)
        assert design._coded_design_matrix.shape[0] == initial_size * 3
        assert design._design_matrix.shape[0] == initial_size * 3
    
    def test_add_replicates_center(self, mock_design_ff):
        """Test adding center point replicates."""
        design = mock_design_ff
        initial_size = design._coded_design_matrix.shape[0]
        
        design.add_replicates(type="center", n_replicates=3)
        
        # Should add 3 center points
        assert design._coded_design_matrix.shape[0] == initial_size + 3
        
        # Check that last 3 points are center points
        for i in range(3):
            assert np.isclose(design._coded_design_matrix.iloc[-(i+1)]["A"], 0.0)
            assert np.isclose(design._coded_design_matrix.iloc[-(i+1)]["B"], 0.0)
    
    def test_add_replicates_manual(self, mock_design_ff):
        """Test manually adding replicates for specific indices."""
        design = mock_design_ff
        initial_matrix = design._coded_design_matrix.copy()
        initial_size = design._coded_design_matrix.shape[0]
        
        design.add_replicates(type="manual", n_replicates=2, indices=[0, 2])
        
        # Should add 4 replicates (2 points × 2 replicates each)
        assert design._coded_design_matrix.shape[0] == initial_size + 4
        assert initial_matrix.iloc[0].equals(design._coded_design_matrix.iloc[initial_size])
        assert initial_matrix.iloc[0].equals(design._coded_design_matrix.iloc[initial_size + 1])
        assert initial_matrix.iloc[2].equals(design._coded_design_matrix.iloc[initial_size + 2])
        assert initial_matrix.iloc[2].equals(design._coded_design_matrix.iloc[initial_size + 3])
    
    def test_add_replicates_leverage(self, mock_design_ff_with_model):
        """Test adding replicates based on leverage."""
        design = mock_design_ff_with_model
        initial_size = design._coded_design_matrix.shape[0]
        
        design.add_replicates(type="leverage", n_replicates=2)
        
        # Should add 2 high-leverage points
        assert design._coded_design_matrix.shape[0] == initial_size + 2
    
    def test_add_replicates_invalid_type(self, mock_design_ff):
        """Test that invalid type raises error."""
        design = mock_design_ff
        
        with pytest.raises(ValueError, match="Type must be one of"):
            design.add_replicates(type="invalid", n_replicates=1)
    
    def test_add_replicates_negative(self, mock_design_ff):
        """Test that negative replicates raises error."""
        design = mock_design_ff
        
        with pytest.raises(ValueError, match="Replicates must be a non-negative integer"):
            design.add_replicates(type="all", n_replicates=-1)


# =============================================================================
#                     Test: Response Conditions
# =============================================================================

class TestResponseConditions:
    """Test the set_response_conditions method."""
    
    def test_set_response_conditions_valid(self, mock_design_ff):
        """Test setting valid response conditions."""
        design = mock_design_ff
        design._response_list = ["Yield", "Purity"]
        
        design.set_response_conditions(
            lower_limits=[50.0, False],
            upper_limits=[False, 95.0],
            maximize=[True, True]
        )
        
        assert "Yield" in design._response_conditions
        assert "Purity" in design._response_conditions
        assert design._response_conditions["Yield"]["lower_limit"] == 50.0
        assert design._response_conditions["Purity"]["upper_limit"] == 95.0
        assert design._response_conditions["Yield"]["maximize"] is True
    
    def test_set_response_conditions_mismatch(self, mock_design_ff):
        """Test that mismatched condition lengths raise error."""
        design = mock_design_ff
        design._response_list = ["Yield", "Purity"]
        
        with pytest.raises(ValueError, match="number of conditions"):
            design.set_response_conditions(
                lower_limits=[50.0],  # Only 1 instead of 2
                upper_limits=[False, 95.0],
                maximize=[True, True]
            )


# =============================================================================
#                     Test: Export/Import Experiments
# =============================================================================

def test_import_responses_accepts_dataframe_defensively(mock_design_ff):
    design = mock_design_ff
    design._response_list = ["Yield"]
    source = pd.DataFrame(
        {
            "Exp. Idx": list(reversed(range(9))),
            "Yield": [float(value) for value in range(9)],
        }
    )

    design.import_responses(source=source)
    source.loc[8, "Yield"] = 999.0

    assert list(design._responses["Yield"]) == list(reversed(range(9)))


@pytest.mark.skipif(
    sys.platform == "win32" and "OneDrive" in Path.cwd().parts,
    reason="Excel export/import tests hang intermittently in a Windows/OneDrive checkout",
)
class TestExportImport:
    """Test export_experiments and import_responses methods."""

    def test_export_experiments_basic(self, mock_design_ff, tmp_path):
        """Test basic export of experiments to Excel."""
        design = mock_design_ff
        output_file = tmp_path / "test_export.xlsx"
        
        design.export_experiments(
            responses=["Yield", "Purity"],
            randomize=False,
            destination=str(output_file),
            coded=False
        )
        
        # Verify file was created
        assert output_file.exists()
        
        # Read back the file and verify structure
        df = pd.read_excel(output_file)
        
        assert "Exp. Order" in df.columns
        assert "Exp. Idx" in df.columns
        assert "A" in df.columns
        assert "B" in df.columns
        assert "Yield" in df.columns
        assert "Purity" in df.columns
        assert df.shape[0] == 9  # Should have 9 experiments
        
        # Verify response columns are zero
        assert all(df["Yield"] == 0.0)
        assert all(df["Purity"] == 0.0)
        
        # Verify experimental order is sequential (not randomized)
        assert list(df["Exp. Order"]) == list(range(9))
    
    def test_export_experiments_coded(self, mock_design_ff, tmp_path):
        """Test export with coded factor values."""
        design = mock_design_ff
        output_file = tmp_path / "test_export_coded.xlsx"
        
        design.export_experiments(
            responses=["Yield"],
            randomize=False,
            destination=str(output_file),
            coded=True
        )
        
        df = pd.read_excel(output_file)
        
        # Verify coded values are in the [-1, 1] range
        assert df["A"].min() == -1.0
        assert df["A"].max() == 1.0
        assert df["B"].min() == -1.0
        assert df["B"].max() == 1.0
    
    def test_export_experiments_randomized(self, mock_design_ff, tmp_path):
        """Test export with randomized run order."""
        design = mock_design_ff
        output_file = tmp_path / "test_export_random.xlsx"
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        design.export_experiments(
            responses=["Yield"],
            randomize=True,
            destination=str(output_file),
            coded=False
        )
        
        df = pd.read_excel(output_file)
        
        # Verify that Exp. Order exists and contains a valid permutation
        assert "Exp. Order" in df.columns
        # All values from 0-8 should be present (valid permutation)
        assert set(df["Exp. Order"]) == set(range(9))
        # Verify order is actually randomized (not sequential)
        assert list(df["Exp. Idx"]) != list(range(9))
    
    def test_export_experiments_no_responses(self, mock_design_ff, tmp_path):
        """Test export without response columns."""
        design = mock_design_ff
        output_file = tmp_path / "test_export_no_resp.xlsx"
        
        design.export_experiments(
            responses=None,
            randomize=False,
            destination=str(output_file),
            coded=False
        )
        
        df = pd.read_excel(output_file)
        
        # Should only have order, index, and factor columns
        assert "Exp. Order" in df.columns
        assert "Exp. Idx" in df.columns
        assert "A" in df.columns
        assert "B" in df.columns
        assert "Yield" not in df.columns
    
    def test_import_responses_valid(self, mock_design_ff, tmp_path):
        """Test importing valid response data."""
        design = mock_design_ff
        
        # First export the design
        export_file = tmp_path / "export_for_import.xlsx"
        design.export_experiments(
            responses=["Yield", "Purity"],
            randomize=False,
            destination=str(export_file),
            coded=False
        )
        
        # Modify the Excel file to add response data
        df = pd.read_excel(export_file)
        df["Yield"] = [45.2, 78.5, 62.3, 89.1, 55.8, 92.4, 88.0, 76.5, 95.0]
        df["Purity"] = [88.5, 91.2, 85.7, 94.5, 87.3, 95.8, 90.1, 89.4, 93.2]
        df.to_excel(export_file, index=False)
        
        # Import the responses
        design.import_responses(source=str(export_file))
        
        # Verify responses were imported correctly
        assert design._responses is not None
        assert design._responses.shape == (9, 2)
        assert "Yield" in design._responses.columns
        assert "Purity" in design._responses.columns
        assert np.isclose(design._responses["Yield"].iloc[0], 45.2)
        assert np.isclose(design._responses["Purity"].iloc[8], 93.2)
    
    def test_import_responses_wrong_size(self, mock_design_ff, tmp_path):
        """Test that importing wrong number of rows raises error."""
        design = mock_design_ff
        
        # Create a file with wrong number of rows
        wrong_file = tmp_path / "wrong_size.xlsx"
        df = pd.DataFrame({
            "Exp. Idx": [0, 1, 2],
            "Yield": [45.2, 78.5, 62.3],
            "Purity": [88.5, 91.2, 85.7]
        })
        df.to_excel(wrong_file, index=False)
        
        design._response_list = ["Yield", "Purity"]
        
        with pytest.raises(ValueError, match="Number of rows"):
            design.import_responses(source=str(wrong_file))
    
    def test_import_responses_missing_columns(self, mock_design_ff, tmp_path):
        """Test that missing response columns raises error."""
        design = mock_design_ff
        
        # Create a file missing expected response columns
        missing_file = tmp_path / "missing_cols.xlsx"
        df = pd.DataFrame({
            "Exp. Idx": list(range(9)),
            "Yield": [45.2, 78.5, 62.3, 89.1, 55.8, 92.4, 88.0, 76.5, 95.0],
            # Missing "Purity" column
        })
        df.to_excel(missing_file, index=False)
        
        design._response_list = ["Yield", "Purity"]
        
        with pytest.raises(ValueError, match="Response columns not found"):
            design.import_responses(source=str(missing_file))
    
    def test_import_responses_reordered(self, mock_design_ff, tmp_path):
        """Test that import correctly handles reordered experiments."""
        design = mock_design_ff
        
        # Export with randomization
        export_file = tmp_path / "export_random.xlsx"
        np.random.seed(42)
        design.export_experiments(
            responses=["Yield"],
            randomize=True,
            destination=str(export_file),
            coded=False
        )
        
        # Read the exported file to see the randomized order
        df = pd.read_excel(export_file)
        
        # Add response data in the randomized order shown in the file
        # These values are assigned based on the Exp. Order (randomized sequence)
        df["Yield"] = [45.2, 78.5, 62.3, 89.1, 55.8, 92.4, 88.0, 76.5, 95.0]
        
        # Map the response values to their original experiment indices
        # This creates the expected order after import reorders by Exp. Idx
        expected_order = df.sort_values("Exp. Idx")["Yield"].values
        
        df.to_excel(export_file, index=False)
        
        # Import should reorder by Exp. Idx
        design.import_responses(source=str(export_file))
        
        # Verify data is correctly ordered by exp index
        assert design._responses is not None
        assert design._responses.shape[0] == 9
        
        np.testing.assert_array_almost_equal(
            design._responses["Yield"].values,
            expected_order,
            decimal=1
        )


# =============================================================================
#                     Test: MLR Model Computation
# =============================================================================

class TestMLRModel:
    """Test compute_mlr_model method."""
    
    @pytest.fixture
    def mock_design_with_responses(self, mock_design_ff, sample_responses):
        """Create a mock design with model terms and responses."""
        design = mock_design_ff
        
        # Set model terms
        terms = ModelTerms(
            intercept=True,
            pro_main=["A", "B"],
            pro_int2=[("A", "B")],
            pro_quadratic=["A", "B"]
        )
        design.set_model_terms(terms)
        
        # Set responses
        design._response_list = ["Yield", "Purity"]
        design._responses = sample_responses
        
        return design
    
    def test_compute_mlr_model_basic(self, mock_design_with_responses):
        """Test basic MLR model computation."""
        design = mock_design_with_responses
        
        design.compute_mlr_model()
        
        # Verify that MLR wrapper was created
        assert design._mlr_wrapper is not None
        
        # Verify that models were fitted for both responses
        assert "Yield" in design._mlr_wrapper.results
        assert "Purity" in design._mlr_wrapper.results
        
        # Verify that models have coefficients
        assert design._mlr_wrapper.results["Yield"].model is not None
        assert design._mlr_wrapper.results["Purity"].model is not None
    
    def test_compute_mlr_model_no_responses(self, mock_design_ff_with_model):
        """Test that error is raised when responses not imported."""
        design = mock_design_ff_with_model
        
        with pytest.raises(ValueError, match="No responses defined"):
            design.compute_mlr_model()
    
    def test_compute_mlr_model_no_model_terms(self, mock_design_ff, sample_responses):
        """Test that error is raised when model terms not set."""
        design = mock_design_ff
        design._response_list = ["Yield", "Purity"]
        design._responses = sample_responses
        
        with pytest.raises(ValueError, match="No model matrix defined"):
            design.compute_mlr_model()
    
    def test_compute_mlr_model_with_replicates(self, simple_continuous_factors, 
                                                 simple_design_with_replicates, 
                                                 sample_responses):
        """Test MLR computation with replicated runs."""
        design = MockDesign()
        design._factors = simple_continuous_factors
        design._coded_design_matrix = simple_design_with_replicates
        design._design_matrix = design._decode_matrix(design._coded_design_matrix)
        
        # Set model terms
        terms = ModelTerms(
            intercept=True,
            pro_main=["A", "B"],
            pro_int2=[],
            pro_quadratic=[]
        )
        design.set_model_terms(terms)
        
        # Set responses (12 rows to match the replicated design)
        design._response_list = ["Yield", "Purity"]
        responses_with_repl = pd.DataFrame({
            "Yield": [45.2, 78.5, 62.3, 89.1, 55.8, 92.4, 88.0, 76.5, 95.0, 44.8, 56.2, 94.5],
            "Purity": [88.5, 91.2, 85.7, 94.5, 87.3, 95.8, 90.1, 89.4, 93.2, 88.2, 87.5, 93.5]
        })
        design._responses = responses_with_repl
        
        # Should compute successfully with replicates
        design.compute_mlr_model()
        
        assert design._mlr_wrapper is not None


# =============================================================================
#                     Test: Prediction
# =============================================================================

class TestPrediction:
    """Test predict method."""
    
    @pytest.fixture
    def mock_design_fitted(self, mock_design_ff, sample_responses):
        """Create a mock design with fitted MLR models."""
        design = mock_design_ff
        
        # Set model terms
        terms = ModelTerms(
            intercept=True,
            pro_main=["A", "B"],
            pro_int2=[("A", "B")],
            pro_quadratic=[]
        )
        design.set_model_terms(terms)
        
        # Set and fit responses
        design._response_list = ["Yield", "Purity"]
        design._responses = sample_responses
        design.compute_mlr_model()
        
        return design
    
    def test_predict_single_point(self, mock_design_fitted):
        """Test prediction at a single point."""
        design = mock_design_fitted
        
        # Create a prediction point at the center
        pred_matrix = pd.DataFrame({
            "A": [0.0],
            "B": [0.0]
        })
        
        predictions = design._predict(pred_matrix, ["Yield", "Purity"])
        
        # Verify predictions were generated
        assert predictions is not None
        assert predictions.shape == (1, 2)
        assert "Yield" in predictions.columns
        assert "Purity" in predictions.columns
        
        # Verify predictions are numeric
        assert np.isfinite(predictions["Yield"].iloc[0])
        assert np.isfinite(predictions["Purity"].iloc[0])
    
    def test_predict_multiple_points(self, mock_design_fitted):
        """Test prediction at multiple points."""
        design = mock_design_fitted
        
        # Create multiple prediction points
        pred_matrix = pd.DataFrame({
            "A": [-1.0, 0.0, 1.0],
            "B": [0.0, 0.0, 0.0]
        })
        
        predictions = design._predict(pred_matrix, ["Yield", "Purity"])
        
        assert predictions.shape == (3, 2)
        assert all(np.isfinite(predictions["Yield"]))
        assert all(np.isfinite(predictions["Purity"]))
    
    def test_predict_single_response(self, mock_design_fitted):
        """Test prediction of a single response variable."""
        design = mock_design_fitted
        
        pred_matrix = pd.DataFrame({
            "A": [0.5],
            "B": [-0.5]
        })
        
        predictions = design._predict(pred_matrix, ["Yield"])
        
        assert predictions.shape == (1, 1)
        assert "Yield" in predictions.columns
        assert "Purity" not in predictions.columns
    
    def test_predict_no_model(self, mock_design_ff):
        """Test that error is raised when MLR model not computed."""
        design = mock_design_ff
        
        pred_matrix = pd.DataFrame({
            "A": [0.0],
            "B": [0.0]
        })
        
        with pytest.raises(ValueError, match="No MLR model computed"):
            design._predict(pred_matrix, ["Yield"])


# =============================================================================
#                     Test: Leverage Computation
# =============================================================================

class TestLeverageComputation:
    """Test _compute_leverage method."""
    
    def test_compute_leverage_basic(self, mock_design_ff_with_model):
        """Test basic leverage computation."""
        design = mock_design_ff_with_model
        
        leverages = design._compute_leverage(design._coded_design_matrix)
        
        # Verify leverages were computed
        assert leverages is not None
        assert len(leverages) == design._coded_design_matrix.shape[0]
        
        # All leverages should be positive
        assert all(leverages > 0)
        
        # Sum of leverages should equal number of model parameters
        # For model with Int, A, B, A:B that's 4 terms
        assert np.isclose(np.sum(leverages), 4, atol=1e-10)
    
    def test_compute_leverage_corner_points(self, mock_design_ff_with_model):
        """Test that corner points have high leverage."""
        design = mock_design_ff_with_model
        
        # Corner points in a 2-factor design
        corners = pd.DataFrame({
            "A": [-1.0, -1.0, 1.0, 1.0],
            "B": [-1.0, 1.0, -1.0, 1.0]
        })
        
        leverages = design._compute_leverage(corners)
        
        # Corner points should have relatively high leverage
        assert all(leverages > 0)
        # In a factorial design, corner points typically have equal leverage
        assert np.std(leverages) < 0.01  # Should be very similar
    
    def test_compute_leverage_center_point(self, mock_design_ff_with_model):
        """Test leverage of center point."""
        design = mock_design_ff_with_model
        
        center = pd.DataFrame({
            "A": [0.0],
            "B": [0.0]
        })
        
        leverages = design._compute_leverage(center)

        assert leverages[0] > 0
        assert leverages[0] < 1.0
    
    def test_compute_leverage_no_model(self, mock_design_ff):
        """Test that error is raised when model terms not set."""
        design = mock_design_ff
        
        with pytest.raises(ValueError, match="No model matrix defined"):
            design._compute_leverage(design._coded_design_matrix)


# =============================================================================
#                     Test: Check Constant Levels
# =============================================================================

class TestCheckConstantLevels:
    """Test _check_constant_levels method (visualization/optimization helper)."""
    
    def test_check_constant_levels_defaults_continuous(self, mock_design_ff):
        """Test default constant level generation for continuous factors."""
        design = mock_design_ff
        
        # No constant levels provided, should generate defaults
        result = design._check_constant_levels(
            design._coded_design_matrix,
            x="A",
            y="B",
            z=None,
            constant_levels=None
        )
        
        # Since all factors are free variables, result should be empty
        assert result == {}
    
    def test_check_constant_levels_defaults_with_extra_factor(self, mock_design_ff):
        """Test default generation when there's a third factor."""
        design = mock_design_ff
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix["C"] = [-1.0] * 9
        
        # A and B are free, C should get default
        result = design._check_constant_levels(
            design._coded_design_matrix,
            x="A",
            y="B",
            z=None,
            constant_levels=None
        )
        
        assert "C" in result
        assert np.isclose(result["C"], 0.0)  # Center point for continuous
    
    def test_check_constant_levels_provided_continuous(self, mock_design_ff):
        """Test with user-provided constant levels."""
        design = mock_design_ff
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix["C"] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        
        result = design._check_constant_levels(
            design._coded_design_matrix,
            x="A",
            y="B",
            z=None,
            constant_levels={"C": 0.5}
        )
        
        assert result["C"] == 0.5
    
    def test_check_constant_levels_categorical_default(self, mixed_factors, mock_design_ff):
        """Test default for categorical factors is baseline (-1)."""
        design = mock_design_ff
        design._factors = mixed_factors
        design._coded_design_matrix = pd.DataFrame({
            "Temp": [-1.0, 0.0, 1.0],
            "Pressure": [-1.0, 0.0, 1.0],
            "Catalyst": [-1.0, 0.0, 1.0]
        })
        
        result = design._check_constant_levels(
            design._coded_design_matrix,
            x="Temp",
            y="Pressure",
            z=None,
            constant_levels=None
        )
        
        assert "Catalyst" in result
        assert np.isclose(result["Catalyst"], -1.0)  # Baseline for categorical

    def test_check_constant_levels_uses_categorical_reference(self, mock_design_ff):
        """Automatic fixed levels use the configured categorical reference."""
        design = mock_design_ff
        design._factors["Catalyst"] = CategoricalFactor(
            levels=["A", "B", "C"], reference_level="B"
        )
        design._coded_design_matrix["Catalyst"] = [-1.0, 0.0, 1.0] * 3

        result = design._check_constant_levels(
            design._coded_design_matrix,
            x="A",
            y="B",
            z=None,
            constant_levels=None,
        )

        assert result["Catalyst"] == 0.0
    
    def test_check_constant_levels_mixture_default(self, unconstrained_mixture_factors):
        """Test default for mixture factors is simplex centroid."""
        design = MockDesign()
        design._factors = unconstrained_mixture_factors
        design._factors["X4"] = MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix = pd.DataFrame({
            "X1": [1.0, 0.0, 0.0, 0.0],
            "X2": [0.0, 1.0, 0.0, 0.0],
            "X3": [0.0, 0.0, 1.0, 0.0],
            "X4": [0.0, 0.0, 0.0, 1.0]
        })
        
        # Only X1 and X2 are free, X3 should get default
        result = design._check_constant_levels(
            design._coded_design_matrix,
            x="X1",
            y="X2",
            z="X3",
            constant_levels=None
        )
        
        assert "X4" in result
        assert np.isclose(result["X4"], 1/4)  # Centroid for unconstrained mixture
    
    def test_check_constant_levels_simplex_all_free(self, unconstrained_mixture_factors):
        """Test simplex plot with all three variables free."""
        design = MockDesign()
        design._factors = unconstrained_mixture_factors
        design._coded_design_matrix = pd.DataFrame({
            "X1": [1.0, 0.0, 0.0],
            "X2": [0.0, 1.0, 0.0],
            "X3": [0.0, 0.0, 1.0]
        })
        
        # All three are free for simplex plot
        result = design._check_constant_levels(
            design._coded_design_matrix,
            x="X1",
            y="X2",
            z="X3",
            constant_levels=None
        )
        
        # No other factors, so result should be empty
        assert result == {}
    
    def test_check_constant_levels_error_same_variables(self, mock_design_ff):
        """Test error when x and y are the same."""
        design = mock_design_ff
        
        with pytest.raises(ValueError, match="x, y, and z must be different"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="A",
                y="A",  # Same as x
                z=None,
                constant_levels=None
            )
    
    def test_check_constant_levels_error_variable_not_found(self, mock_design_ff):
        """Test error when variable not in factors."""
        design = mock_design_ff
        
        with pytest.raises(ValueError, match="must be in"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="A",
                y="NonExistent",
                z=None,
                constant_levels=None
            )
    
    def test_check_constant_levels_error_wrong_type_2d(self, unconstrained_mixture_factors):
        """Test error when 2D plot uses mixture factors."""
        design = MockDesign()
        design._factors = unconstrained_mixture_factors
        design._coded_design_matrix = pd.DataFrame({
            "X1": [1.0, 0.0],
            "X2": [0.0, 1.0],
            "X3": [0.0, 0.0]
        })
        
        with pytest.raises(ValueError, match="x and y must be continuous or categorical"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="X1",  # Mixture factor
                y="X2",  # Mixture factor
                z=None,
                constant_levels=None
            )
    
    def test_check_constant_levels_error_wrong_type_simplex(self, simple_continuous_factors):
        """Test error when simplex plot uses non-mixture factors."""
        design = MockDesign()
        design._factors = simple_continuous_factors
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix = pd.DataFrame({
            "A": [-1.0, 1.0, 1.0, -1.0],
            "B": [-1.0, 1.0, 1.0, -1.0],
            "C": [-1.0, 1.0, -1.0, 1.0]
        })
        
        with pytest.raises(ValueError, match="x, y, and z must be mixture"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="A",
                y="B",
                z="C",  # Not a mixture factor
                constant_levels=None
            )
    
    def test_check_constant_levels_error_missing_keys(self, mock_design_ff):
        """Test error when constant_levels is missing required keys."""
        design = mock_design_ff
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix["C"] = [0.0] * 9
        
        with pytest.raises(ValueError, match="missing keys"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="A",
                y="B",
                z=None,
                constant_levels={}  # Missing "C"
            )
    
    def test_check_constant_levels_error_extra_keys(self, mock_design_ff):
        """Test error when constant_levels has unexpected keys."""
        design = mock_design_ff
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix["C"] = [0.0] * 9
        
        with pytest.raises(ValueError, match="must be in"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="A",
                y="B",
                z=None,
                constant_levels={"C": 0.0, "D": 1.0}  # "D" doesn't exist
            )
    
    def test_check_constant_levels_error_continuous_out_of_range(self, mock_design_ff):
        """Test error when continuous constant level is out of range."""
        design = mock_design_ff
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix["C"] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        
        with pytest.raises(ValueError, match="not in"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="A",
                y="B",
                z=None,
                constant_levels={"C": 5.0}  # Out of range [-1, 1]
            )
    
    def test_check_constant_levels_error_continuous_not_numeric(self, mock_design_ff):
        """Test error when continuous level is not numeric."""
        design = mock_design_ff
        design._factors["C"] = ContinuousFactor(n_levels=3, lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix["C"] = [0.0] * 9
        
        with pytest.raises(TypeError, match="must be a number"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="A",
                y="B",
                z=None,
                constant_levels={"C": "invalid"}
            )
    
    def test_check_constant_levels_error_categorical_invalid_level(self, mixed_factors):
        """Test error when categorical level is invalid."""
        design = MockDesign()
        design._factors = mixed_factors
        design._coded_design_matrix = pd.DataFrame({
            "Temp": [-1.0, 0.0, 1.0],
            "Pressure": [-1.0, 0.0, 1.0],
            "Catalyst": [-1.0, 0.0, 1.0]
        })
        
        with pytest.raises(ValueError, match="not in"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="Temp",
                y="Pressure",
                z=None,
                constant_levels={"Catalyst": 5.0}  # Invalid coded level
            )
    
    def test_check_constant_levels_error_mixture_out_of_range(self, unconstrained_mixture_factors):
        """Test error when mixture level is out of range."""
        design = MockDesign()
        design._factors = unconstrained_mixture_factors
        design._factors["X4"] = MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        design._coded_design_matrix = pd.DataFrame({
            "X1": [1.0, 0.0, 0.0, 0.5],
            "X2": [0.0, 1.0, 0.0, 0.5],
            "X3": [0.0, 0.0, 1.0, 0.0],
            "X4": [0.0, 0.0, 0.0, 0.0]
        })
        
        with pytest.raises(ValueError, match="not in"):
            design._check_constant_levels(
                design._coded_design_matrix,
                x="X1",
                y="X2",
                z="X3",
                constant_levels={"X4": 2.0}  # Out of range [0, 1]
            )


# =============================================================================
#                     Test: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_build_model_matrix_quadratic_two_levels_error(self, frf_2levels_3factors):
        """Test error when quadratic requested for 2-level factor."""
        design = MockDesign()
        design._factors = frf_2levels_3factors
        design._coded_design_matrix = pd.DataFrame({
            "A": [-1.0, 1.0],
            "B": [-1.0, 1.0],
            "C": [-1.0, 1.0]
        })
        
        from doetools.utils.model_spec import ModelSpec
        model_spec = ModelSpec(
            intercept=True,
            main=["A", "B", "C"],
            interaction2=[],
            quadratic=["A"],  # A only has 2 levels
            interaction3=[]
        )
        
        with pytest.raises(KeyError, match="has only 2 levels"):
            design._build_model_matrix(design._coded_design_matrix, model_spec)
    
    def test_add_center_points_mixture_bounds_exceed_one(self):
        """Test error when mixture lower bounds exceed 1."""
        design = MockDesign()
        design._factors = {
            "X1": MixtureFactor(lower_bound=0.6, upper_bound=1.0),
            "X2": MixtureFactor(lower_bound=0.5, upper_bound=1.0),
            "X3": MixtureFactor(lower_bound=0.1, upper_bound=1.0)
        }
        
        matrix = pd.DataFrame({
            "X1": [0.7],
            "X2": [0.2],
            "X3": [0.1]
        })
        
        with pytest.raises(ValueError, match="Sum of lower_bounds must be"):
            design._add_center_points(matrix, center_points=1)
    
    def test_number_of_center_points_mixture_bounds_exceed_one(self):
        """Test error in count when mixture lower bounds exceed 1."""
        design = MockDesign()
        design._factors = {
            "X1": MixtureFactor(lower_bound=0.6, upper_bound=1.0),
            "X2": MixtureFactor(lower_bound=0.5, upper_bound=1.0),
            "X3": MixtureFactor(lower_bound=0.1, upper_bound=1.0)
        }
        
        matrix = pd.DataFrame({
            "X1": [0.7],
            "X2": [0.2],
            "X3": [0.1]
        })
        
        with pytest.raises(ValueError, match="Sum of lower_bounds must be"):
            design._number_of_center_points(matrix)
    
    def test_add_replicates_zero(self, mock_design_ff):
        """Test that adding zero replicates doesn't change the design."""
        design = mock_design_ff
        initial_size = design._coded_design_matrix.shape[0]
        
        design.add_replicates(type="all", n_replicates=0)
        
        assert design._coded_design_matrix.shape[0] == initial_size
    
    def test_add_replicates_manual_no_indices(self, mock_design_ff):
        """Test error when manual replicates requested without indices."""
        design = mock_design_ff
        
        with pytest.raises(ValueError, match="provide a list of indices"):
            design.add_replicates(type="manual", n_replicates=2, indices=None)
    
    def test_set_response_conditions_all_fields(self, mock_design_ff):
        """Test setting response conditions with all fields populated."""
        design = mock_design_ff
        design._response_list = ["Yield", "Purity", "Cost"]
        
        design.set_response_conditions(
            lower_limits=[40.0, 80.0, False],
            upper_limits=[100.0, 99.0, 50.0],
            maximize=[True, True, False]
        )
        
        # Verify all responses have all conditions set
        for resp in ["Yield", "Purity", "Cost"]:
            assert "lower_limit" in design._response_conditions[resp]
            assert "upper_limit" in design._response_conditions[resp]
            assert "maximize" in design._response_conditions[resp]
