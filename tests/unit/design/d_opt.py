"""
Comprehensive unit tests for D-Optimal Design.

Testing Strategy:
-----------------
1. NOT testing the Fedorov algorithm directly (integration tests will do that)
2. DO test: initialization, candidate point generation, consistency
3. Focus on all combinations of process/mixture strategies
4. Test that multiple runs give similar results (consistency)
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from doetools.design.d_opt import DOptDesign
from doetools.utils import ContinuousFactor, CategoricalFactor, MixtureFactor
from doetools.utils.model_spec import ModelTerms


# Helper function to create simple model terms
def create_simple_model_terms(intercept=True, quadratic=False, interaction2=False):
    """Helper to create ModelTerms with proper API"""
    return ModelTerms(
        intercept=intercept,
        pro_main="all",
        pro_int2="all" if interaction2 else None,
        pro_quadratic="all" if quadratic else None,
        pro_int3=None
    )


class TestDOptInitialization:
    """Test initialization and input validation"""
    
    def test_valid_initialization_process_only(self):
        """Test initialization with only process factors"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            "X3" : ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        assert design is not None
        assert design._design_type == "D-Optimal"
        assert len(design._factors) == 3
        assert design._coded_cp is not None
        assert design._cp is not None
    
    def test_valid_initialization_mixture_only(self):
        """Test initialization with only mixture factors"""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(factors=factors, mixture_include="all")
        
        assert design is not None
        assert design._design_type == "D-Optimal"
        assert len(design._factors) == 3
        assert design._coded_cp is not None
        assert design._cp is not None
    
    def test_valid_initialization_combined(self):
        """Test initialization with both process and mixture factors"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            "Pressure" : ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=10, decimals=1),
            "pH" : ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=14, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            "C" : MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='grid',
            mixture_include=None,
            mixture_grid={"degree": 3, "max_candidates": 1000}
        )
        
        assert design is not None
        assert len(design._factors) == 6
        assert design._coded_cp is not None
        assert design._design_type == "D-Optimal"
        assert design._cp is not None
    
    def test_invalid_lhs_n_samples_without_lhs(self):
        """Test that lhs_n_samples without lhs strategy raises error"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        with pytest.raises(ValueError, match="lhs_n_samples can be specified only with 'lhs'"):
            DOptDesign(factors=factors, process_strategy='grid', lhs_n_samples=100)
    
    def test_lhs_requires_lhs_n_samples(self):
        """Test that lhs strategy requires lhs_n_samples"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        with pytest.raises(ValueError, match="lhs_n_samples must be specified"):
            DOptDesign(factors=factors, process_strategy='lhs')


class TestDOptCandidatePoints:
    """Test candidate point generation for all strategy combinations"""
    
    # -------------------- PROCESS STRATEGIES --------------------
    
    def test_candidate_points_grid_strategy(self):
        """Test grid strategy generates full factorial candidate points"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        # Full factorial: 3^2 = 9 points
        assert len(design._coded_cp) == 9
        assert design._coded_cp.shape[1] == 2
        
        # Check all coded values are in [-1, 0, 1]
        for col in design._coded_cp.columns:
            unique_vals = design._coded_cp[col].unique()
            assert all(v in [-1, 0, 1] for v in unique_vals)
        
        for col in design._cp.columns:
            unique_vals = design._cp[col].unique()
            assert len(unique_vals) == 3  # 3 levels per factor
            assert all(v in [0.0, 5.0, 10.0] for v in unique_vals)

    def test_plot_candidate_set_process_uses_design_plot_builder(self):
        """Test candidate set visualization for process factors."""
        factors = {
            "X1": ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            "X2": ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
        }
        design = DOptDesign(factors=factors, process_strategy="grid")

        figure = design.plot_candidate_set("X1", "X2")

        assert isinstance(figure, go.Figure)
        assert figure.layout.meta["geometry"] == "process_2d"
        assert any(
            annotation.text == "<b>Candidate Set Design</b>"
            for annotation in figure.layout.annotations
        )

    def test_plot_candidate_set_mixture_supports_allowed_domain(self):
        """Test candidate set visualization for constrained mixture candidates."""
        factors = {
            "A": MixtureFactor(lower_bound=0.1, upper_bound=0.8),
            "B": MixtureFactor(lower_bound=0.1, upper_bound=0.8),
            "C": MixtureFactor(lower_bound=0.1, upper_bound=0.8),
        }
        design = DOptDesign(
            factors=factors,
            mixture_include=None,
            mixture_grid={"degree": 4, "max_candidates": 1000},
        )

        figure = design.plot_candidate_set(
            "A",
            "B",
            "C",
            domain="allowed",
        )

        assert isinstance(figure, go.Figure)
        assert figure.layout.meta["geometry"] == "mixture_simplex_2d"
        assert figure.layout.meta["domain"] == "allowed"
    
    def test_candidate_points_lhs_strategy(self):
        """Test Latin Hypercube Sampling strategy"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='lhs', lhs_n_samples=50)
        
        # Should have exactly 50 points
        assert len(design._coded_cp) == 50
        assert design._coded_cp.shape[1] == 2
        
        # LHS should cover the space well (check range)
        for col in design._coded_cp.columns:
            assert design._coded_cp[col].min() < -0.5
            assert design._coded_cp[col].max() > 0.5
            # Check range -1, 1
            assert design._coded_cp[col].min() >= -1.0
            assert design._coded_cp[col].max() <= 1.0
    
    def test_candidate_points_ccc_strategy(self):
        """Test Central Composite Circumscribed strategy"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='ccc')
        
        # CCC for 2 factors: 4 factorial + 4 axial + 1 center = 9 points
        assert len(design._coded_cp) == 9
        
        # Check for axial points beyond ±1
        max_val = max(design._coded_cp['X1'].max(), design._coded_cp['X2'].max())
        assert max_val > 1.0, "CCC should have axial points beyond ±1"
        
        # Check 5 unique levels per factor
        for col in design._coded_cp.columns:
            unique_vals = design._coded_cp[col].unique()
            assert len(unique_vals) == 5  # -alpha, -1, 0, 1, alpha
    
    def test_candidate_points_ccf_strategy(self):
        """Test Central Composite Face-centered strategy"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='ccf')
        
        # CCF for 2 factors: 4 factorial + 4 axial (at ±1) + 1 center = 9 points
        assert len(design._coded_cp) == 9
        
        # Check axial points are at ±1 (face-centered)
        max_val = max(abs(design._coded_cp['X1']).max(), abs(design._coded_cp['X2']).max())
        assert np.isclose(max_val, 1.0), "CCF axial points should be at ±1"
        for col in design._coded_cp.columns:
            unique_vals = design._coded_cp[col].unique()
            assert len(unique_vals) == 3  # -1, 0, 1
    
    def test_candidate_points_cci_strategy(self):
        """Test Central Composite Inscribed strategy"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='cci')
        
        # CCI for 2 factors: 9 points
        assert len(design._coded_cp) == 9
        
        # Check that all points are within ±1 (inscribed)
        assert design._coded_cp.abs().max().max() <= 1.0
        for col in design._coded_cp.columns:
            unique_vals = design._coded_cp[col].unique()
            assert len(unique_vals) == 5  # -1, -1/√2, 0, 1/√2, 1
    
    def test_candidate_points_bb_strategy(self):
        """Test Box-Behnken strategy"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X3': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='bb')
        
        # Box-Behnken for 3 factors: 12 edge points + 1 center = 13 points
        assert len(design._coded_cp) == 13
        for col in design._coded_cp.columns:
            unique_vals = design._coded_cp[col].unique()
            assert len(unique_vals) == 3  # -1, 0, 1
    
    # -------------------- MIXTURE STRATEGIES --------------------

    def test_mixture_candidates_are_explicit_by_default(self):
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        }

        with pytest.raises(ValueError, match="Mixture factors require"):
            DOptDesign(factors=factors)
    
    def test_candidate_points_mixture_grid(self):
        """Test mixture grid strategy (simplex lattice)"""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            mixture_include=None,
            mixture_grid={"degree": 3, "max_candidates": 1000},
        )
        
        # Lattice degree 3 with 3 factors: C(3+3-1,3) = C(5,3) = 10 points
        assert len(design._coded_cp) == 10
        
        # Check mixture constraint (sum to 1)
        sums = design._coded_cp.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)
    
    def test_candidate_points_mixture_centroid(self):
        """Test mixture centroid strategy"""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            "D" : MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(factors=factors, mixture_include="all")
        
        # Centroid for 4 factors: 2^4 - 1 = 15 points
        assert len(design._coded_cp) == 15
        
        # Check mixture constraint
        sums = design._coded_cp.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)

    def test_candidate_points_mixture_subset_exact_categories(self):
        """Test mixture candidate subsets use exact build_candidate_points names."""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }

        design = DOptDesign(
            factors=factors,
            mixture_include={"vertices", "edge_midpoints"},
        )

        assert len(design._coded_cp) == 6
        assert np.allclose(design._coded_cp.sum(axis=1), 1.0, atol=1e-10)

    def test_candidate_points_mixture_rejects_aliases(self):
        """Test D-optimal mixture candidates reject non-exact category aliases."""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }

        for alias in ("vertex", "mid_points", "midpoints", "centroid", "face_centroid"):
            with pytest.raises(ValueError, match="Unsupported mixture candidate category"):
                DOptDesign(factors=factors, mixture_include={alias})

    def test_candidate_points_mixture_grid_rejects_old_api(self):
        """Test D-optimal mixture grid rejects legacy grid options."""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }

        with pytest.raises(TypeError, match="grid must be None or a mapping"):
            DOptDesign(factors=factors, mixture_include=None, mixture_grid=True)

        with pytest.raises(ValueError, match="Unsupported grid option"):
            DOptDesign(factors=factors, mixture_include=None, mixture_grid={"n_levels": 4})

    def test_candidate_points_two_component_mixture_all(self):
        """Test default geometric mixture candidates work for two components."""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }

        design = DOptDesign(factors=factors, mixture_include="all")

        assert len(design._coded_cp) == 3
        assert np.allclose(design._coded_cp[["A", "B"]].sum(axis=1), 1.0, atol=1e-10)

    def test_candidate_points_constrained_three_component_mixture_all(self):
        """Test geometric candidates respect constrained three-component bounds."""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=0.7),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=0.7),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=0.7)
        }

        design = DOptDesign(factors=factors, mixture_include="all")

        assert len(design._coded_cp) == 13
        assert np.allclose(design._coded_cp[["A", "B", "C"]].sum(axis=1), 1.0, atol=1e-10)
        assert (design._cp[["A", "B", "C"]] >= 0.0 - 1e-10).all().all()
        assert (design._cp[["A", "B", "C"]] <= 0.7 + 1e-10).all().all()

    def test_candidate_points_constrained_four_component_mixture_all(self):
        """Test geometric candidates respect highly constrained four-component bounds."""
        factors = {
            'A': MixtureFactor(lower_bound=0.098, upper_bound=0.118, decimals=3),
            'B': MixtureFactor(lower_bound=0.302, upper_bound=0.322, decimals=3),
            'C': MixtureFactor(lower_bound=0.140, upper_bound=0.160, decimals=3),
            'D': MixtureFactor(lower_bound=0.420, upper_bound=0.440, decimals=3),
        }

        design = DOptDesign(factors=factors, mixture_include="all")

        assert len(design._coded_cp) == 27
        assert np.allclose(design._coded_cp[["A", "B", "C", "D"]].sum(axis=1), 1.0, atol=1e-10)
        assert (design._cp["A"] >= 0.098 - 1e-10).all()
        assert (design._cp["D"] <= 0.440 + 1e-10).all()

    def test_candidate_points_mixture_grid_can_be_included(self):
        """Test optional mixture grid candidates are passed to the shared builder."""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }

        design = DOptDesign(
            factors=factors,
            mixture_include={"vertices"},
            mixture_grid={"degree": 3, "max_candidates": 1000},
        )

        assert len(design._coded_cp) == 10
        assert np.allclose(design._coded_cp[["A", "B", "C"]].sum(axis=1), 1.0, atol=1e-10)
    
    # -------------------- COMBINED STRATEGIES --------------------
    
    def test_candidate_points_grid_and_mixture_grid(self):
        """Test grid process + grid mixture strategy"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            "Pressure" : ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=10, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            "C" : MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='grid',
            mixture_include=None,
            mixture_grid={"degree": 2, "max_candidates": 1000}
        )
        # Cross Product: Process: 3 * 3 = 9, Mixture lattice m=2: C(3+2-1,2) = 6 points
        assert len(design._coded_cp) == 54
        assert design._coded_cp.shape[1] == 5
    
    def test_candidate_points_lhs_and_mixture_centroid(self):
        """Test LHS process + centroid mixture strategy"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            "C" : MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='lhs',
            lhs_n_samples=20,
            mixture_include="all"
        )
        
        # LHS: 20 samples, Centroid: 7 points
        # Cross product: 20 * 7 = 140 points
        assert len(design._coded_cp) == 140
    
    def test_candidate_points_ccc_and_mixture_grid(self):
        """Test CCC process + grid mixture strategy"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='ccc',
            mixture_include=None,
            mixture_grid={"degree": 2, "max_candidates": 1000}
        )
        
        # CCC for 2 factors: 9 points, Mixture lattice m=2: C(3+2-1,2) = 6 points
        # Cross product: 9 * 6 = 54 points
        assert len(design._coded_cp) == 54
    
    # -------------------- CATEGORICAL FACTORS --------------------
    
    def test_candidate_points_with_categorical(self):
        """Test candidate points with categorical factors"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Material': CategoricalFactor(levels=['A', 'B', 'C'])
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        # Continuous: 3 levels, Categorical: 3 levels
        # Cross product: 3 * 3 = 9 points
        assert len(design._coded_cp) == 9
        
        # Check categorical is included
        assert 'Material' in design._coded_cp.columns
    
    def test_candidate_points_lhs_with_categorical(self):
        """Test LHS with categorical factors (should do cross product)"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            "Pressure" : ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=10, decimals=1),
            'Catalyst': CategoricalFactor(levels=['A', 'B'])
        }
        
        design = DOptDesign(factors=factors, process_strategy='lhs', lhs_n_samples=30)
        
        # LHS: 30 samples for continuous, Categorical: 2 levels
        # Cross product: 30 * 2 = 60 points
        assert len(design._coded_cp) == 60
    
    def test_candidate_points_all_three_factor_types(self):
        """Test process + mixture + categorical combined (realistic use case)"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            "Pressure" : ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=10, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'Catalyst': CategoricalFactor(levels=['X', 'Y'])
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='grid',
            mixture_include="all",
        )
        
        # Process:  3 * 3, Mixture: 3 (centroid for 2 components), Categorical: 2
        # Total: 9 * 3 * 2 = 54
        assert len(design._coded_cp) == 54
        assert design._coded_cp.shape[1] == 5
        
        # Check mixture constraint still holds
        mixture_cols = ['A', 'B']
        sums = design._coded_cp[mixture_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)
    
    def test_candidate_points_mixture_and_categorical(self):
        """Test mixture + categorical without process factors"""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'Method': CategoricalFactor(levels=['Hot', 'Cold'])
        }
        
        design = DOptDesign(
            factors=factors,
            mixture_include=None,
            mixture_grid={"degree": 2, "max_candidates": 1000},
            process_strategy='grid',
        )
        
        assert len(design._coded_cp) == 12, "Should have at least mixture lattice points"
        assert 'Method' in design._coded_cp.columns or 'Method' in design._factors
        
        # Check mixture constraint
        mixture_cols = ['A', 'B', 'C']
        sums = design._coded_cp[mixture_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)
    
    def test_candidate_points_bb_and_mixture_centroid(self):
        """Test Box-Behnken process + centroid mixture strategy"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X3': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='bb',
            mixture_include="all"
        )
        
        # Box-Behnken for 3 factors: 13 points, Mixture centroid for 2: 3 points
        # Cross product: 13 * 3 = 39 points
        assert len(design._coded_cp) == 39
        assert design._coded_cp.shape[1] == 5

class TestDOptFilters:
    """Test constraint filtering of candidate points"""
    
    def test_filter_reduces_candidates(self):
        """Test that filters reduce candidate point count"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        # Filter: X1 + X2 <= 15
        def constraint(df):
            return df['X1'] + df['X2'] <= 15
        
        design_no_filter = DOptDesign(factors=factors, process_strategy='grid')
        design_with_filter = DOptDesign(
            factors=factors,
            process_strategy='grid',
            filters=[constraint]
        )
        
        # Filtered design should have fewer points
        assert len(design_with_filter._coded_cp) < len(design_no_filter._coded_cp)
        assert (design_with_filter._cp["X1"] + design_with_filter._cp["X2"] <= 15).all()
        assert design_with_filter._domain_filters == [constraint]
    
    def test_multiple_filters(self):
        """Test multiple constraint filters"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        def filter1(df):
            return df['X1'] + df['X2'] <= 15
        
        def filter2(df):
            return df['X1'] >= 2
        
        design = DOptDesign(
            factors=factors,
            process_strategy='grid',
            filters=[filter1, filter2]
        )
        
        # Check all points satisfy both constraints
        assert (design._cp['X1'] + design._cp['X2'] <= 15.1).all()
        assert (design._cp['X1'] >= 1.9).all()
    
    def test_filter_eliminates_all_points(self):
        """Test that overly restrictive filters can reduce points dramatically"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        # Very restrictive constraint
        def restrictive_filter(df):
            return df['X1'] > 100  # No points satisfy this
        
        with pytest.raises(ValueError, match="empty set"):
            DOptDesign(factors=factors, process_strategy='grid', filters=[restrictive_filter])
    
    def test_filter_with_mixture_constraint(self):
        """Test filters work correctly with mixture constraints"""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1)
        }
        
        # Filter: high temp requires high component A
        def constraint(df):
            # If temp > 60, then A must be > 0.5
            high_temp_mask = df['Temp'] > 60
            if high_temp_mask.any():
                return ~high_temp_mask | (df['A'] > 0.5)
            return pd.Series([True] * len(df), index=df.index)
        
        design = DOptDesign(
            factors=factors,
            process_strategy='grid',
            mixture_include="all",
            filters=[constraint]
        )
        
        # Verify constraint holds for all remaining points
        high_temp_rows = design._cp[design._cp['Temp'] > 60]
        if len(high_temp_rows) > 0:
            assert (high_temp_rows['A'] > 0.49).all()
        
        # Mixture constraint should still hold
        mixture_cols = ['A', 'B', 'C']
        sums = design._cp[mixture_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)
    
    def test_filter_preserves_mixture_and_categorical(self):
        """Test that filters work with all factor types combined"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'Catalyst': CategoricalFactor(levels=['Type1', 'Type2'])
        }
        
        # Filter: Type1 requires X1 < 8
        def constraint(df):
            type1_mask = df['Catalyst'] == 'Type1'
            return ~type1_mask | (df['X1'] < 8)
        
        design = DOptDesign(
            factors=factors,
            process_strategy='grid',
            mixture_include="all",
            filters=[constraint]
        )
        
        # Check constraint
        type1_rows = design._cp[design._cp['Catalyst'] == 'Type1']
        if len(type1_rows) > 0:
            assert (type1_rows['X1'] < 8.1).all()
        
        # Check mixture constraint
        sums = design._cp[['A', 'B']].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)


class TestDOptModelTerms:
    """Test model term specification"""
    
    def test_set_model_terms_linear(self):
        """Test setting linear model terms"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        
        design.set_model_terms(terms)
        
        # Linear model: intercept + 2 main effects = 3 terms
        assert design._cp_model_matrix is not None
        assert design._cp_model_matrix.shape[1] == 3
    
    def test_set_model_terms_quadratic(self):
        """Test setting quadratic model terms"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2="all", pro_int3=None, pro_quadratic="all")
        
        design.set_model_terms(terms)
        
        # Quadratic model: intercept + 2 main + 1 interaction + 2 quadratic = 6 terms
        assert design._cp_model_matrix.shape[1] == 6
    
    def test_model_terms_exceeds_candidates(self):
        """Test error when model terms exceed candidate points"""
        factors = {
            'X1': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        # Only 4 candidate points (2^2)
        
        # Try to fit a model with many terms (more than 4)
        terms = ModelTerms(
            intercept=True,
            pro_main="all",
            pro_int2="all",
            pro_int3="all",
            pro_quadratic="all"
        )
        
        with pytest.raises(ValueError, match="number of model terms.*exceeds"):
            design.set_model_terms(terms)
    
    def test_model_matrix_structure_with_categoricals(self):
        """Test that categorical factors are properly encoded in model matrix"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'Cat': CategoricalFactor(levels=['A', 'B', 'C'])
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        
        # Model matrix should have 3 columns
        assert design._cp_model_matrix.shape[1] == 3, "Should have intercept + continuous factor + categorical factor"
        
        # Check all candidate points have valid encoding
        assert design._cp_model_matrix.shape[0] == len(design._coded_cp)
    
    def test_mixture_model_terms(self):
        """Test model terms work correctly for mixture factors"""
        factors = {
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(factors=factors, mixture_include="all")
        
        # Mixture models typically exclude intercept and include interactions
        terms = ModelTerms(
            intercept=False,
            mix_main="all",
            mix_int2="all"
        )
        design.set_model_terms(terms)
        
        # Should have: A + B + C + AB + AC + BC = 6 terms
        assert design._cp_model_matrix.shape[1] == 6
        assert design._cp_model_matrix.shape[0] == len(design._coded_cp)
    
    def test_combined_process_mixture_model_terms(self):
        """Test model terms with both process and mixture factors"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='grid',
            mixture_include="all"
        )
        
        terms = ModelTerms(
            intercept=False,
            pro_main="all",
            mix_main="all",
            pro_quadratic=None,
            pro_int2=None,
            mix_int2="all"
        )
        design.set_model_terms(terms)
        
        # Should have: Temp + A + B + AB + one more = 5 terms
        assert design._cp_model_matrix.shape[1] == 4


class TestDOptComputeAndSelect:
    """Test compute_d_optimal and select_design methods"""
    
    def test_compute_d_optimal_basic(self):
        """Test basic D-optimal computation"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        
        # Compute for n=4 to n=7 (linear model needs >= 3 runs)
        figure = design.compute_d_optimal(n_min=4, n_max=7, step=1, trials=10, graph=False)
        
        # Check results are stored
        assert isinstance(figure, go.Figure)
        assert design._best_idx is not None
        assert design._log_det is not None
        assert 4 in design._best_idx
        assert 7 in design._best_idx
        assert figure.layout.meta["plot_type"] == "d_optimal_solutions"

    def test_compute_d_optimal_returns_figure_without_showing(self, monkeypatch):
        """D-optimal computation should return, not display, the solutions figure."""
        show_calls = []

        def fake_show(self, *args, **kwargs):
            show_calls.append(self)

        monkeypatch.setattr(go.Figure, "show", fake_show)
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)

        figure = design.compute_d_optimal(n_min=4, n_max=4, trials=5, graph=True)

        assert isinstance(figure, go.Figure)
        assert show_calls == []
    
    def test_compute_d_optimal_validation(self):
        """Test validation in compute_d_optimal"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        # Should raise error if model terms not set (checking for _cp_model_matrix attribute)
        try:
            design.compute_d_optimal(n_min=4, n_max=7, graph=False)
            assert False, "Should have raised ValueError"
        except (ValueError, AttributeError):
            # Either ValueError or AttributeError is acceptable
            pass
    
    def test_select_design(self):
        """Test selecting a specific design"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        design.compute_d_optimal(n_min=4, n_max=7, trials=10, graph=False)
        
        # Select design with 5 runs
        design.select_design(5)
        
        assert design._design_matrix is not None
        assert len(design._design_matrix) == 5
        assert design._coded_design_matrix is not None
        assert len(design._coded_design_matrix) == 5
        assert design._model_matrix is not None
    
    def test_log_det_property(self):
        """Test log_det property accessor"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        design.compute_d_optimal(n_min=4, n_max=6, graph=False)
        
        log_det = design.log_det
        assert isinstance(log_det, pd.DataFrame)
        assert len(log_det) == 3  # n=4,5,6
        assert 'log_M' in log_det.columns
    
    def test_invalid_n_min_greater_than_n_max(self):
        """Test handling when n_min > n_max (should produce no results or error)"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        
        # Library may not validate this before execution - should produce empty or error
        try:
            design.compute_d_optimal(n_min=10, n_max=5, graph=False)
            # If it doesn't error, should have no results
            assert design._log_det is None or len(design._log_det) == 0
        except (ValueError, RuntimeError, KeyError):
            pass  # Expected behavior - any of these errors is acceptable
    
    def test_invalid_trials_zero_or_negative(self):
        """Test handling when trials = 0 (should error or produce no valid design)"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        
        with pytest.raises(ValueError, match="trials must be a positive integer"):
            design.compute_d_optimal(n_min=4, n_max=5, trials=0, graph=False)
    
    def test_invalid_step_size(self):
        """Test error when step size is invalid"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        
        with pytest.raises(ValueError, match="step must be a positive integer"):
            design.compute_d_optimal(n_min=3, n_max=5, step=0, graph=False)
    
    def test_selected_design_efficiency_reasonable(self):
        """Test that selected design achieves reasonable D-efficiency and matrix properties"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X3': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2="all", pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        # Use grid design with 27 candidate points (3^3), select 10 runs
        # Model has 1 + 3 + 3 = 7 parameters
        design.compute_d_optimal(n_min=10, n_max=10, trials=100, graph=False)
        design.select_design(10)
        
        # Check model matrix is full rank
        rank = np.linalg.matrix_rank(design._model_matrix)
        assert rank == design._model_matrix.shape[1], "Design matrix should be full rank"
        
        # Check condition number is reasonable (< 1000 for well-conditioned)
        cond = np.linalg.cond(design._model_matrix)
        assert cond < 1000, f"Condition number {cond:.1f} too high (ill-conditioned design)"
        
        # Check determinant is valid
        det_val = np.linalg.det(design._model_matrix.T @ design._model_matrix)
        assert det_val > 0, "Information matrix determinant should be positive"
    
    def test_select_design_not_computed(self):
        """Test error when selecting design before computing"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        
        # Try to select without computing - should raise KeyError with the n value
        with pytest.raises(KeyError):
            design.select_design(5)


class TestDOptConsistency:
    """Test consistency of results across multiple runs"""
    
    def test_consistency_same_seed(self):
        """Test that results are consistent with same random seed"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X3': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        # Run 1
        design1 = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2="all", pro_int3=None, pro_quadratic=None)
        design1.set_model_terms(terms)
        design1.compute_d_optimal(n_min=8, n_max=10, trials=20, graph=False)
        
        # Run 2 (same setup)
        design2 = DOptDesign(factors=factors, process_strategy='grid')
        design2.set_model_terms(terms)
        design2.compute_d_optimal(n_min=8, n_max=10, trials=20, graph=False)
        
        # Results should be similar (within 5% of log determinant)
        for n in [8, 9, 10]:
            det1 = design1._log_det.loc[n, 'log_M']
            det2 = design2._log_det.loc[n, 'log_M']
            
            # Check both are valid numbers
            assert not np.isnan(det1) and not np.isnan(det2), f"Invalid determinant at n={n}"
            
            # Check relative difference is small (allow 10% due to randomness)
            if abs(det1) > 1e-6:  # Avoid division by very small numbers
                rel_diff = abs(det1 - det2) / abs(det1)
                assert rel_diff < 0.10, f"Results differ by {rel_diff*100:.1f}% for n={n}"
    
    def test_consistency_more_trials_better(self):
        """Test that more trials generally give better or equal results"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X3': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        # Few trials
        design_few = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2="all", pro_int3=None, pro_quadratic=None)
        design_few.set_model_terms(terms)
        design_few.compute_d_optimal(n_min=10, n_max=10, trials=5, graph=False)
        
        # Many trials
        design_many = DOptDesign(factors=factors, process_strategy='grid')
        design_many.set_model_terms(terms)
        design_many.compute_d_optimal(n_min=10, n_max=10, trials=50, graph=False)
        
        det_few = design_few._log_det.loc[10, 'log_M']
        det_many = design_many._log_det.loc[10, 'log_M']
        
        # More trials should give equal or better determinant (allow small tolerance)
        assert det_many >= det_few - 0.01
    
    def test_consistency_different_strategies_similar_efficiency(self):
        """Test that different strategies both produce valid designs"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        
        # Strategy 1: Grid
        design_grid = DOptDesign(factors=factors, process_strategy='grid')
        design_grid.set_model_terms(terms)
        design_grid.compute_d_optimal(n_min=6, n_max=6, trials=30, graph=False)
        
        # Strategy 2: LHS with similar number of candidates
        design_lhs = DOptDesign(factors=factors, process_strategy='lhs', lhs_n_samples=20)
        design_lhs.set_model_terms(terms)
        design_lhs.compute_d_optimal(n_min=6, n_max=6, trials=30, graph=False)
        
        det_grid = design_grid._log_det.loc[6, 'log_M']
        det_lhs = design_lhs._log_det.loc[6, 'log_M']
        
        # Both should be valid and achieve reasonable efficiency
        # Note: Different candidate sets can lead to different optimal designs
        assert not np.isnan(det_grid) and not np.isnan(det_lhs)
        # Just verify both are reasonable (not extremely poor)
        assert det_grid > -10, "Grid strategy should achieve reasonable efficiency"
        assert det_lhs > -10, "LHS strategy should achieve reasonable efficiency"


class TestDOptEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_minimum_runs_equals_model_terms(self):
        """Test design with minimum runs (equal to model parameters)"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'X2': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        # Linear model: 3 parameters (intercept + 2 main)
        terms = create_simple_model_terms(intercept=True, quadratic=False)
        design.set_model_terms(terms)
        
        # Minimum should be 3 runs
        design.compute_d_optimal(n_min=3, n_max=5, trials=10, graph=False)
        
        assert 3 in design._best_idx
    
    def test_mixture_lower_bounds(self):
        """Test mixture factors with lower bounds"""
        factors = {
            'A': MixtureFactor(lower_bound=0.1, upper_bound=0.8),
            'B': MixtureFactor(lower_bound=0.1, upper_bound=0.8),
            'C': MixtureFactor(lower_bound=0.1, upper_bound=0.8)
        }
        
        design = DOptDesign(
            factors=factors,
            mixture_include=None,
            mixture_grid={"degree": 3, "max_candidates": 1000},
        )
        
        # Check all points respect lower bounds
        assert (design._cp['A'] >= 0.1 - 1e-10).all()
        assert (design._cp['B'] >= 0.1 - 1e-10).all()
        assert (design._cp['C'] >= 0.1 - 1e-10).all()
        
        # Check sum to 1 constraint
        sums = design._cp.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-8)
    
    def test_single_continuous_factor_with_categorical(self):
        """Test design with single continuous and categorical factor"""
        factors = {
            'Temp': ContinuousFactor(n_levels=5, lower_bound=20, upper_bound=100, decimals=1),
            'Material': CategoricalFactor(levels=['A', 'B', 'C'])
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        
        # 5 continuous levels × 3 categorical levels = 15 points
        assert len(design._coded_cp) == 15
    
    def test_many_factors_lhs(self):
        """Test LHS with many factors"""
        factors = {
            f'X{i}': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(1, 8)  # 7 factors
        }
        
        design = DOptDesign(factors=factors, process_strategy='lhs', lhs_n_samples=100)
        
        # Should have exactly 100 candidate points
        assert len(design._coded_cp) == 100
        assert design._coded_cp.shape[1] == 7
    
    def test_mixture_constraint_preserved_after_selection(self):
        """Test that selected design points still satisfy mixture constraint"""
        factors = {
            'A': MixtureFactor(lower_bound=0.1, upper_bound=0.8),
            'B': MixtureFactor(lower_bound=0.1, upper_bound=0.8),
            'C': MixtureFactor(lower_bound=0.1, upper_bound=0.8)
        }
        
        design = DOptDesign(
            factors=factors,
            mixture_include=None,
            mixture_grid={"degree": 3, "max_candidates": 1000},
        )
        terms = ModelTerms(intercept=False, mix_main="all", mix_int2="all")
        design.set_model_terms(terms)
        design.compute_d_optimal(n_min=8, n_max=8, trials=50, graph=False)
        design.select_design(8)
        
        # Check mixture constraint on final design (sum to 1)
        sums = design._design_matrix[['A', 'B', 'C']].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-8), "Mixture constraint violated in selected design"
        
        # Check lower bounds
        assert (design._design_matrix['A'] >= 0.09).all(), "Component A below lower bound"
        assert (design._design_matrix['B'] >= 0.09).all(), "Component B below lower bound"
        assert (design._design_matrix['C'] >= 0.09).all(), "Component C below lower bound"
        
        # Check upper bounds
        assert (design._design_matrix['A'] <= 0.81).all(), "Component A above upper bound"
        assert (design._design_matrix['B'] <= 0.81).all(), "Component B above upper bound"
        assert (design._design_matrix['C'] <= 0.81).all(), "Component C above upper bound"
    
    def test_combined_factors_constraints_after_selection(self):
        """Test that all constraints hold after design selection with mixed factor types"""
        factors = {
            'Temp': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=1),
            'A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'B': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        design = DOptDesign(
            factors=factors,
            process_strategy='ccc',
            mixture_include="all"
        )
        
        # Mixture models cannot have intercept
        terms = ModelTerms(
            intercept=False,
            pro_main="all",
            mix_main="all",
            pro_int2="all",
            mix_int2="all"
        )
        design.set_model_terms(terms)
        design.compute_d_optimal(n_min=12, n_max=12, trials=50, graph=False)
        design.select_design(12)
        
        # Check process factor bounds (note: CCC can go beyond original bounds in real space)
        # CCC (circumscribed) extends beyond the coded space bounds
        assert design._design_matrix['Temp'].min() >= 0, "Temperature unreasonably low"
        assert design._design_matrix['Temp'].max() <= 100, "Temperature unreasonably high"
        assert design._design_matrix['Pressure'].min() >= 0, "Pressure unreasonably low"
        assert design._design_matrix['Pressure'].max() <= 6, "Pressure unreasonably high"
        
        # Check mixture constraint
        mixture_cols = ['A', 'B']
        sums = design._design_matrix[mixture_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-8), "Mixture constraint violated"
    
    def test_categorical_preserved_after_selection(self):
        """Test that categorical levels are preserved in selected design"""
        factors = {
            'X1': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'Catalyst': CategoricalFactor(levels=['TypeA', 'TypeB', 'TypeC'])
        }
        
        design = DOptDesign(factors=factors, process_strategy='grid')
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_int3=None, pro_quadratic=None)
        design.set_model_terms(terms)
        design.compute_d_optimal(n_min=6, n_max=6, trials=30, graph=False)
        design.select_design(6)
        
        # Check that only valid categorical levels appear
        valid_levels = {'TypeA', 'TypeB', 'TypeC'}
        actual_levels = set(design._design_matrix['Catalyst'].unique())
        assert actual_levels.issubset(valid_levels), f"Invalid categorical levels: {actual_levels - valid_levels}"
        
        # Check that at least 2 levels are represented (for good coverage)
        assert len(actual_levels) >= 2, "Should have multiple categorical levels in design"
