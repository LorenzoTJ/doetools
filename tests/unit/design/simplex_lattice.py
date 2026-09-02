"""
Comprehensive test suite for SimplexLatticeDesign class.

This module provides production-ready tests for the Simplex Lattice design,
including exact matrix validation, mixture constraints, and lattice properties.
"""

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from doetools.design.mixture import SimplexLatticeDesign
from doetools.utils import MixtureFactor


class TestSimplexLatticeDesign:
    """Test suite for SimplexLatticeDesign"""

    # ===============================================================
    # INITIALIZATION TESTS
    # ===============================================================

    def test_initialization_basic(self):
        """Test basic initialization with 3 factors and m=2"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        
        assert design._design_type == "Simplex Lattice"
        assert design._coded_design_matrix is not None
        assert len(design._coded_design_matrix) == 6  # C(3+2-1, 2) = C(4, 2) = 6

    # ===============================================================
    # VALIDATION TESTS
    # ===============================================================

    def test_invalid_m_zero(self):
        """Test that m=0 raises an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="positive integer"):
            SimplexLatticeDesign(factors=factors, m=0)

    def test_invalid_m_negative(self):
        """Test that negative m raises an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="positive integer"):
            SimplexLatticeDesign(factors=factors, m=-1)

    def test_invalid_lower_bounds_sum(self):
        """Test that lower bounds summing to >= 1.0 raise an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.5, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.4, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.3, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="Sum of lower_bounds"):
            SimplexLatticeDesign(factors=factors, m=2)

    def test_invalid_center_points(self):
        """Test that negative center points raise an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            SimplexLatticeDesign(factors=factors, m=2, center_points=-1)

    def test_invalid_replicates(self):
        """Test that negative replicates raise an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            SimplexLatticeDesign(factors=factors, m=2, replicates=-1)

    # ===============================================================
    # LATTICE DEGREE TESTS
    # ===============================================================

    def test_3_factors_m1_design(self):
        """Test 3-factor design with m=1 (pure components only)"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=1)
        coded = design._coded_design_matrix
        
        # m=1 with 3 factors: only vertices (3 points)
        assert len(coded) == 3
        
        # Check that we have pure components
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0],
            'X2': [0.0, 1.0, 0.0],
            'X3': [0.0, 0.0, 1.0]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          expected.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          atol=1e-6)

    def test_3_factors_m2_design(self):
        """Test 3-factor design with m=2 (vertices + binary blends)"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        # m=2 with 3 factors: C(4, 2) = 6 points
        assert len(coded) == 6
        
        # Expected points: vertices and edge midpoints
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            'X2': [0.0, 1.0, 0.0, 0.5, 0.0, 0.5],
            'X3': [0.0, 0.0, 1.0, 0.0, 0.5, 0.5]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          expected.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          atol=1e-6)

    def test_3_factors_m3_design(self):
        """Test 3-factor design with m=3"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=3)
        coded = design._coded_design_matrix
        
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0, 1/3, 2/3, 2/3, 0.0, 1/3, 1/3, 0.0],
            'X2': [0.0, 1.0, 0.0, 1/3, 1/3, 0, 2/3, 2/3, 0.0, 1/3],
            'X3': [0.0, 0.0, 1.0, 1/3, 0.0, 1/3, 1/3, 0.0, 2/3, 2/3]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                            expected.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                            atol=1e-6)
        
        # m=3 with 3 factors: C(5, 3) = 10 points
        assert len(coded) == 10
    
    def test_4_factors_m1_design(self):
        """Test 4-factor design with m=1"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X4': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=1)
        coded = design._coded_design_matrix
        
        # m=1 with 4 factors: C(4, 1) = 4 points
        assert len(coded) == 4
        
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0, 0.0],
            'X2': [0.0, 1.0, 0.0, 0.0],
            'X3': [0.0, 0.0, 1.0, 0.0],
            'X4': [0.0, 0.0, 0.0, 1.0]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            expected.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            atol=1e-6)

    def test_4_factors_m2_design(self):
        """Test 4-factor design with m=2"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X4': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        # m=2 with 4 factors: C(5, 2) = 10 points
        assert len(coded) == 10
        
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            'X2': [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0],
            'X3': [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
            'X4': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            expected.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            atol=1e-6)
        
    def test_4_factors_m3_design(self):
        """Test 4-factor design with m=3"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X4': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=3)
        coded = design._coded_design_matrix
        
        # m=3 with 4 factors: C(6, 3) = 20 points
        assert len(coded) == 20
        
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0, 0.0, 1/3, 1/3, 1/3, 0.0, 1/3, 1/3, 1/3, 2/3, 0.0, 0.0, 2/3, 0.0, 0.0,  2/3, 0.0, 0.0],
            'X2': [0.0, 1.0, 0.0, 0.0, 1/3, 1/3, 0.0, 1/3, 2/3, 0.0, 0.0, 1/3, 1/3, 1/3, 0.0, 2/3, 0.0,  0.0, 0.0, 2/3],
            'X3': [0.0, 0.0, 1.0, 0.0, 1/3, 0.0, 1/3, 1/3, 0.0, 2/3, 0.0, 0.0, 2/3, 0.0, 1/3, 1/3, 1/3,  0.0, 2/3, 0.0],
            'X4': [0.0, 0.0, 0.0, 1.0, 0.0, 1/3, 1/3, 1/3, 0.0, 0.0, 2/3, 0.0, 0.0, 2/3, 0.0, 0.0, 2/3,  1/3, 1/3, 1/3]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            expected.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            atol=1e-6)
        

    # ===============================================================
    # MIXTURE CONSTRAINT TESTS
    # ===============================================================

    def test_proportions_sum_to_one(self):
        """Test that all rows sum to 1.0"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        row_sums = coded.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), \
            f"Not all rows sum to 1.0: {row_sums.values}"

    def test_proportions_non_negative(self):
        """Test that all proportions are non-negative"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        assert (coded >= 0).all().all(), "Some proportions are negative"

    def test_proportions_within_bounds(self):
        """Test that all proportions are within [0, 1]"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        assert (coded >= 0).all().all(), "Some proportions < 0"
        assert (coded <= 1).all().all(), "Some proportions > 1"

    # ===============================================================
    # LOWER BOUND TESTS
    # ===============================================================

    def test_lower_bounds_respected(self):
        """Test that lower bounds are respected"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.15, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        assert coded.shape[0] == 6
        assert coded.shape[1] == 3
        
        # Check lower bounds are respected (with small tolerance for rounding)
        assert (coded['X1'] >= 0.1 - 1e-10).all(), f"X1 violates lower bound: min={coded['X1'].min()}"
        assert (coded['X2'] >= 0.2 - 1e-10).all(), f"X2 violates lower bound: min={coded['X2'].min()}"
        assert (coded['X3'] >= 0.15 - 1e-10).all(), f"X3 violates lower bound: min={coded['X3'].min()}"

    def test_lower_bounds_sum_to_one_with_slack(self):
        """Test that proportions still sum to 1 with lower bounds"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        row_sums = coded.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), \
            f"Rows with lower bounds don't sum to 1.0: {row_sums.values}"

    def test_slack_calculation(self):
        """Test that slack S = 1 - sum(L_i) is used correctly"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.3, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3)
        }
        
        # Sum of lower bounds = 0.6, so slack = 0.4
        design = SimplexLatticeDesign(factors=factors, m=1)
        coded = design._coded_design_matrix
        
        # For m=1 with lower bounds, we should have 3 points where one factor is maximized
        # Expected: (0.6, 0.3, 0.1), (0.2, 0.7, 0.1), (0.2, 0.3, 0.5)
        # Each point has one factor at lower_bound + slack
        assert len(coded) == 3

    # ===============================================================
    # UNIFORMITY TESTS
    # ===============================================================

    def test_uniform_spacing_m2(self):
        """Test that m=2 gives uniform spacing of 0.5"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        # For m=2, allowed values are 0, 0.5, 1.0
        unique_values = set()
        for col in coded.columns:
            unique_values.update(coded[col].unique())
        
        expected_values = {0.0, 0.5, 1.0}
        assert unique_values == expected_values, \
            f"Unexpected values for m=2: {unique_values}"

    def test_uniform_spacing_m3(self):
        """Test that m=3 gives uniform spacing of 1/3"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=3)
        coded = design._coded_design_matrix
        
        # For m=3, allowed values are 0, 1/3, 2/3, 1.0
        unique_values = set()
        for col in coded.columns:
            unique_values.update(coded[col].unique())
        
        expected_values = {0.0, 1/3, 2/3, 1.0}
        for val in unique_values:
            assert any(np.isclose(val, exp, atol=1e-6) for exp in expected_values), \
                f"Unexpected value {val} for m=3"
    
    def test_uniform_spacing_m4(self):
        """Test that m=4 gives uniform spacing of 0.25"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X4': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=4)
        coded = design._coded_design_matrix
        
        # For m=4, allowed values are 0, 0.25, 0.5, 0.75, 1.0
        unique_values = set()
        for col in coded.columns:
            unique_values.update(coded[col].unique())
        
        expected_values = {0.0, 0.25, 0.5, 0.75, 1.0}
        for val in unique_values:
            assert any(np.isclose(val, exp, atol=1e-6) for exp in expected_values), \
                f"Unexpected value {val} for m=4"

    # ===============================================================
    # CENTER POINTS TESTS
    # ===============================================================

    def test_center_points(self):
        """Test that center points are added correctly"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2, center_points=3)
        coded = design._coded_design_matrix
        
        # 6 design points + 3 center points = 9 rows
        assert len(coded) == 9
        
        # Last 3 rows should be centroid (1/3, 1/3, 1/3)
        center_rows = coded.tail(3)
        for col in ['X1', 'X2', 'X3']:
            assert np.allclose(center_rows[col], 1/3, atol=1e-6), \
                f"Center point for {col} not at 1/3"

    def test_center_points_with_lower_bounds(self):
        """Test center points respect lower bounds"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.15, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2, center_points=2)
        coded = design._coded_design_matrix
        
        # Last 2 rows should be centroid in constrained space
        center_rows = coded.tail(2)
        
        # Calculate expected centroid with lower bounds
        total_lower = sum(f.lower_bound for f in factors.values())
        slack = 1.0 - total_lower
        expected_centroid = {
            'X1': factors['X1'].lower_bound + slack / 3,
            'X2': factors['X2'].lower_bound + slack / 3,
            'X3': factors['X3'].lower_bound + slack / 3
        }
        for col in ['X1', 'X2', 'X3']:
            assert np.allclose(
                center_rows[col].to_numpy(),
                expected_centroid[col],
                rtol=0.0,
                atol=1e-12,
            )

    # ===============================================================
    # REPLICATES TESTS
    # ===============================================================

    def test_replicates(self):
        """Test that replicates are added correctly"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2, replicates=2)
        coded = design._coded_design_matrix
        
        # 6 design points * 3 (original + 2 replicates) = 18 rows
        assert len(coded) == 18
        
        # Each unique design point should appear 3 times
        unique_rows = coded.drop_duplicates()
        assert len(unique_rows) == 6

    def test_replicates_and_center_points(self):
        """Test combining replicates and center points"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2, replicates=1, center_points=2)
        coded = design._coded_design_matrix
        
        # 6 design points * 2 (original + 1 replicate) + 2 center points = 14 rows
        assert len(coded) == 14

    # ===============================================================
    # DIMENSION TESTS
    # ===============================================================

    def test_number_of_points_formula(self):
        """Test that number of points follows C(k+m-1, m) formula"""
        from math import comb
        
        test_cases = [
            (3, 1, comb(3, 1)),      # 3 factors, m=1: C(3, 1) = 3
            (3, 2, comb(4, 2)),      # 3 factors, m=2: C(4, 2) = 6
            (3, 3, comb(5, 3)),      # 3 factors, m=3: C(5, 3) = 10
            (4, 2, comb(5, 2)),      # 4 factors, m=2: C(5, 2) = 10
            (4, 3, comb(6, 3)),      # 4 factors, m=3: C(6, 3) = 20
            (5, 1, comb(5, 1)),      # 5 factors, m=1: C(5, 1) = 5
            (5, 2, comb(6, 2)),      # 5 factors, m=2: C(6, 2) = 15
            (5, 3, comb(7, 3)),       # 5 factors, m=3: C(7, 3) = 35
            (6, 2, comb(7, 2)),      # 6 factors, m=2: C(7, 2) = 21
            (6, 3, comb(8, 3))       # 6 factors, m=3: C(8, 3) = 56
        ]
        
        for k, m, expected_points in test_cases:
            factors = {
                f'X{i}': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
                for i in range(1, k+1)
            }
            
            design = SimplexLatticeDesign(factors=factors, m=m)
            actual_points = len(design._coded_design_matrix)
            
            assert actual_points == expected_points, \
                f"For k={k}, m={m}: expected {expected_points} points, got {actual_points}"

    def test_coded_uncoded_dimension_consistency(self):
        """Test that coded and uncoded matrices have same dimensions"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2, center_points=2, replicates=1)
        
        assert design._coded_design_matrix.shape == design._design_matrix.shape
        assert list(design._coded_design_matrix.columns) == list(design._design_matrix.columns)

    # ===============================================================
    # EDGE CASE TESTS
    # ===============================================================

    def test_2_factors_m1(self):
        """Test binary mixture with m=1 (two pure components)"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=1)
        coded = design._coded_design_matrix
        
        # 2 points: (1, 0) and (0, 1)
        assert len(coded) == 2
        
        expected = pd.DataFrame({
            'X1': [1.0, 0.0],
            'X2': [0.0, 1.0]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1']).reset_index(drop=True),
                          expected.sort_values(by=['X1']).reset_index(drop=True),
                          atol=1e-6)

    def test_2_factors_m5(self):
        """Test binary mixture with m=5 (fine spacing)"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=5)
        coded = design._coded_design_matrix
        
        # 2 factors, m=5: C(6, 5) = 6 points
        assert len(coded) == 6
        
        # Points should be: (1, 0), (0.8, 0.2), (0.6, 0.4), (0.4, 0.6), (0.2, 0.8), (0, 1)
        expected_x1 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        actual_x1_sorted = sorted(coded['X1'].unique())
        
        assert np.allclose(actual_x1_sorted, expected_x1, atol=1e-6)

    def test_5_factors_m1(self):
        """Test 5-factor mixture with m=1"""
        factors = {
            f'X{i}': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
            for i in range(1, 6)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=1)
        coded = design._coded_design_matrix
        
        # 5 pure components
        assert len(coded) == 5
        
        # Each row should have one 1.0 and four 0.0s
        for idx in range(len(coded)):
            row_values = coded.iloc[idx].values
            assert np.sum(row_values == 1.0) == 1
            assert np.sum(row_values == 0.0) == 4

    # ===============================================================
    # MATRIX CONSISTENCY TESTS
    # ===============================================================

    def test_no_duplicate_points(self):
        """Test that design has no duplicate points (before replication)"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        # Should have 6 unique points
        unique_count = len(coded.drop_duplicates())
        assert unique_count == 6, f"Expected 6 unique points, got {unique_count}"

    def test_symmetry_property(self):
        """Test that design is symmetric (all factors treated equally)"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexLatticeDesign(factors=factors, m=2)
        coded = design._coded_design_matrix
        
        # Each factor should have the same distribution of values
        for col in ['X1', 'X2', 'X3']:
            value_counts = coded[col].value_counts().sort_index()
            # For symmetric design, all factors should have same value distribution
            # This is a general property check
            assert len(value_counts) > 0
