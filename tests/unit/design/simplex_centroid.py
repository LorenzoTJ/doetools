"""
Comprehensive test suite for SimplexCentroidDesign class.

This module provides production-ready tests for the Simplex Centroid design,
including exact matrix validation, mixture constraints, and centroid properties.
"""

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from doetools.design.mixture import SimplexCentroidDesign
from doetools.utils import MixtureFactor


class TestSimplexCentroidDesign:
    """Test suite for SimplexCentroidDesign"""

    # ===============================================================
    # INITIALIZATION TESTS
    # ===============================================================

    def test_initialization_basic(self):
        """Test basic initialization with 3 factors"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        
        assert design._design_type == "Simplex Centroid"
        assert design._coded_design_matrix is not None
        assert len(design._coded_design_matrix) == 7  # 2^3 - 1 = 7

    # ===============================================================
    # VALIDATION TESTS
    # ===============================================================

    def test_invalid_lower_bounds_sum(self):
        """Test that lower bounds summing to >= 1.0 raise an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.5, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.4, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.3, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="Sum of lower_bounds"):
            SimplexCentroidDesign(factors=factors)

    def test_invalid_center_points(self):
        """Test that negative center points raise an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            SimplexCentroidDesign(factors=factors, center_points=-1)

    def test_invalid_replicates(self):
        """Test that negative replicates raise an error"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            SimplexCentroidDesign(factors=factors, replicates=-1)

    # ===============================================================
    # EXACT MATRIX TESTS
    # ===============================================================

    def test_2_factor_centroid_matrix(self):
        """Test exact matrix for 2-factor centroid design"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 2^2 - 1 = 3 points: two vertices + one binary blend
        assert len(coded) == 3
        
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.5],
            'X2': [0.0, 1.0, 0.5]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2']).reset_index(drop=True),
                          expected.sort_values(by=['X1', 'X2']).reset_index(drop=True),
                          atol=1e-6)

    def test_3_factor_centroid_matrix(self):
        """Test exact matrix for 3-factor centroid design"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 2^3 - 1 = 7 points
        assert len(coded) == 7
        
        # Expected points:
        # - 3 vertices: (1,0,0), (0,1,0), (0,0,1)
        # - 3 binary blends: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        # - 1 ternary blend: (1/3,1/3,1/3)
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1/3],
            'X2': [0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 1/3],
            'X3': [0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1/3]
        })
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          expected.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          atol=1e-6)
    
    def test_3_factor_centroid_constraints_matrix(self):
        
        """Test exact matrix for 3-factor centroid design with constraints"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Expected points after applying constraints
        expected = pd.DataFrame({
            'X1': [0.8, 0.2, 0.2, 0.4, 0.5, 0.5, 0.2],
            'X2': [0.1, 0.7, 0.1, 0.3, 0.4, 0.1, 0.4],
            'X3': [0.1, 0.1, 0.7, 0.3, 0.1, 0.4, 0.4]
        })
        
        assert len(coded) == 7
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          expected.sort_values(by=['X1', 'X2', 'X3']).reset_index(drop=True),
                          atol=1e-6)

    def test_4_factor_centroid_dimensions(self):
        """Test dimensions for 4-factor centroid design"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X4': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        expected = pd.DataFrame({
            'X1': [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1/3, 0.0, 1/3, 1/3, 0.25],
            'X2': [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 1/3, 1/3, 0.0, 1/3, 0.25],
            'X3': [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 1/3, 1/3, 1/3, 0.0, 0.25],
            'X4': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 1/3, 1/3, 1/3, 0.25]
        })
        
        # 2^4 - 1 = 15 points
        assert len(coded) == 15
        assert coded.shape[1] == 4
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                    expected.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                    atol=1e-6)
    
    def test_4_factor_centroid_constraints_matrix(self):
        """Test exact matrix for 4-factor centroid design with constraints"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3),
            'X4': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Expected points after applying constraints
        expected = pd.DataFrame({
            "X1": [0.5, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 7/30, 7/30, 7/30, 0.1, 0.2],
            "X2": [0.2, 0.6, 0.2, 0.2, 0.4, 0.2, 0.2, 0.4, 0.4, 0.2, 1/3, 1/3, 0.2, 1/3, 0.3],
            "X3": [0.1, 0.1, 0.5, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 7/30, 0.1,7/30, 7/30, 0.2],
            "X4": [0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 1/3, 1/3, 1/3, 0.3],
        })
        
        assert len(coded) == 15
        
        assert_frame_equal(coded.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            expected.sort_values(by=['X1', 'X2', 'X3', 'X4']).reset_index(drop=True),
                            atol=1e-6)
        

    # ===============================================================
    # NUMBER OF POINTS TESTS
    # ===============================================================

    def test_number_of_points_formula(self):
        """Test that number of points follows 2^k - 1 formula"""
        test_cases = [
            (2, 2**2 - 1),  # 2 factors: 3 points
            (3, 2**3 - 1),  # 3 factors: 7 points
            (4, 2**4 - 1),  # 4 factors: 15 points
            (5, 2**5 - 1),  # 5 factors: 31 points
            (6, 2**6 - 1),  # 6 factors: 63 points
            (7, 2**7 - 1)   # 7 factors: 127 points
        ]
        
        for k, expected_points in test_cases:
            factors = {
                f'X{i}': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
                for i in range(1, k+1)
            }
            
            design = SimplexCentroidDesign(factors=factors)
            actual_points = len(design._coded_design_matrix)
            
            assert actual_points == expected_points, \
                f"For k={k}: expected {expected_points} points, got {actual_points}"

    def test_point_type_count_3_factors(self):
        """Test that 3-factor design has correct number of each point type"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Count point types by number of non-zero components
        point_types = []
        for idx in range(len(coded)):
            row = coded.iloc[idx]
            non_zero_count = (row > 1e-10).sum()
            point_types.append(non_zero_count)
        
        # Should have:
        # - 3 vertices (1 non-zero component)
        # - 3 binary blends (2 non-zero components)
        # - 1 ternary blend (3 non-zero components)
        assert point_types.count(1) == 3, "Should have 3 vertices"
        assert point_types.count(2) == 3, "Should have 3 binary blends"
        assert point_types.count(3) == 1, "Should have 1 ternary blend"

    def test_point_type_count_4_factors(self):
        """Test that 4-factor design has correct number of each point type"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X4': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Count point types
        point_types = []
        for idx in range(len(coded)):
            row = coded.iloc[idx]
            non_zero_count = (row > 1e-10).sum()
            point_types.append(non_zero_count)
        
        # Should have: C(4,1) + C(4,2) + C(4,3) + C(4,4) = 4 + 6 + 4 + 1 = 15
        from math import comb
        assert point_types.count(1) == comb(4, 1), "Wrong number of vertices"
        assert point_types.count(2) == comb(4, 2), "Wrong number of binary blends"
        assert point_types.count(3) == comb(4, 3), "Wrong number of ternary blends"
        assert point_types.count(4) == comb(4, 4), "Wrong number of quaternary blend"

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
        
        design = SimplexCentroidDesign(factors=factors)
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
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        assert (coded >= 0).all().all(), "Some proportions are negative"

    def test_proportions_within_bounds(self):
        """Test that all proportions are within [0, 1]"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        assert (coded >= 0).all().all(), "Some proportions < 0"
        assert (coded <= 1).all().all(), "Some proportions > 1"

    # ===============================================================
    # CENTROID PROPERTY TESTS
    # ===============================================================

    def test_vertices_exist(self):
        """Test that all pure component vertices exist"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Check for vertices (1, 0, 0), (0, 1, 0), (0, 0, 1)
        vertices = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        
        for vertex in vertices:
            # Check if this vertex exists in the design
            matches = coded.apply(lambda row: np.allclose(row.values, vertex, atol=1e-6), axis=1)
            assert matches.any(), f"Vertex {vertex} not found in design"

    def test_binary_blends_exist(self):
        """Test that all binary blend centroids exist"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Check for binary blends (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        binary_blends = [
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
        
        for blend in binary_blends:
            matches = coded.apply(lambda row: np.allclose(row.values, blend, atol=1e-6), axis=1)
            assert matches.any(), f"Binary blend {blend} not found in design"

    def test_overall_centroid_exists(self):
        """Test that overall centroid exists"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Check for overall centroid (1/3, 1/3, 1/3)
        centroid = [1/3, 1/3, 1/3]
        matches = coded.apply(lambda row: np.allclose(row.values, centroid, atol=1e-6), axis=1)
        assert matches.any(), "Overall centroid (1/3, 1/3, 1/3) not found in design"

    def test_centroid_values_correct(self):
        """Test that centroid values are correct for each subset size"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # For each row, non-zero components should have equal values
        for idx in range(len(coded)):
            row = coded.iloc[idx]
            non_zero = row[row > 1e-10]
            
            if len(non_zero) > 0:
                # All non-zero values should be equal to 1/m where m is the subset size
                expected_value = 1.0 / len(non_zero)
                assert np.allclose(non_zero.values, expected_value, atol=1e-6), \
                    f"Row {idx}: non-zero values not equal to 1/{len(non_zero)}"

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
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
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
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        row_sums = coded.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), \
            f"Rows with lower bounds don't sum to 1.0: {row_sums.values}"

    def test_constrained_vertices(self):
        """Test that vertices respect constraints in constrained region"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.3, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # With constraints, "vertices" are not pure components anymore
        # Check that all points are valid
        assert len(coded) == 7
        
        # All rows should sum to 1.0
        row_sums = coded.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

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
        
        design = SimplexCentroidDesign(factors=factors, center_points=3)
        coded = design._coded_design_matrix
        
        # 7 design points + 3 center points = 10 rows
        assert len(coded) == 10
        
        # Last 3 rows should be centroid (1/3, 1/3, 1/3)
        center_rows = coded.tail(3)
        for col in ['X1', 'X2', 'X3']:
            assert np.allclose(center_rows[col], 1/3, atol=1e-6), \
                f"Center point for {col} not at 1/3"

    def test_center_points_with_lower_bounds(self):
        """Test center points in constrained region"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.2, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.1, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.15, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors, center_points=2)
        coded = design._coded_design_matrix
        
        # Last 2 rows should be centroid in constrained space
        center_rows = coded.tail(2)
        
        # Check validity
        for col in ['X1', 'X2', 'X3']:
            assert (center_rows[col] >= factors[col].lower_bound - 1e-10).all()

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
        
        design = SimplexCentroidDesign(factors=factors, replicates=2)
        coded = design._coded_design_matrix
        
        # 7 design points * 3 (original + 2 replicates) = 21 rows
        assert len(coded) == 21
        
        # Each unique design point should appear 3 times
        unique_rows = coded.drop_duplicates()
        assert len(unique_rows) == 7

    def test_replicates_and_center_points(self):
        """Test combining replicates and center points"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors, replicates=1, center_points=2)
        coded = design._coded_design_matrix
        
        # 7 design points * 2 (original + 1 replicate) + 2 center points = 16 rows
        assert len(coded) == 16

    # ===============================================================
    # DIMENSION TESTS
    # ===============================================================

    def test_coded_uncoded_dimension_consistency(self):
        """Test that coded and uncoded matrices have same dimensions"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors, center_points=2, replicates=1)
        
        assert design._coded_design_matrix.shape == design._design_matrix.shape
        assert list(design._coded_design_matrix.columns) == list(design._design_matrix.columns)

    def test_5_factor_design_dimensions(self):
        """Test dimensions for 5-factor design"""
        factors = {
            f'X{i}': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
            for i in range(1, 6)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 2^5 - 1 = 31 points
        assert len(coded) == 31
        assert coded.shape[1] == 5

    # ===============================================================
    # EDGE CASE TESTS
    # ===============================================================

    def test_2_factors_design(self):
        """Test binary mixture centroid design"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 3 points: (1, 0), (0, 1), (0.5, 0.5)
        assert len(coded) == 3

    def test_no_duplicate_points(self):
        """Test that design has no duplicate points (before replication)"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # Should have 7 unique points
        unique_count = len(coded.drop_duplicates())
        assert unique_count == 7, f"Expected 7 unique points, got {unique_count}"

    def test_clipping_for_feasibility(self):
        """Test that negative values from mapping are clipped to 0"""
        # With tight lower bounds, mapping might produce small negative values
        factors = {
            'X1': MixtureFactor(lower_bound=0.3, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.3, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.3, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # All values should be non-negative due to clipping
        assert (coded >= 0).all().all(), "Found negative values after clipping"

    # ===============================================================
    # SYMMETRY TESTS
    # ===============================================================

    def test_symmetry_property(self):
        """Test that design treats all factors symmetrically"""
        factors = {
            'X1': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X2': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3),
            'X3': MixtureFactor(lower_bound=0.0, upper_bound=1.0, decimals=3)
        }
        
        design = SimplexCentroidDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # For symmetric unconstrained design, each factor should appear
        # in the same types of centroids with same frequencies
        # Check that max value for each factor is 1.0 (appears in a vertex)
        for col in ['X1', 'X2', 'X3']:
            assert np.isclose(coded[col].max(), 1.0, atol=1e-6), \
                f"Factor {col} max is not 1.0"
