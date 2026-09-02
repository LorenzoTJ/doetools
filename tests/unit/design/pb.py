"""
Comprehensive test suite for PlackettBurmanDesign class.

This module provides production-ready tests for the Plackett-Burman design,
including exact matrix validation, structural properties, and edge cases.
"""

import pytest
import numpy as np

from doetools.design.process import PlackettBurmanDesign
from doetools.utils import ContinuousFactor, CategoricalFactor


class TestPlackettBurmanDesign:
    """Test suite for PlackettBurmanDesign"""

    # ===============================================================
    # INITIALIZATION TESTS
    # ===============================================================

    def test_initialization_basic(self):
        """Test basic initialization with 4 factors"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        
        assert design._design_type == "Plackett-Burman"
        assert len(design._coded_design_matrix) == 8
        assert "A" in design._coded_design_matrix.columns
        assert "B" in design._coded_design_matrix.columns
        assert "C" in design._coded_design_matrix.columns
        assert "D" in design._coded_design_matrix.columns

    # ===============================================================
    # VALIDATION TESTS
    # ===============================================================

    def test_invalid_factor_count_too_few(self):
        """Test that designs with less than 4 factors raise an error"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
        }
        
        with pytest.raises(ValueError, match="at least 4 factors"):
            PlackettBurmanDesign(factors=factors)

    def test_invalid_factor_count_too_many(self):
        """Test that designs with more than 24 factors raise an error"""
        factors = {
            f"F{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(25)
        }
        
        with pytest.raises(ValueError, match="at most 24 factors"):
            PlackettBurmanDesign(factors=factors)

    def test_invalid_center_points(self):
        """Test that negative center points raise an error"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            PlackettBurmanDesign(factors=factors, center_points=-1)
    
    def test_invalid_replicates(self):
        """Test that negative replicates raise an error"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            PlackettBurmanDesign(factors=factors, replicates=-2)

    def test_invalid_factor_levels(self):
        """Test that factors with more than 2 levels raise an error"""
        factors = {
            "A": ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=50, decimals=1),  # 3 levels!
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        with pytest.raises(ValueError, match="must have 2 levels"):
            PlackettBurmanDesign(factors=factors)

    # ===============================================================
    # RUN SIZE VALIDATION TESTS
    # ===============================================================
    @pytest.mark.parametrize("n_vars, expected_runs", [
        (4, 8),
        (5, 8),
        (7, 8),
        (8, 12),
        (11, 12),
        (12, 16),
        (15, 16),
        (16, 20),
        (19, 20),
        (20, 24),
        (23, 24),
    ])
    def test_next_valid_pb_run_size(self, n_vars, expected_runs):
        """Test run size calculation for various factor counts"""
        assert PlackettBurmanDesign.next_valid_pb_run_size(n_vars) == expected_runs

    def test_run_size_property_4_factors(self):
        """Test that 4 factors generate 8 runs"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        assert len(design._coded_design_matrix) == 8

    def test_run_size_property_8_factors(self):
        """Test that 8 factors generate 12 runs"""
        factors = {
            f"F{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(8)
        }
        
        design = PlackettBurmanDesign(factors=factors)
        assert len(design._coded_design_matrix) == 12

    def test_run_size_property_12_factors(self):
        """Test that 12 factors generate 16 runs"""
        factors = {
            f"F{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(12)
        }
        
        design = PlackettBurmanDesign(factors=factors)
        assert len(design._coded_design_matrix) == 16

    # ===============================================================
    # EXACT MATRIX TESTS
    # ===============================================================

    def test_4_factor_pb_matrix(self):
        """Test exact matrix for 4-factor PB design"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 4 factors + 3 dummy = 7 columns in 8 runs
        assert coded.shape == (8, 7)
        
        # Check generator row (first row)
        # For 8 runs: [1, 1, 1, -1, 1, -1, -1]
        assert list(coded.iloc[0].values) == [1, 1, 1, -1, 1, -1, -1]
        
        # Check final row (all -1)
        assert list(coded.iloc[7].values) == [-1, -1, -1, -1, -1, -1, -1]
        
        # All values should be -1 or 1
        assert set(coded.values.flatten()) == {-1, 1}
    
    def test_4_factor_exact_pb_matrix(self):
        
        """Test exact matrix for 4-factor PB design"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 4 factors + 3 dummy = 7 columns in 8 runs
        assert coded.shape == (8, 7)
        
        # Check generator row (first row)
        # For 8 runs: [1, 1, 1, -1, 1, -1, -1]
        assert list(coded.iloc[0].values) == [1, 1, 1, -1, 1, -1, -1]
        
        # Check final row (all -1)
        assert list(coded.iloc[7].values) == [-1, -1, -1, -1, -1, -1, -1]

        expected_matrix = np.array([
            [ 1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0],
            [-1.0,  1.0,  1.0,  1.0, -1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0,  1.0,  1.0, -1.0,  1.0],
            [ 1.0, -1.0, -1.0,  1.0,  1.0,  1.0, -1.0],
            [-1.0,  1.0, -1.0, -1.0,  1.0,  1.0,  1.0],
            [ 1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  1.0],
            [ 1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ])
        
        assert np.array_equal(coded.values, expected_matrix)

    def test_7_factor_pb_matrix(self):
        """Test exact matrix for 7-factor PB design (maximum for 8-run design)"""
        factors = {
            f"F{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(7)
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 7 factors, no dummy factors needed
        assert coded.shape == (8, 7)
        
        # Check that factor names are correct (F0 to F6)
        assert list(coded.columns) == [f"F{i}" for i in range(7)]
        
        # Check generator row
        assert list(coded.iloc[0].values) == [1, 1, 1, -1, 1, -1, -1]
        
        # Check final row (all -1)
        assert list(coded.iloc[7].values) == [-1, -1, -1, -1, -1, -1, -1]
        
        # All values should be -1 or 1
        assert set(coded.values.flatten()) == {-1, 1}

    def test_5_factor_pb_matrix_with_dummies(self):
        """Test that 5 factors generate 2 dummy factors"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
            "E": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=1, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 5 real factors + 2 dummy factors = 7 columns
        assert coded.shape == (8, 7)
        
        # Check that dummy factors are added
        assert "d6" in coded.columns
        assert "d7" in coded.columns
        
        # Check that dummy factors are in the factors dictionary
        assert "d6" in design._factors
        assert "d7" in design._factors
        assert isinstance(design._factors["d6"], CategoricalFactor)
        assert isinstance(design._factors["d7"], CategoricalFactor)

    def test_11_factor_pb_matrix(self):
        """Test exact matrix for 11-factor PB design (12-run design)"""
        factors = {
            f"F{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(11)
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 11 factors, 12 runs
        assert coded.shape == (12, 11)
        
        # Check generator row for 12 runs
        # [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]
        expected_first_row = [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]
        assert list(coded.iloc[0].values) == expected_first_row
        
        # Check final row (all -1)
        assert list(coded.iloc[11].values) == [-1] * 11

    # ===============================================================
    # STRUCTURAL PROPERTY TESTS
    # ===============================================================

    def test_coded_values_only_plus_minus_one(self):
        """Test that all coded values are -1 or 1"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        unique_values = set(coded.values.flatten())
        assert unique_values == {-1, 1}

    @pytest.mark.parametrize("num_factors", [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20, 22])
    def test_orthogonality_property(self, num_factors):
        """Test that columns are orthogonal (dot product ≈ 0)"""
        factors = {f"X{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
                        for i in range(num_factors)}
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        tol = 1e-8
        # Check orthogonality between first num_factors columns (real factors)
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                dot_product = np.dot(coded.iloc[:, i].values, coded.iloc[:, j].values)
                assert abs(dot_product) <= 0 + tol, f"Factors {i} and {j} not orthogonal: dot={dot_product}"

    @pytest.mark.parametrize("num_factors", [6, 9, 14, 18, 22])
    def test_balanced_levels(self, num_factors):
        """Test that each factor has equal numbers of -1 and 1 (or nearly equal)"""
        factors = {f"X{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
                     for i in range(num_factors)
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # For PB designs, each column should be nearly balanced
        for col in coded.columns:
            value_counts = coded[col].value_counts()
            n_minus_one = value_counts.get(-1, 0)
            n_plus_one = value_counts.get(1, 0)

            assert abs(n_minus_one - n_plus_one) == 0, f"Factor {col} not balanced"

    def test_uncoded_values_within_bounds(self):
        """Test that uncoded design matrix values are within factor bounds"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        uncoded = design._design_matrix
        
        # Check each real factor (exclude dummy factors)
        assert uncoded["A"].min() >= 10 and uncoded["A"].max() <= 50
        assert uncoded["B"].min() >= 5 and uncoded["B"].max() <= 15
        assert uncoded["C"].min() >= 1 and uncoded["C"].max() <= 5
        assert uncoded["D"].min() >= 2 and uncoded["D"].max() <= 8

    def test_uncoded_values_at_bounds(self):
        """Test that uncoded values are exactly at low/high bounds"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        uncoded = design._design_matrix
        
        # For 2-level designs, values should be exactly at low or high
        # Only check original factors, not dummy factors
        for factor_name, factor_obj in factors.items():
            if hasattr(factor_obj, 'lower_bound'):  # Only continuous factors
                unique_values = set(uncoded[factor_name].values)
                assert unique_values == {factor_obj.lower_bound, factor_obj.upper_bound}, \
                    f"Factor {factor_name} has unexpected values: {unique_values}"

    # ===============================================================
    # CENTER POINTS TESTS
    # ===============================================================

    def test_center_points(self):
        """Test that center points are added correctly"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors, center_points=3)
        coded = design._coded_design_matrix
        
        # 8 design points + 3 center points = 11 rows
        assert len(coded) == 11
        
        # Last 3 rows should be all zeros (center points in coded space)
        center_rows = coded.iloc[-3:, :4]  # Only check real factors
        assert (center_rows == 0).all().all()

    def test_center_points_uncoded(self):
        """Test that center points in uncoded space are at midpoints"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors, center_points=2)
        uncoded = design._design_matrix
        
        # Last 2 rows should be at midpoints
        expected_centers = {
            "A": 30.0,  # (10 + 50) / 2
            "B": 10.0,  # (5 + 15) / 2
            "C": 3.0,   # (1 + 5) / 2
            "D": 5.0,   # (2 + 8) / 2
        }
        
        for factor_name, expected_value in expected_centers.items():
            actual_value = uncoded[factor_name].iloc[-1]
            assert np.isclose(actual_value, expected_value, atol=0.1), \
                f"Center point for {factor_name}: {actual_value} != {expected_value}"

    # ===============================================================
    # REPLICATES TESTS
    # ===============================================================

    def test_replicates(self):
        """Test that replicates are added correctly"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors, replicates=2)
        coded = design._coded_design_matrix
        
        # 8 design points * 3 (original + 2 replicates) = 24 rows
        assert len(coded) == 24
        
        # Each unique design point should appear 3 times
        unique_rows = coded.drop_duplicates()
        assert len(unique_rows) == 8

    def test_replicates_and_center_points(self):
        """Test combining replicates and center points"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors, replicates=1, center_points=2)
        coded = design._coded_design_matrix
        
        # 8 design points * 2 (original + 1 replicate) + 2 center points = 18 rows
        assert len(coded) == 18
        
        # Last 2 rows should be center points (all zeros for real factors)
        center_rows = coded.iloc[-2:, :4]
        assert (center_rows == 0).all().all()

    # ===============================================================
    # CATEGORICAL FACTORS TESTS
    # ===============================================================

    def test_categorical_factors(self):
        """Test PB design with categorical factors"""
        factors = {
            "A": CategoricalFactor(levels=["low", "high"]),
            "B": CategoricalFactor(levels=["cold", "hot"]),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        uncoded = design._design_matrix
        
        # Coded matrix should still be -1/1
        assert set(coded["A"].values) == {-1, 1}
        assert set(coded["B"].values) == {-1, 1}
        
        # Uncoded matrix should have categorical values
        assert set(uncoded["A"].values) <= {"low", "high"}
        assert set(uncoded["B"].values) <= {"cold", "hot"}

    # ===============================================================
    # LARGE DESIGN TESTS
    # ===============================================================

    def test_20_factor_pb_design(self):
        """Test large PB design with 20 factors (24 runs)"""
        factors = {
            f"F{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(20)
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 20 factors, 24 runs
        assert coded.shape[0] == 24
        assert coded.shape[1] == 23  # 20 real + 3 dummy factors
        
        # All values should be -1 or 1
        assert set(coded.values.flatten()) == {-1, 1}
        
        # Last row should be all -1
        assert list(coded.iloc[-1].values) == [-1] * 23

    def test_23_factor_pb_design(self):
        """Test maximum PB design with 23 factors (24 runs)"""
        factors = {
            f"F{i}": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(23)
        }
        
        design = PlackettBurmanDesign(factors=factors)
        coded = design._coded_design_matrix
        
        # 23 factors, 24 runs
        assert coded.shape[0] == 24
        assert coded.shape[1] == 23  # Exactly 23 factors, no dummy factors needed

    # ===============================================================
    # DIMENSION CONSISTENCY TESTS
    # ===============================================================

    def test_coded_uncoded_dimension_consistency(self):
        """Test that coded and uncoded matrices have same dimensions"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors, center_points=2, replicates=1)
        
        assert design._coded_design_matrix.shape == design._design_matrix.shape
        assert list(design._coded_design_matrix.columns) == list(design._design_matrix.columns)

    # ===============================================================
    # TEST SET MODEL TERMS
    # ===============================================================
    
    def test_set_model_terms(self):
        
        """Test setting model terms in PB design"""
        factors = {
            "A": ContinuousFactor(n_levels=2, lower_bound=10, upper_bound=50, decimals=1),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15, decimals=1),
            "C": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=1),
            "D": ContinuousFactor(n_levels=2, lower_bound=2, upper_bound=8, decimals=1),
        }
        
        design = PlackettBurmanDesign(factors=factors)
        
        # Set model terms
        design.set_model_terms(intercept=True)
        
        assert design._model_spec.intercept is True
        assert design._model_spec.main == list(factors.keys())
        assert design._model_spec.interaction2 == []
        assert design._model_spec.interaction3 == []
        assert design._model_spec.quadratic == []
        assert design._model_matrix.shape[1] == len(factors) + 1  # +1 for intercept

