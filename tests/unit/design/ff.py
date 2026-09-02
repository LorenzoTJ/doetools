from doetools import ContinuousFactor, CategoricalFactor
from doetools import FullFactorialDesign
from doetools.utils.model_spec import ModelTerms
import pytest
import pandas as pd

class TestFullFactorialDesign:
    
    # Test initialization with valid factors
    def test_initialization(self):
        
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        design = FullFactorialDesign(factors, center_points=1, replicates=0)
        
        assert design._design_type == "Full Factorial"
        assert len(design._factors) == 2
        assert design._coded_design_matrix is not None
        assert design._design_matrix is not None
    
    # Test initialization with invalid center points and replicates
    def test_invalid_initialization(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        with pytest.raises(ValueError):
            FullFactorialDesign(factors, center_points=-1, replicates=0)
        
        with pytest.raises(ValueError):
            FullFactorialDesign(factors, center_points=1, replicates=-2)
    
    # Test design matrix generation
    # 2 factors 3 levels 
    def test_2f3l_ff_matrix(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        expected_matrix = pd.DataFrame({
            'Temperature': [20.0, 20.0, 20.0, 50.0, 50.0, 50.0, 80.0, 80.0, 80.0],
            'Pressure': [1.00, 3.00, 5.00, 1.00, 3.00, 5.00, 1.00, 3.00, 5.00]
        })
        
        expected_coded_matrix = pd.DataFrame({
            'Temperature': [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            'Pressure': [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        })
        
        pd.testing.assert_frame_equal(design._design_matrix.reset_index(drop=True), expected_matrix)
        pd.testing.assert_frame_equal(design._coded_design_matrix.reset_index(drop=True), expected_coded_matrix)
    
    # 3 factors 2 levels
    def test_3f2l_ff_matrix(self):
        factors = {
            'Catalyst': CategoricalFactor(levels=['A', 'B']),
            'Solvent': CategoricalFactor(levels=['X', 'Y']),
            'Additive': CategoricalFactor(levels=['M', 'N'])
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        expected_matrix = pd.DataFrame({
            'Catalyst': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'Solvent': ['X', 'X', 'Y', 'Y', 'X', 'X', 'Y', 'Y'],
            'Additive': ['M', 'N', 'M', 'N', 'M', 'N', 'M', 'N']
        })
        expected_coded_matrix = pd.DataFrame({
            'Catalyst': [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            'Solvent': [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            'Additive': [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        })
        
        pd.testing.assert_frame_equal(design._design_matrix.reset_index(drop=True), expected_matrix)
        pd.testing.assert_frame_equal(design._coded_design_matrix.reset_index(drop=True), expected_coded_matrix)

    def test_categorical_reference_does_not_change_factorial_matrix(self):
        """The reference only affects operational defaults, not factor coding."""
        default_design = FullFactorialDesign(
            {"Catalyst": CategoricalFactor(levels=["A", "B", "C"])},
            center_points=0,
        )
        referenced_design = FullFactorialDesign(
            {
                "Catalyst": CategoricalFactor(
                    levels=["A", "B", "C"], reference_level="C"
                )
            },
            center_points=0,
        )
        terms = ModelTerms(intercept=True, pro_main="all")
        default_design.set_model_terms(terms)
        referenced_design.set_model_terms(terms)

        pd.testing.assert_frame_equal(
            referenced_design._coded_design_matrix,
            default_design._coded_design_matrix,
        )
        pd.testing.assert_frame_equal(
            referenced_design._design_matrix,
            default_design._design_matrix,
        )
        pd.testing.assert_frame_equal(
            referenced_design._model_matrix,
            default_design._model_matrix,
        )
        
    # 3 factors 3 levels -> Mixed test
    def test_3f3l_ff_matrix(self):
        factors = {
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=3, decimals=1),
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Additive': CategoricalFactor(levels=['M', 'N', 'O'])
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        expected_matrix = pd.DataFrame({
            'Pressure': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            'Temperature': [20.0, 20.0, 20.0, 50.0, 50.0, 50.0, 80.0, 80.0, 80.0, 20.0, 20.0, 20.0, 50.0, 50.0, 50.0, 80.0, 80.0, 80.0, 20.0, 20.0, 20.0, 50.0, 50.0, 50.0, 80.0, 80.0, 80.0],
            'Additive': ['M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O']
        })
        expected_coded_matrix = pd.DataFrame({
            'Pressure': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'Temperature': [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            'Additive': [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        })
        
        pd.testing.assert_frame_equal(design._design_matrix.reset_index(drop=True), expected_matrix)
        pd.testing.assert_frame_equal(design._coded_design_matrix.reset_index(drop=True), expected_coded_matrix)
    
    # Test complex design with 5 factors and mixed types
    def test_5_factor_dimensions(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2),
            'Catalyst': CategoricalFactor(levels=['A', 'B', 'C']),
            'Time': ContinuousFactor(n_levels=4, lower_bound=10, upper_bound=40, decimals=0),
            'pH': ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=9, decimals=1)
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        # Expected number of runs: 3 * 2 * 3 * 4 * 2 = 144
        expected_runs = 3 * 2 * 3 * 4 * 2
        
        assert design._design_matrix.shape[0] == expected_runs
        assert design._coded_design_matrix.shape[0] == expected_runs
        assert design._design_matrix.shape[1] == 5
        assert design._coded_design_matrix.shape[1] == 5
    
    # Test value ranges for continuous factors
    def test_continuous_factor_ranges(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=5, lower_bound=20, upper_bound=100, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=0.5, upper_bound=2.5, decimals=2),
            'Flow': ContinuousFactor(n_levels=4, lower_bound=10, upper_bound=40, decimals=0)
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        # Check Temperature range
        assert design._design_matrix['Temperature'].min() == 20.0
        assert design._design_matrix['Temperature'].max() == 100.0
        
        # Check Pressure range
        assert design._design_matrix['Pressure'].min() == 0.5
        assert design._design_matrix['Pressure'].max() == 2.5
        
        # Check Flow range
        assert design._design_matrix['Flow'].min() == 10.0
        assert design._design_matrix['Flow'].max() == 40.0
        
        # Check coded matrix range (should be -1 to 1)
        assert design._coded_design_matrix['Temperature'].min() == -1.0
        assert design._coded_design_matrix['Temperature'].max() == 1.0
        assert design._coded_design_matrix['Pressure'].min() == -1.0
        assert design._coded_design_matrix['Pressure'].max() == 1.0
    
    # Test center points addition
    def test_center_points(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        center_points = 5
        design = FullFactorialDesign(factors, center_points=center_points, replicates=0)
        
        # Expected runs: 3 * 3 = 9 factorial points + 5 center points = 14
        expected_runs = 9 + center_points
        assert design._design_matrix.shape[0] == expected_runs
        
        # Check that center points have correct values (midpoint of range)
        center_rows = design._design_matrix.tail(center_points)
        assert all(center_rows['Temperature'] == 50.0)
        assert all(center_rows['Pressure'] == 3.0)
        
        # Check coded center points are all zeros
        coded_center_rows = design._coded_design_matrix.tail(center_points)
        assert all(coded_center_rows['Temperature'] == 0.0)
        assert all(coded_center_rows['Pressure'] == 0.0)
    
    # Test replicates
    def test_replicates(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        replicates = 3
        design = FullFactorialDesign(factors, center_points=0, replicates=replicates)
        
        # Expected runs: (2 * 2) * (3 replicates + 1 original) = 16
        base_runs = 2 * 2
        expected_runs = base_runs * (replicates + 1)
        assert design._design_matrix.shape[0] == expected_runs
        
        # Check that each unique combination appears exactly (replicates + 1) times
        # There should be 4 unique combinations: (20,1), (20,5), (80,1), (80,5)
        unique_combos = design._design_matrix.drop_duplicates()
        assert len(unique_combos) == base_runs
        
        # Check that each unique combination appears exactly 4 times
        for _, row in unique_combos.iterrows():
            matches = ((design._design_matrix['Temperature'] == row['Temperature']) & 
                      (design._design_matrix['Pressure'] == row['Pressure']))
            assert matches.sum() == (replicates + 1)
    
    # Test center points and replicates together
    def test_center_points_and_replicates(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        center_points = 3
        replicates = 2
        design = FullFactorialDesign(factors, center_points=center_points, replicates=replicates)
        
        # Expected runs: (2 * 2) * (2 replicates + 1 original) + 3 center points = 15
        base_runs = 2 * 2
        expected_runs = base_runs * (replicates + 1) + center_points
        assert design._design_matrix.shape[0] == expected_runs
        
        # Check center points are at the end
        center_rows = design._design_matrix.tail(center_points)
        assert all(center_rows['Temperature'] == 50.0)
        assert all(center_rows['Pressure'] == 3.0)
    
    # Test large design with 6 factors
    def test_6_factor_large_design(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'D': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'E': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'F': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        # Expected runs: 2^6 = 64
        expected_runs = 2 ** 6
        assert design._design_matrix.shape[0] == expected_runs
        assert design._design_matrix.shape[1] == 6
        
        # Check that all factors have exactly 2 unique values (0 and 10)
        for factor_name in factors.keys():
            unique_values = design._design_matrix[factor_name].unique()
            assert len(unique_values) == 2
            assert 0.0 in unique_values
            assert 10.0 in unique_values
    
    # Test categorical factor levels
    def test_categorical_factor_levels(self):
        factors = {
            'Catalyst': CategoricalFactor(levels=['A', 'B', 'C', 'D']),
            'Solvent': CategoricalFactor(levels=['Water', 'Ethanol', 'Acetone']),
            'Method': CategoricalFactor(levels=['Batch', 'Continuous'])
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        # Expected runs: 4 * 3 * 2 = 24
        expected_runs = 4 * 3 * 2
        assert design._design_matrix.shape[0] == expected_runs
        
        # Check that all specified levels appear in the design
        assert set(design._design_matrix['Catalyst'].unique()) == {'A', 'B', 'C', 'D'}
        assert set(design._design_matrix['Solvent'].unique()) == {'Water', 'Ethanol', 'Acetone'}
        assert set(design._design_matrix['Method'].unique()) == {'Batch', 'Continuous'}
        
        # Check coded values for categorical factors
        # Should be evenly spaced from -1 to 1
        assert design._coded_design_matrix['Catalyst'].min() == -1.0
        assert design._coded_design_matrix['Catalyst'].max() == 1.0
        assert design._coded_design_matrix['Method'].min() == -1.0
        assert design._coded_design_matrix['Method'].max() == 1.0
    
    # Test mixed design with different level counts
    def test_mixed_level_design(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=5, lower_bound=20, upper_bound=100, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=3, decimals=1),
            'Catalyst': CategoricalFactor(levels=['A', 'B']),
            'Additive': CategoricalFactor(levels=['X', 'Y', 'Z', 'W'])
        }
        
        design = FullFactorialDesign(factors, center_points=0, replicates=0)
        
        # Expected runs: 5 * 3 * 2 * 4 = 120
        expected_runs = 5 * 3 * 2 * 4
        assert design._design_matrix.shape[0] == expected_runs
        assert design._design_matrix.shape[1] == 4
        
        # Check number of unique levels per factor
        assert len(design._design_matrix['Temperature'].unique()) == 5
        assert len(design._design_matrix['Pressure'].unique()) == 3
        assert len(design._design_matrix['Catalyst'].unique()) == 2
        assert len(design._design_matrix['Additive'].unique()) == 4
        
