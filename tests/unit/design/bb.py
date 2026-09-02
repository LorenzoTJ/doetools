import itertools
from doetools import ContinuousFactor, CategoricalFactor
from doetools import BoxBehnkenDesign
import pytest
import numpy as np

class TestBoxBehnkenDesign:
    
    # Test initialization with valid factors
    def test_initialization(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0)
        }
        
        design = BoxBehnkenDesign(factors, center_points=3, replicates=0)
        
        assert design._design_type == "Box Behnken"
        assert len(design._factors) == 3
        assert design._coded_design_matrix is not None
        assert design._design_matrix is not None
    
    # Test initialization with invalid number of factors
    def test_invalid_factor_count(self):
        # Fewer than the three factors required by a Box-Behnken design
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1)
        }
        
        with pytest.raises(ValueError, match="at least 3 factors"):
            BoxBehnkenDesign(factors, center_points=3, replicates=0)
    
    # Test initialization with invalid factor levels
    def test_invalid_factor_levels(self):
        # Factor with 2 levels instead of 3
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0)
        }
        
        with pytest.raises(ValueError, match="exactly 3 levels"):
            BoxBehnkenDesign(factors, center_points=3, replicates=0)
    
    # Test initialization with categorical factors
    def test_invalid_categorical_factors(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Catalyst': CategoricalFactor(levels=['A', 'B', 'C'])
        }
        
        with pytest.raises(ValueError, match="continuous"):
            BoxBehnkenDesign(factors, center_points=3, replicates=0)
    
    def test_2_factor_design_is_rejected(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        with pytest.raises(ValueError, match="at least 3 factors"):
            BoxBehnkenDesign(factors, center_points=0, replicates=0)
    
    # Test exact design matrix for 3 factors
    def test_3_factor_bb_matrix(self):
        factors = {
            'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = BoxBehnkenDesign(factors, center_points=0, replicates=0)
        
        # For k=3, Box-Behnken has 2*3*(3-1) = 12 runs
        # Pairs: (A,B), (A,C), (B,C)
        # Each pair contributes 4 runs: (-1,-1,0), (-1,1,0), (1,-1,0), (1,1,0)
        assert design._coded_design_matrix.shape[0] == 12
        assert design._coded_design_matrix.shape[1] == 3
        
        expected = np.array([
            [-1, -1, 0],
            [-1,  1, 0],
            [ 1, -1, 0],
            [ 1,  1, 0],
            [-1, 0, -1],
            [-1, 0,  1],
            [ 1, 0, -1],
            [ 1, 0,  1],
            [0, -1, -1],
            [0, -1,  1],
            [0,  1, -1],
            [0,  1,  1]
        ])

        grid = design._coded_design_matrix.copy().to_numpy()
        
        # Check that all values are in {-1, 0, 1}
        for col in design._coded_design_matrix.columns:
            unique_vals = set(design._coded_design_matrix[col].unique())
            assert unique_vals.issubset({-1.0, 0.0, 1.0})
        
        # Check that uncoded values are correct
        assert design._design_matrix['A'].min() == 0.0
        assert design._design_matrix['A'].max() == 10.0
        assert design._design_matrix['B'].min() == 0.0
        assert design._design_matrix['B'].max() == 10.0
        assert design._design_matrix['C'].min() == 0.0
        assert design._design_matrix['C'].max() == 10.0
        
        # Function to compare two DataFrames ignoring row order
        expected = expected[np.lexsort(expected.T)]
        grid = grid[np.lexsort(grid.T)]
        assert np.array_equal(expected, grid)
    
    # Test exact design matrix for 4 factors with detailed validation
    def test_4_factor_bb_matrix(self):
        factors = {
            'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'D': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = BoxBehnkenDesign(factors, center_points=0, replicates=0)
        
        # For k=4, Box-Behnken has 2*4*(4-1) = 24 runs
        # 6 pairs: (A,B), (A,C), (A,D), (B,C), (B,D), (C,D)
        # Each pair contributes 4 runs
        assert design._coded_design_matrix.shape[0] == 24
        assert design._coded_design_matrix.shape[1] == 4
        
        coded = design._coded_design_matrix
        
        # Verify all 6 pairs exist with exactly 4 runs each
        pair_checks = [
            ('A', 'B', ['C', 'D']),
            ('A', 'C', ['B', 'D']),
            ('A', 'D', ['B', 'C']),
            ('B', 'C', ['A', 'D']),
            ('B', 'D', ['A', 'C']),
            ('C', 'D', ['A', 'B'])
        ]
        
        for factor1, factor2, zero_factors in pair_checks:
            # Find runs where the other factors are at 0
            mask = (coded[zero_factors[0]] == 0.0) & (coded[zero_factors[1]] == 0.0)
            pair_runs = coded[mask]
            
            assert len(pair_runs) == 4, f"Pair ({factor1},{factor2}) should have 4 runs, got {len(pair_runs)}"
            
            # Check that the pair has all 4 combinations: (-1,-1), (-1,1), (1,-1), (1,1)
            combinations = set()
            for _, row in pair_runs.iterrows():
                combinations.add((row[factor1], row[factor2]))
            
            expected_combinations = {(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)}
            assert combinations == expected_combinations, \
                f"Pair ({factor1},{factor2}) missing combinations: {expected_combinations - combinations}"
        
        # Check uncoded values are correct
        for factor in ['A', 'B', 'C', 'D']:
            assert design._design_matrix[factor].min() == 0.0
            assert design._design_matrix[factor].max() == 10.0
            # Center value should be 5.0
            assert 5.0 in design._design_matrix[factor].values
    
    # Test number of runs formula: 2k(k-1) + center_points
    def test_number_of_runs_formula(self):
        # Test supported factor counts.
        test_cases = [
            (3, 12),  # 2*3*(3-1) = 12
            (4, 24),  # 2*4*(4-1) = 24
            (5, 40),  # 2*5*(5-1) = 40
            (6, 60),   # 2*6*(6-1) = 60
            (7, 84)   # 2*7*(7-1) = 84
        ]
        
        for k, expected_base_runs in test_cases:
            factors = {
                f'Factor{i}': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
                for i in range(k)
            }
            
            center_points = 3
            design = BoxBehnkenDesign(factors, center_points=center_points, replicates=0)
            
            expected_total_runs = expected_base_runs + center_points
            assert design._coded_design_matrix.shape[0] == expected_total_runs, \
                f"For k={k}, expected {expected_total_runs} runs, got {design._coded_design_matrix.shape[0]}"
    
    # Test that no extreme corners exist (no runs with all factors at ±1)
    def test_no_extreme_corners(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0),
            'pH': ContinuousFactor(n_levels=3, lower_bound=5, upper_bound=9, decimals=1)
        }
        
        design = BoxBehnkenDesign(factors, center_points=0, replicates=0)
        
        # Check that no row has all factors at ±1
        coded_matrix = design._coded_design_matrix
        
        for idx, row in coded_matrix.iterrows():
            abs_values = row.abs()
            # At least one factor must be at 0 in every run
            assert (abs_values == 0.0).any(), f"Row {idx} has no zeros: {row.to_dict()}"
            
            # Check no row has all non-zero values
            num_zeros = (abs_values == 0.0).sum()
            assert num_zeros >= 1, f"Row {idx} has all factors at extreme levels"
    
    # Test center points
    def test_center_points(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0)
        }
        
        center_points = 5
        design = BoxBehnkenDesign(factors, center_points=center_points, replicates=0)
        
        # Expected runs: 12 (base) + 5 (center) = 17
        k = 3
        base_runs = 2 * k * (k - 1)
        expected_runs = base_runs + center_points
        assert design._coded_design_matrix.shape[0] == expected_runs
        
        # Check that center points are at the end and all zeros in coded matrix
        center_rows_coded = design._coded_design_matrix.tail(center_points)
        for col in center_rows_coded.columns:
            assert all(center_rows_coded[col] == 0.0)
        
        # Check that center points have midpoint values in uncoded matrix
        center_rows = design._design_matrix.tail(center_points)
        assert all(center_rows['Temperature'] == 50.0)
        assert all(center_rows['Pressure'] == 3.0)
        assert all(center_rows['Flow'] == 20.0)
    
    # Test replicates
    def test_replicates(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0)
        }
        
        replicates = 2
        design = BoxBehnkenDesign(factors, center_points=0, replicates=replicates)
        
        # For k=3, base runs = 12; every design point is repeated twice.
        base_runs = 12
        expected_runs = base_runs * (replicates + 1)
        assert design._coded_design_matrix.shape[0] == expected_runs
        
        # Check that each unique combination appears exactly (replicates + 1) times
        unique_combos = design._coded_design_matrix.drop_duplicates()
        assert len(unique_combos) == base_runs
        
        for _, row in unique_combos.iterrows():
            matches = (
                (design._coded_design_matrix['Temperature'] == row['Temperature'])
                & (design._coded_design_matrix['Pressure'] == row['Pressure'])
                & (design._coded_design_matrix['Flow'] == row['Flow'])
            )
            assert matches.sum() == (replicates + 1)
    
    def test_all_combinations_5_factors(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'D': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'E': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
    
        design = BoxBehnkenDesign(factors, center_points=0, replicates=0)
        
        # For k=5, Box-Behnken has 2*5*(5-1) = 40 runs
        assert design._coded_design_matrix.shape[0] == 40
        assert design._coded_design_matrix.shape[1] == 5
        
        combinations = itertools.combinations(range(5), 2)
        for combo in combinations:
            factor1, factor2 = combo
            factor_names = list(factors.keys())
            f1 = factor_names[factor1]
            f2 = factor_names[factor2]
            zero_factors = [fn for fn in factor_names if fn not in (f1, f2)]
            
            # Find runs where the other factors are at 0
            mask = np.ones(len(design._coded_design_matrix), dtype=bool)
            for zf in zero_factors:
                mask &= (design._coded_design_matrix[zf] == 0.0)
            pair_runs = design._coded_design_matrix[mask]
            
            assert len(pair_runs) == 4, f"Pair ({f1},{f2}) should have 4 runs, got {len(pair_runs)}"
            
            # Check that the pair has all 4 combinations: (-1,-1), (-1,1), (1,-1), (1,1)
            found_combinations = set()
            for _, row in pair_runs.iterrows():
                found_combinations.add((row[f1], row[f2]))
            
            expected_combinations = {(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)}
            assert found_combinations == expected_combinations, \
                f"Pair ({f1},{f2}) missing combinations: {expected_combinations - found_combinations}"
    
    def test_all_combinations_6_factors(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'D': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'E': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'F': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
    
        design = BoxBehnkenDesign(factors, center_points=0, replicates=0)
        
        # For k=6, Box-Behnken has 2*6*(6-1) = 60 runs
        assert design._coded_design_matrix.shape[0] == 60
        assert design._coded_design_matrix.shape[1] == 6
        
        combinations = itertools.combinations(range(6), 2)
        for combo in combinations:
            factor1, factor2 = combo
            factor_names = list(factors.keys())
            f1 = factor_names[factor1]
            f2 = factor_names[factor2]
            zero_factors = [fn for fn in factor_names if fn not in (f1, f2)]
            
            # Find runs where the other factors are at 0
            mask = np.ones(len(design._coded_design_matrix), dtype=bool)
            for zf in zero_factors:
                mask &= (design._coded_design_matrix[zf] == 0.0)
            pair_runs = design._coded_design_matrix[mask]
            
            assert len(pair_runs) == 4, f"Pair ({f1},{f2}) should have 4 runs, got {len(pair_runs)}"
            
            # Check that the pair has all 4 combinations: (-1,-1), (-1,1), (1,-1), (1,1)
            found_combinations = set()
            for _, row in pair_runs.iterrows():
                found_combinations.add((row[f1], row[f2]))
            
            expected_combinations = {(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)}
            assert found_combinations == expected_combinations, \
                f"Pair ({f1},{f2}) missing combinations: {expected_combinations - found_combinations}"
        
        
    # Test center points and replicates together
    def test_center_points_and_replicates(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0)
        }
        
        center_points = 3
        replicates = 1
        design = BoxBehnkenDesign(factors, center_points=center_points, replicates=replicates)
        
        # Expected runs: 12 * (1+1) + 3 = 27
        k = 3
        base_runs = 2 * k * (k - 1)  # 12
        expected_runs = base_runs * (replicates + 1) + center_points
        assert design._coded_design_matrix.shape[0] == expected_runs
        
        # Check center points are at the end
        center_rows = design._design_matrix.tail(center_points)
        assert all(center_rows['Temperature'] == 50.0)
        assert all(center_rows['Pressure'] == 3.0)
        assert all(center_rows['Flow'] == 20.0)
    
    # Test 4-factor design dimensions
    def test_4_factor_dimensions(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0),
            'pH': ContinuousFactor(n_levels=3, lower_bound=5, upper_bound=9, decimals=1)
        }
        
        design = BoxBehnkenDesign(factors, center_points=3, replicates=0)
        
        # Expected runs: 2*4*(4-1) + 3 = 24 + 3 = 27
        k = 4
        base_runs = 2 * k * (k - 1)
        center_points = 3
        expected_runs = base_runs + center_points
        
        assert design._coded_design_matrix.shape[0] == expected_runs
        assert design._coded_design_matrix.shape[1] == k
    
    # Test 5-factor design dimensions
    def test_5_factor_dimensions(self):
        factors = {
            'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'D': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'E': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = BoxBehnkenDesign(factors, center_points=5, replicates=0)
        
        # Expected runs: 2*5*(5-1) + 5 = 40 + 5 = 45
        k = 5
        base_runs = 2 * k * (k - 1)
        center_points = 5
        expected_runs = base_runs + center_points
        
        assert design._coded_design_matrix.shape[0] == expected_runs
        assert design._coded_design_matrix.shape[1] == k
    
    # Test value ranges for continuous factors
    def test_continuous_factor_ranges(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=100, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=0.5, upper_bound=2.5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=40, decimals=0)
        }
        
        design = BoxBehnkenDesign(factors, center_points=0, replicates=0)
        
        # Check Temperature range
        assert design._design_matrix['Temperature'].min() == 20.0
        assert design._design_matrix['Temperature'].max() == 100.0
        
        # Check Pressure range
        assert design._design_matrix['Pressure'].min() == 0.5
        assert design._design_matrix['Pressure'].max() == 2.5
        
        # Check Flow range
        assert design._design_matrix['Flow'].min() == 10.0
        assert design._design_matrix['Flow'].max() == 40.0
        
        # Check coded matrix values are in {-1, 0, 1}
        for col in design._coded_design_matrix.columns:
            unique_vals = set(design._coded_design_matrix[col].unique())
            assert unique_vals.issubset({-1.0, 0.0, 1.0})
    
    # Test that each factor appears at all three levels
    def test_all_factors_at_all_levels(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0)
        }
        
        design = BoxBehnkenDesign(factors, center_points=3, replicates=0)
        
        # Check that each factor has runs at -1, 0, and 1 in coded matrix
        for col in design._coded_design_matrix.columns:
            unique_vals = sorted(design._coded_design_matrix[col].unique())
            assert -1.0 in unique_vals, f"Factor {col} missing low level (-1)"
            assert 0.0 in unique_vals, f"Factor {col} missing center level (0)"
            assert 1.0 in unique_vals, f"Factor {col} missing high level (1)"
    
    # Test exactly two factors vary in each run (excluding center points)
    def test_two_factors_vary_per_run(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5, decimals=2),
            'Flow': ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=30, decimals=0)
        }
        
        design = BoxBehnkenDesign(factors, center_points=3, replicates=0)
        
        # Remove center points (last 3 rows)
        non_center_runs = design._coded_design_matrix.iloc[:-3]
        
        # Each non-center run should have exactly 1 zero and 2 non-zeros
        for idx, row in non_center_runs.iterrows():
            num_zeros = (row == 0.0).sum()
            num_nonzeros = (row != 0.0).sum()
            assert num_zeros == 1, f"Row {idx} has {num_zeros} zeros, expected 1"
            assert num_nonzeros == 2, f"Row {idx} has {num_nonzeros} non-zeros, expected 2"
    
    # Test proper pairing of factors
    def test_factor_pairings(self):
        factors = {
            'A': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = BoxBehnkenDesign(factors, center_points=0, replicates=0)
        
        # For k=3, we should have 3 pairs: (A,B), (A,C), (B,C)
        # Each pair should have 4 combinations: (-1,-1), (-1,1), (1,-1), (1,1)
        coded = design._coded_design_matrix
        
        # Count occurrences of each pair combination
        # Pair (A,B) with C=0
        ab_runs = coded[coded['C'] == 0.0]
        assert len(ab_runs) == 4
        
        # Pair (A,C) with B=0
        ac_runs = coded[coded['B'] == 0.0]
        assert len(ac_runs) == 4
        
        # Pair (B,C) with A=0
        bc_runs = coded[coded['A'] == 0.0]
        assert len(bc_runs) == 4
    
