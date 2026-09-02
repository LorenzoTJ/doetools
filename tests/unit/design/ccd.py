import pandas as pd
from doetools import ContinuousFactor, CategoricalFactor
from doetools import CentralCompositeDesign
import pytest
import numpy as np
from pandas.testing import assert_frame_equal


class TestCentralCompositeDesign:
    
    # Test initialization with valid factors for each design type
    def test_initialization_ccc(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        design = CentralCompositeDesign(factors, design='ccc', center_points=1, replicates=0)
        
        assert design._design_type == "Central Composite"
        assert design._type == "ccc"
        assert len(design._factors) == 2
        assert design._coded_design_matrix is not None
        assert design._design_matrix is not None
    
    def test_initialization_ccf(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        design = CentralCompositeDesign(factors, design='ccf', center_points=1, replicates=0)
        
        assert design._design_type == "Central Composite"
        assert design._type == "ccf"
        assert len(design._factors) == 2
        assert design._coded_design_matrix is not None
        assert design._design_matrix is not None
    
    def test_initialization_cci(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        design = CentralCompositeDesign(factors, design='cci', center_points=1, replicates=0)
        
        assert design._design_type == "Central Composite"
        assert design._type == "cci"
        assert len(design._factors) == 2
        assert design._coded_design_matrix is not None
        assert design._design_matrix is not None
    
    # Test invalid design type
    def test_invalid_design_type(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        with pytest.raises(ValueError, match="ccc, ccf, or cci"):
            CentralCompositeDesign(factors, design='invalid', center_points=1, replicates=0)
    
    # Test invalid factor count
    def test_invalid_factor_count(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1)
        }
        
        with pytest.raises(ValueError, match="at least 2 factors"):
            CentralCompositeDesign(factors, design='ccc', center_points=1, replicates=0)
    
    # Test invalid center points
    def test_invalid_center_points(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            CentralCompositeDesign(factors, design='ccc', center_points=-1, replicates=0)
    
    # Test categorical factors not allowed
    def test_invalid_categorical_factors(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Catalyst': CategoricalFactor(levels=['A', 'B'])
        }
        
        with pytest.raises(ValueError, match="continuous"):
            CentralCompositeDesign(factors, design='ccc', center_points=1, replicates=0)
    
    # Test number of runs formula: 2^k + 2k + center_points
    def test_number_of_runs_formula(self):
        test_cases = [
            (2, 9),   # 2^2 + 2*2 + 1 = 4 + 4 + 1 = 9
            (3, 15),  # 2^3 + 2*3 + 1 = 8 + 6 + 1 = 15
            (4, 25),  # 2^4 + 2*4 + 1 = 16 + 8 + 1 = 25
            (5, 43),  # 2^5 + 2*5 + 1 = 32 + 10 + 1 = 43
        ]
        
        for k, expected_runs in test_cases:
            factors = {
                f'Factor{i}': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
                for i in range(k)
            }
            
            center_points = 1
            design = CentralCompositeDesign(factors, design='ccc', center_points=center_points, replicates=0)
            
            assert design._coded_design_matrix.shape[0] == expected_runs, \
                f"For k={k}, expected {expected_runs} runs, got {design._coded_design_matrix.shape[0]}"
    
    # Test CCC alpha value (rotatable)
    def test_ccc_alpha_value_3f(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        k = 3
        design = CentralCompositeDesign(factors, design='ccc', center_points=0, replicates=0)
        
        # For CCC, alpha = (2^k)^(1/4)
        expected_alpha = (2**k)**(1/4)
        
        # Check that axial points exist at ±alpha
        coded = design._coded_design_matrix
        
        # Find rows where only one factor is non-zero (axial points)
        for factor in ['A', 'B', 'C']:
            other_factors = [f for f in ['A', 'B', 'C'] if f != factor]
            axial_mask = (coded[other_factors[0]] == 0) & (coded[other_factors[1]] == 0) & (coded[factor] != 0)
            axial_values = coded.loc[axial_mask, factor].abs().unique()
            
            assert len(axial_values) == 1, f"Factor {factor} should have axial points at one distance"
            assert np.isclose(axial_values[0], expected_alpha, atol=1e-4), \
                f"Factor {factor} axial distance {axial_values[0]:.6f} != expected {expected_alpha:.6f}"
    
    # Test CCC alpha value (rotatable)
    def test_ccc_alpha_value_4f(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'D': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        k = 4
        design = CentralCompositeDesign(factors, design='ccc', center_points=0, replicates=0)
        
        # For CCC, alpha = (2^k)^(1/4)
        expected_alpha = (2**k)**(1/4)
        
        # Check that axial points exist at ±alpha
        coded = design._coded_design_matrix
        
        # Find rows where only one factor is non-zero (axial points)
        for factor in ['A', 'B', 'C', 'D']:
            other_factors = [f for f in ['A', 'B', 'C', 'D'] if f != factor]
            axial_mask = (coded[other_factors[0]] == 0) & (coded[other_factors[1]] == 0) & (coded[other_factors[2]] == 0) & (coded[factor] != 0)
            axial_values = coded.loc[axial_mask, factor].abs().unique()
            
            assert len(axial_values) == 1, f"Factor {factor} should have axial points at one distance"
            assert np.isclose(axial_values[0], expected_alpha, atol=1e-4), \
                f"Factor {factor} axial distance {axial_values[0]:.6f} != expected {expected_alpha:.6f}"
    
    # Test CCC alpha value (rotatable)
    def test_ccc_alpha_value_5f(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'D': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'E': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        k = 5
        design = CentralCompositeDesign(factors, design='ccc', center_points=0, replicates=0)
        
        # For CCC, alpha = (2^k)^(1/4)
        expected_alpha = (2**k)**(1/4)
        
        # Check that axial points exist at ±alpha
        coded = design._coded_design_matrix
        
        # Find rows where only one factor is non-zero (axial points)
        for factor in ['A', 'B', 'C', 'D', 'E']:
            other_factors = [f for f in ['A', 'B', 'C', 'D', 'E'] if f != factor]
            axial_mask = (coded[other_factors[0]] == 0) & (coded[other_factors[1]] == 0) & (coded[other_factors[2]] == 0) & (coded[other_factors[3]] == 0) & (coded[factor] != 0)
            axial_values = coded.loc[axial_mask, factor].abs().unique()
            
            assert len(axial_values) == 1, f"Factor {factor} should have axial points at one distance"
            assert np.isclose(axial_values[0], expected_alpha, atol=1e-4), \
                f"Factor {factor} axial distance {axial_values[0]:.6f} != expected {expected_alpha:.6f}"
    
    # Test CCF alpha value (face-centered)
    def test_ccf_alpha_value(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = CentralCompositeDesign(factors, design='ccf', center_points=0, replicates=0)
        
        coded = design._coded_design_matrix
        
        # All coded values should be in {-1, 0, 1}
        for col in coded.columns:
            unique_vals = set(coded[col].unique())
            assert unique_vals.issubset({-1.0, 0.0, 1.0}), \
                f"CCF should only have values in {{-1, 0, 1}}, got {unique_vals} for {col}"
    
    # Test CCI alpha value (inscribed)
    def test_cci_alpha_value_3f(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        k = 3
        design = CentralCompositeDesign(factors, design='cci', center_points=0, replicates=0)
        
        # For CCI, alpha = (2^k)^(1/4), factorial points at ±1/alpha, axial at ±1
        expected_alpha = (2**k)**(1/4)
        expected_factorial_distance = 1 / expected_alpha
        
        coded = design._coded_design_matrix
        
        # Find factorial points (all factors non-zero)
        factorial_mask = (coded['A'] != 0) & (coded['B'] != 0) & (coded['C'] != 0)
        factorial_values = coded.loc[factorial_mask].abs()
        
        # Check that factorial points are at ±1/alpha
        for col in ['A', 'B', 'C']:
            unique_factorial = factorial_values[col].unique()
            assert len(unique_factorial) == 1
            assert np.isclose(unique_factorial[0], expected_factorial_distance, atol=1e-4)
        
        # Check that axial points are at ±1
        for factor in ['A', 'B', 'C']:
            other_factors = [f for f in ['A', 'B', 'C'] if f != factor]
            axial_mask = (coded[other_factors[0]] == 0) & (coded[other_factors[1]] == 0) & (coded[factor] != 0)
            axial_values = coded.loc[axial_mask, factor].abs().unique()
            
            assert len(axial_values) == 1
            assert np.isclose(axial_values[0], 1.0, atol=1e-6)
    
    # Test CCI alpha value (inscribed)
    def test_cci_alpha_value_4f(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'D': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        k = 4
        design = CentralCompositeDesign(factors, design='cci', center_points=0, replicates=0)
        
        # For CCI, alpha = (2^k)^(1/4), factorial points at ±1/alpha, axial at ±1
        expected_alpha = (2**k)**(1/4)
        expected_factorial_distance = 1 / expected_alpha
        
        coded = design._coded_design_matrix
        
        # Find factorial points (all factors non-zero)
        factorial_mask = (coded['A'] != 0) & (coded['B'] != 0) & (coded['C'] != 0) & (coded['D'] != 0)
        factorial_values = coded.loc[factorial_mask].abs()
        
        # Check that factorial points are at ±1/alpha
        for col in ['A', 'B', 'C', 'D']:
            unique_factorial = factorial_values[col].unique()
            assert len(unique_factorial) == 1
            assert np.isclose(unique_factorial[0], expected_factorial_distance, atol=1e-4)
        
        # Check that axial points are at ±1
        for factor in ['A', 'B', 'C', 'D']:
            other_factors = [f for f in ['A', 'B', 'C', 'D'] if f != factor]
            axial_mask = (coded[other_factors[0]] == 0) & (coded[other_factors[1]] == 0) & (coded[other_factors[2]] == 0) & (coded[factor] != 0)
            axial_values = coded.loc[axial_mask, factor].abs().unique()
            
            assert len(axial_values) == 1
            assert np.isclose(axial_values[0], 1.0, atol=1e-6)
        
    # Test CCI alpha value (inscribed)
    def test_cci_alpha_value_5f(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'D': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'E': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        k = 5
        design = CentralCompositeDesign(factors, design='cci', center_points=0, replicates=0)
        
        # For CCI, alpha = (2^k)^(1/4), factorial points at ±1/alpha, axial at ±1
        expected_alpha = (2**k)**(1/4)
        expected_factorial_distance = 1 / expected_alpha
        
        coded = design._coded_design_matrix
        
        # Find factorial points (all factors non-zero)
        factorial_mask = (coded['A'] != 0) & (coded['B'] != 0) & (coded['C'] != 0) & (coded['D'] != 0) & (coded['E'] != 0)
        factorial_values = coded.loc[factorial_mask].abs()
        
        # Check that factorial points are at ±1/alpha
        for col in ['A', 'B', 'C', 'D', 'E']:
            unique_factorial = factorial_values[col].unique()
            assert len(unique_factorial) == 1
            assert np.isclose(unique_factorial[0], expected_factorial_distance, atol=1e-4)
        
        # Check that axial points are at ±1
        for factor in ['A', 'B', 'C', 'D', 'E']:
            other_factors = [f for f in ['A', 'B', 'C', 'D', 'E'] if f != factor]
            axial_mask = (coded[other_factors[0]] == 0) & (coded[other_factors[1]] == 0) & (coded[other_factors[2]] == 0) & (coded[other_factors[3]] == 0) & (coded[factor] != 0)
            axial_values = coded.loc[axial_mask, factor].abs().unique()
            
            assert len(axial_values) == 1
            assert np.isclose(axial_values[0], 1.0, atol=1e-6)
            
    # Test 2-factor CCF exact matrix
    def test_2_factor_ccf_matrix(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = CentralCompositeDesign(factors, design='ccf', center_points=0, replicates=0)
        
        # Expected: 2^2 + 2*2 = 8 runs
        # 4 factorial: (-1,-1), (-1,1), (1,-1), (1,1)
        # 4 axial: (0,1), (0,-1), (1,0), (-1,0)
        assert design._coded_design_matrix.shape[0] == 8
                
        coded = design._coded_design_matrix
        
        expected_matrix = np.array([
            [-1.0, -1.0],
            [-1.0,  1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [ 0.0,  1.0],
            [ 0.0, -1.0],
            [ 1.0,  0.0],
            [-1.0,  0.0]
        ])
        
        expected_matrix = pd.DataFrame(expected_matrix, columns=['A', 'B'])
                
        assert_frame_equal(coded.sort_values(by=['A', 'B']).reset_index(drop=True),
                          expected_matrix.sort_values(by=['A', 'B']).reset_index(drop=True),
                          atol=1e-6)
    
    # Test 3-factor CCF exact matrix
    def test_3_factor_ccf_matrix(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        design = CentralCompositeDesign(factors, design='ccf', center_points=0, replicates=0)
        
        # Expected: 2^3 + 2*3 = 14 runs
        assert design._coded_design_matrix.shape[0] == 14
        
        coded = design._coded_design_matrix
        
        expected_matrix = np.array([
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0,  1.0,  1.0],
            [ 0.0,  0.0, -1.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 1.0,  0.0,  0.0]
        ])
        
        expected_matrix = pd.DataFrame(expected_matrix, columns=['A', 'B', 'C'])
                
        assert_frame_equal(coded.sort_values(by=['A', 'B', 'C']).reset_index(drop=True),
                          expected_matrix.sort_values(by=['A', 'B', 'C']).reset_index(drop=True),
                          atol=1e-6)
    
    # Test 2-factor CCI exact matrix
    def test_2_factor_cci_matrix(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        design = CentralCompositeDesign(factors, design='cci', center_points=0, replicates=0)
        
        # Expected: 2^2 + 2*2 = 8 runs
        # 4 factorial: (-1/α,-1/α), (-1/α,1/α), (1/α,-1/α), (1/α,1/α) where α = (2^2)^(1/4) = √2
        # 4 axial: (-1,0), (1,0), (0,-1), (0,1)
        assert design._coded_design_matrix.shape[0] == 8
                
        coded = design._coded_design_matrix
        
        alpha = (2**2)**(1/4)
        inv_alpha = 1 / alpha
        
        expected_matrix = np.array([
            [-inv_alpha, -inv_alpha],
            [-inv_alpha,  inv_alpha],
            [ inv_alpha, -inv_alpha],
            [ inv_alpha,  inv_alpha],
            [-1.0,       0.0      ],
            [ 1.0,       0.0      ],
            [ 0.0,      -1.0      ],
            [ 0.0,       1.0      ]
        ])
        
        expected_matrix = pd.DataFrame(expected_matrix, columns=['A', 'B'])
                
        assert_frame_equal(coded.sort_values(by=['A', 'B']).reset_index(drop=True),
                          expected_matrix.sort_values(by=['A', 'B']).reset_index(drop=True),
                          atol=1e-6)
        
    # Test 3-factor CCI exact matrix
    def test_3_factor_cci_matrix(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        design = CentralCompositeDesign(factors, design='cci', center_points=0, replicates=0)
        
        # Expected: 2^3 + 2*3 = 14 runs
        # 8 factorial: (±1/α, ±1/α, ±1/α) where α = (2^3)^(1/4)
        # 6 axial: (±1,0,0), (0,±1,0), (0,0,±1)
        assert design._coded_design_matrix.shape[0] == 14
                
        coded = design._coded_design_matrix
        
        k = 3
        alpha = (2**k)**(1/4)
        inv_alpha = 1 / alpha
        
        expected_matrix = np.array([
            [-inv_alpha, -inv_alpha, -inv_alpha],
            [-inv_alpha, -inv_alpha,  inv_alpha],
            [-inv_alpha,  inv_alpha, -inv_alpha],
            [-inv_alpha,  inv_alpha,  inv_alpha],
            [ inv_alpha, -inv_alpha, -inv_alpha],
            [ inv_alpha, -inv_alpha,  inv_alpha],
            [ inv_alpha,  inv_alpha, -inv_alpha],
            [ inv_alpha,  inv_alpha,  inv_alpha],
            [-1.0,       0.0,       0.0      ],
            [ 1.0,       0.0,       0.0      ],
            [ 0.0,      -1.0,       0.0      ],
            [ 0.0,       1.0,       0.0      ],
            [ 0.0,       0.0,      -1.0      ],
            [ 0.0,       0.0,       1.0      ]
        ])
        
        expected_matrix = pd.DataFrame(expected_matrix, columns=['A', 'B', 'C'])
                
        assert_frame_equal(coded.sort_values(by=['A', 'B', 'C']).reset_index(drop=True),
                            expected_matrix.sort_values(by=['A', 'B', 'C']).reset_index(drop=True),
                            atol=1e-6)
    
    # Test 2-factor CCC exact matrix
    def test_2_factor_ccc_matrix(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        design = CentralCompositeDesign(factors, design='ccc', center_points=0, replicates=0)
        
        # Expected: 2^2 + 2*2 = 8 runs
        # 4 factorial: (-1,-1), (-1,1), (1,-1), (1,1)
        # 4 axial: (±α,0), (0,±α) where α = (2^2)^(1/4) = √2
        assert design._coded_design_matrix.shape[0] == 8
                
        coded = design._coded_design_matrix
        
        alpha = (2**2)**(1/4)
        
        expected_matrix = np.array([
            [-1.0,      -1.0      ],
            [-1.0,       1.0      ],
            [ 1.0,      -1.0      ],
            [ 1.0,       1.0      ],
            [ alpha,     0.0      ],
            [-alpha,     0.0      ],
            [ 0.0,      alpha     ],
            [ 0.0,     -alpha     ]
        ])
        
        expected_matrix = pd.DataFrame(expected_matrix, columns=['A', 'B'])
                
        assert_frame_equal(coded.sort_values(by=['A', 'B']).reset_index(drop=True),
                          expected_matrix.sort_values(by=['A', 'B']).reset_index(drop=True),
                          atol=1e-6)
        
    # Test 3-factor CCC exact matrix
    def test_3_factor_ccc_matrix(self):
        
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        design = CentralCompositeDesign(factors, design='ccc', center_points=0, replicates=0)
        
        # Expected: 2^3 + 2*3 = 14 runs
        # 8 factorial: (±1, ±1, ±1)
        # 6 axial: (±α,0,0), (0,±α,0), (0,0,±α) where α = (2^3)^(1/4)
        assert design._coded_design_matrix.shape[0] == 14
                
        coded = design._coded_design_matrix
        
        k = 3
        alpha = (2**k)**(1/4)
        
        expected_matrix = np.array([
            [-1.0,      -1.0,      -1.0     ],
            [-1.0,      -1.0,       1.0     ],
            [-1.0,       1.0,      -1.0     ],
            [-1.0,       1.0,       1.0     ],
            [ 1.0,      -1.0,      -1.0     ],
            [ 1.0,      -1.0,       1.0     ],
            [ 1.0,       1.0,      -1.0     ],
            [ 1.0,       1.0,       1.0     ],
            [ alpha,     0.0,       0.0     ],
            [-alpha,     0.0,       0.0     ],
            [ 0.0,      alpha,      0.0     ],
            [ 0.0,     -alpha,      0.0     ],
            [ 0.0,       0.0,      alpha    ],
            [ 0.0,       0.0,     -alpha    ]
        ])
        
        expected_matrix = pd.DataFrame(expected_matrix, columns=['A', 'B', 'C'])
                
        assert_frame_equal(coded.sort_values(by=['A', 'B', 'C']).reset_index(drop=True),
                          expected_matrix.sort_values(by=['A', 'B', 'C']).reset_index(drop=True),
                          atol=1e-6)
    
    # Test center points
    def test_center_points(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        center_points = 5
        design = CentralCompositeDesign(factors, design='ccc', center_points=center_points, replicates=0)
        
        # Expected: 8 + 5 = 13
        k = 2
        base_runs = 2**k + 2*k
        expected_runs = base_runs + center_points
        assert design._coded_design_matrix.shape[0] == expected_runs
        
        # Check center points are at the end
        center_rows = design._coded_design_matrix.tail(center_points)
        for col in center_rows.columns:
            assert all(center_rows[col] == 0.0)
    
    # Test replicates
    def test_replicates(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        replicates = 2
        design = CentralCompositeDesign(factors, design='ccf', center_points=0, replicates=replicates)
        
        # Base runs: 2^2 + 2*2 = 8, with replicates: 8 * (2+1) = 24
        k = 2
        base_runs = 2**k + 2*k
        expected_runs = base_runs * (replicates + 1)
        assert design._coded_design_matrix.shape[0] == expected_runs
        
        # Check unique combinations
        unique_combos = design._coded_design_matrix.drop_duplicates()
        assert len(unique_combos) == base_runs
    
    # Test center points and replicates together
    def test_center_points_and_replicates(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        center_points = 3
        replicates = 1
        design = CentralCompositeDesign(factors, design='ccc', center_points=center_points, replicates=replicates)
        
        # Expected: 8 * 2 + 3 = 19
        k = 2
        base_runs = 2**k + 2*k
        expected_runs = base_runs * (replicates + 1) + center_points
        assert design._coded_design_matrix.shape[0] == expected_runs
    
    # Test factorial points structure
    def test_factorial_points_structure(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        for design_type in ['ccc', 'ccf', 'cci']:
            design = CentralCompositeDesign(factors, design=design_type, center_points=0, replicates=0)
            coded = design._coded_design_matrix
            
            # Count factorial points (all factors non-zero)
            factorial_mask = (coded['A'] != 0) & (coded['B'] != 0) & (coded['C'] != 0)
            factorial_count = factorial_mask.sum()
            
            # Should be 2^k = 8
            assert factorial_count == 8, f"Design {design_type} should have 8 factorial points, got {factorial_count}"
    
    # Test axial points structure
    def test_axial_points_structure(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1),
            'C': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
        }
        
        for design_type in ['ccc', 'ccf', 'cci']:
            design = CentralCompositeDesign(factors, design=design_type, center_points=0, replicates=0)
            coded = design._coded_design_matrix
            
            k = 3
            # Count axial points (exactly one factor non-zero)
            axial_count = 0
            for factor in ['A', 'B', 'C']:
                other_factors = [f for f in ['A', 'B', 'C'] if f != factor]
                axial_mask = (coded[factor] != 0) & (coded[other_factors[0]] == 0) & (coded[other_factors[1]] == 0)
                axial_count += axial_mask.sum()
            
            # Should be 2*k = 6
            assert axial_count == 2*k, f"Design {design_type} should have {2*k} axial points, got {axial_count}"
    
    # Test that all designs have same uncoded bounds
    def test_uncoded_bounds_consistency(self):
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80, decimals=1),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5, decimals=2)
        }
        
        for design_type in ['ccc', 'ccf', 'cci']:
            design = CentralCompositeDesign(factors, design=design_type, center_points=0, replicates=0)
            
            # CCC and CCI extend beyond original bounds in uncoded space
            # CCF stays within bounds
            if design_type == 'ccf':
                assert design._design_matrix['Temperature'].min() >= 20.0
                assert design._design_matrix['Temperature'].max() <= 80.0
                assert design._design_matrix['Pressure'].min() >= 1.0
                assert design._design_matrix['Pressure'].max() <= 5.0
            # CCC extends beyond
            elif design_type == 'ccc':
                # Should have values outside the original range
                temp_range = design._design_matrix['Temperature'].max() - design._design_matrix['Temperature'].min()
                pressure_range = design._design_matrix['Pressure'].max() - design._design_matrix['Pressure'].min()
                assert temp_range > (80 - 20), "CCC should extend beyond factorial range"
                assert pressure_range > (5 - 1), "CCC should extend beyond factorial range"
    
    # Test CCC rotatability property
    def test_ccc_rotatability(self):
        factors = {
            'A': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5),
            'B': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=5)
        }
        
        k = 2
        design = CentralCompositeDesign(factors, design='ccc', center_points=0, replicates=0)
        coded = design._coded_design_matrix
        
        # For CCC, all points should be at constant distance from center except factorial
        # Factorial points at distance sqrt(k) from center
        # Axial points at distance alpha from center
        
        alpha = (2**k)**(1/4)
        
        # Calculate distances from center for each point
        distances = np.sqrt((coded**2).sum(axis=1))
        unique_distances = sorted(distances.unique())
        
        # Should have 3 unique distances: 0 (if center point), sqrt(k), alpha
        factorial_distance = np.sqrt(k)
        
        assert np.any(np.isclose(unique_distances, factorial_distance, atol=1e-5))
        assert np.any(np.isclose(unique_distances, alpha, atol=1e-5))
    
    # Test 5-factor CCC structure
    def test_5_factor_ccc_structure(self):
        factors = {
            f'F{i}': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10, decimals=1)
            for i in range(5)
        }
        
        k = 5
        design = CentralCompositeDesign(factors, design='ccc', center_points=0, replicates=0)
        
        # Expected: 2^5 + 2*5 = 32 + 10 = 42
        expected_runs = 2**k + 2*k
        assert design._coded_design_matrix.shape[0] == expected_runs
        
        coded = design._coded_design_matrix
        
        # Check factorial points
        factorial_mask = np.ones(len(coded), dtype=bool)
        for col in coded.columns:
            factorial_mask &= (coded[col] != 0)
        assert factorial_mask.sum() == 2**k
        
        # Check axial points
        axial_count = 0
        for col in coded.columns:
            other_cols = [c for c in coded.columns if c != col]
            axial_mask = (coded[col] != 0)
            for other_col in other_cols:
                axial_mask &= (coded[other_col] == 0)
            axial_count += axial_mask.sum()
        assert axial_count == 2*k
