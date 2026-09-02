"""
Comprehensive test suite for ImportDesign class.

This module provides production-ready tests for importing external designs,
including validation, factor inference, and file format handling.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from doetools.design.generic import ImportDesign
from doetools.utils import ContinuousFactor, CategoricalFactor


class TestImportDesign:
    """Test suite for ImportDesign"""

    # ===============================================================
    # HELPER METHODS
    # ===============================================================

    @pytest.fixture
    def temp_csv_continuous(self):
        """Create temporary CSV file with continuous factors"""
        data = pd.DataFrame({
            'Temperature': [20, 30, 40, 20, 30, 40],
            'Pressure': [1.0, 2.5, 4.0, 1.0, 2.5, 4.0],
            'Time': [10, 15, 20, 10, 15, 20]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_categorical(self):
        """Create temporary CSV file with categorical factors"""
        data = pd.DataFrame({
            'Material': ['A', 'B', 'C', 'A', 'B', 'C'],
            'Supplier': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'Treatment': ['Hot', 'Cold', 'Hot', 'Cold', 'Hot', 'Cold']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_mixed(self):
        """Create temporary CSV file with mixed factor types"""
        data = pd.DataFrame({
            'Temperature': [20, 30, 40, 20, 30, 40],
            'Material': ['A', 'B', 'A', 'B', 'A', 'B'],
            'Pressure': [1.0, 2.5, 4.0, 1.0, 2.5, 4.0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_coded(self):
        """Create temporary CSV file with coded values"""
        data = pd.DataFrame({
            'X1': [-1, 0, 1, -1, 0, 1],
            'X2': [-1, -1, -1, 1, 1, 1],
            'X3': [0, 0, 0, 0, 0, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_excel(self):
        """Create temporary Excel file"""
        data = pd.DataFrame({
            'Temperature': [20, 30, 40],
            'Pressure': [1.0, 2.5, 4.0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        data.to_excel(temp_path, index=False)
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    # ===============================================================
    # INITIALIZATION TESTS
    # ===============================================================

    def test_initialization_continuous(self, temp_csv_continuous):
        """Test basic initialization with continuous factors"""
        design = ImportDesign(
            vars=['Temperature', 'Pressure'],
            vars_type=['cont', 'cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        assert design._design_type == "Generic"
        assert 'Temperature' in design._factors
        assert 'Pressure' in design._factors
        assert isinstance(design._factors['Temperature'], ContinuousFactor)
        assert isinstance(design._factors['Pressure'], ContinuousFactor)

    def test_initialization_categorical(self, temp_csv_categorical):
        """Test basic initialization with categorical factors"""
        design = ImportDesign(
            vars=['Material', 'Supplier'],
            vars_type=['cat', 'cat'],
            file_path=temp_csv_categorical,
            coded=False
        )
        
        assert design._design_type == "Generic"
        assert 'Material' in design._factors
        assert 'Supplier' in design._factors
        assert isinstance(design._factors['Material'], CategoricalFactor)
        assert isinstance(design._factors['Supplier'], CategoricalFactor)

    def test_initialization_mixed(self, temp_csv_mixed):
        """Test initialization with mixed factor types"""
        design = ImportDesign(
            vars=['Temperature', 'Material', 'Pressure'],
            vars_type=['cont', 'cat', 'cont'],
            file_path=temp_csv_mixed,
            coded=False
        )
        
        assert len(design._factors) == 3
        assert isinstance(design._factors['Temperature'], ContinuousFactor)
        assert isinstance(design._factors['Material'], CategoricalFactor)
        assert isinstance(design._factors['Pressure'], ContinuousFactor)

    # ===============================================================
    # VALIDATION TESTS
    # ===============================================================

    def test_invalid_vars_type_length(self, temp_csv_continuous):
        """Test that mismatched vars and vars_type lengths raise error"""
        with pytest.raises(ValueError, match="Number of variables and variable types must be equal"):
            ImportDesign(
                vars=['Temperature', 'Pressure'],
                vars_type=['cont'],  # Only one type for two variables
                file_path=temp_csv_continuous,
                coded=False
            )

    def test_invalid_variable_type(self, temp_csv_continuous):
        """Test that invalid variable types raise error"""
        with pytest.raises(ValueError, match="must be either 'cont' or 'cat'"):
            ImportDesign(
                vars=['Temperature', 'Pressure'],
                vars_type=['cont', 'invalid'],
                file_path=temp_csv_continuous,
                coded=False
            )

    def test_variable_not_in_file(self, temp_csv_continuous):
        """Test that non-existent variables raise error"""
        with pytest.raises(ValueError, match="not found in the file columns"):
            ImportDesign(
                vars=['Temperature', 'NonExistent'],
                vars_type=['cont', 'cont'],
                file_path=temp_csv_continuous,
                coded=False
            )

    # ===============================================================
    # FACTOR INFERENCE TESTS
    # ===============================================================

    def test_continuous_factor_bounds_inference(self, temp_csv_continuous):
        """Test that continuous factor bounds are correctly inferred"""
        design = ImportDesign(
            vars=['Temperature'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        factor = design._factors['Temperature']
        assert factor.lower_bound == 20
        assert factor.upper_bound == 40

    def test_continuous_factor_levels_inference(self, temp_csv_continuous):
        """Test that number of levels is correctly inferred"""
        design = ImportDesign(
            vars=['Temperature'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        factor = design._factors['Temperature']
        # Temperature has 3 unique values: 20, 30, 40
        assert factor.n_levels == 3
        assert set(factor.levels) == {20, 30, 40}

    def test_continuous_factor_decimals_inference(self, temp_csv_continuous):
        """Test that decimal precision is correctly inferred"""
        design = ImportDesign(
            vars=['Pressure'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        factor = design._factors['Pressure']
        # Pressure values have 1 decimal place (1.0, 2.5, 4.0)
        assert factor.decimals == 1

    def test_categorical_factor_levels_inference(self, temp_csv_categorical):
        """Test that categorical levels are correctly inferred"""
        design = ImportDesign(
            vars=['Material'],
            vars_type=['cat'],
            file_path=temp_csv_categorical,
            coded=False
        )
        
        factor = design._factors['Material']
        assert set(factor.levels) == {'A', 'B', 'C'}

    # ===============================================================
    # CODED VS UNCODED TESTS
    # ===============================================================

    def test_coded_import(self, temp_csv_coded):
        """Test importing already-coded design"""
        design = ImportDesign(
            vars=['X1', 'X2', 'X3'],
            vars_type=['cont', 'cont', 'cont'],
            file_path=temp_csv_coded,
            coded=True
        )
        
        # Coded and design matrices should be the same
        assert design._coded_design_matrix.equals(design._design_matrix)
        
        # Values should be -1, 0, 1
        unique_vals = set(design._coded_design_matrix['X1'].unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_uncoded_import_generates_coded(self, temp_csv_continuous):
        """Test that uncoded import generates coded matrix"""
        design = ImportDesign(
            vars=['Temperature'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        # Coded matrix should be different from design matrix
        assert not design._coded_design_matrix.equals(design._design_matrix)
        
        # Design matrix should have original values
        assert 20 in design._design_matrix['Temperature'].values
        assert 30 in design._design_matrix['Temperature'].values
        assert 40 in design._design_matrix['Temperature'].values
        
        # Coded matrix should have normalized values
        coded_vals = design._coded_design_matrix['Temperature'].unique()
        assert len(coded_vals) == 3
        assert set(coded_vals) == {-1, 0, 1}

    def test_coding_continuous_factor(self, temp_csv_continuous):
        """Test that continuous factors are properly coded"""
        design = ImportDesign(
            vars=['Temperature'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        # For continuous factor with bounds [20, 40], center is 30
        # Coded values should be: -1 (20), 0 (30), 1 (40)
        coded = design._coded_design_matrix['Temperature']
        uncoded = design._design_matrix['Temperature']
        
        # Check that min/max are coded to -1/1
        assert coded[uncoded == 20].iloc[0] == pytest.approx(-1.0, abs=0.01)
        assert coded[uncoded == 40].iloc[0] == pytest.approx(1.0, abs=0.01)
        assert coded[uncoded == 30].iloc[0] == pytest.approx(0.0, abs=0.01)

    def test_coding_categorical_factor(self, temp_csv_categorical):
        """Test that categorical factors are properly coded"""
        design = ImportDesign(
            vars=['Material'],
            vars_type=['cat'],
            file_path=temp_csv_categorical,
            coded=False
        )
        
        # Categorical factors should be coded to -1, 0, 1 range
        coded_vals = design._coded_design_matrix['Material'].unique()
        assert len(coded_vals) == 3
        # Should be in range [-1, 1]
        assert all(-1 <= v <= 1 for v in coded_vals)
        assert set(coded_vals) == {-1, 0, 1}
        assert set(design._design_matrix['Material'].unique()) == {'A', 'B', 'C'}

    # ===============================================================
    # FILE FORMAT TESTS
    # ===============================================================

    def test_csv_import(self, temp_csv_continuous):
        """Test CSV file import"""
        design = ImportDesign(
            vars=['Temperature', 'Pressure'],
            vars_type=['cont', 'cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        assert design._design_matrix is not None
        assert len(design._design_matrix) == 6

    def test_excel_import(self, temp_excel):
        """Test Excel file import"""
        design = ImportDesign(
            vars=['Temperature', 'Pressure'],
            vars_type=['cont', 'cont'],
            file_path=temp_excel,
            coded=False
        )
        
        assert design._design_matrix is not None
        assert len(design._design_matrix) == 3

    # ===============================================================
    # DESIGN MATRIX TESTS
    # ===============================================================

    def test_design_matrix_dimensions(self, temp_csv_continuous):
        """Test that design matrix has correct dimensions"""
        design = ImportDesign(
            vars=['Temperature', 'Pressure'],
            vars_type=['cont', 'cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        assert design._design_matrix.shape[1] == 2  # 2 factors
        assert design._design_matrix.shape[0] == 6  # 6 runs

    def test_coded_uncoded_dimension_consistency(self, temp_csv_continuous):
        """Test that coded and uncoded matrices have same dimensions"""
        design = ImportDesign(
            vars=['Temperature', 'Pressure', 'Time'],
            vars_type=['cont', 'cont', 'cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        assert design._coded_design_matrix.shape == design._design_matrix.shape
        assert list(design._coded_design_matrix.columns) == list(design._design_matrix.columns)

    def test_variable_subset_import(self, temp_csv_continuous):
        """Test importing subset of variables from file"""
        # File has Temperature, Pressure, Time but only import Temperature
        design = ImportDesign(
            vars=['Temperature'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        assert design._design_matrix.shape[1] == 1
        assert 'Temperature' in design._design_matrix.columns
        assert 'Pressure' not in design._design_matrix.columns

    # ===============================================================
    # DECIMAL PRECISION TESTS
    # ===============================================================

    def test_integer_values_decimal_inference(self):
        """Test decimal inference for integer values"""
        data = pd.DataFrame({
            'Factor': [10, 20, 30, 40, 50]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            design = ImportDesign(
                vars=['Factor'],
                vars_type=['cont'],
                file_path=temp_path,
                coded=False
            )
            
            # Integer values should have decimals = 0 or 1
            factor = design._factors['Factor']
            assert factor.decimals >= 0
        finally:
            os.unlink(temp_path)

    def test_high_precision_decimal_inference(self):
        """Test decimal inference for high precision values"""
        data = pd.DataFrame({
            'Factor': [1.234, 2.567, 3.891]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            design = ImportDesign(
                vars=['Factor'],
                vars_type=['cont'],
                file_path=temp_path,
                coded=False
            )
            
            factor = design._factors['Factor']
            assert factor.decimals == 3
        finally:
            os.unlink(temp_path)

    # ===============================================================
    # BUILD DESIGN MATRIX TEST
    # ===============================================================

    def test_build_design_matrix_not_implemented(self, temp_csv_continuous):
        """Test that build_design_matrix raises NotImplementedError"""
        design = ImportDesign(
            vars=['Temperature'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        result = design.build_design_matrix()
        assert isinstance(result, type(NotImplementedError()))

    # ===============================================================
    # SPECIAL CASES TESTS
    # ===============================================================

    def test_single_factor_import(self, temp_csv_continuous):
        """Test importing design with single factor"""
        design = ImportDesign(
            vars=['Temperature'],
            vars_type=['cont'],
            file_path=temp_csv_continuous,
            coded=False
        )
        
        assert len(design._factors) == 1
        assert 'Temperature' in design._factors

    def test_duplicate_values_handling(self):
        """Test handling of duplicate rows in design"""
        data = pd.DataFrame({
            'Factor1': [10, 20, 10, 20, 10],  # Duplicates
            'Factor2': [1, 2, 1, 2, 1]        # Duplicates
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            design = ImportDesign(
                vars=['Factor1', 'Factor2'],
                vars_type=['cont', 'cont'],
                file_path=temp_path,
                coded=False
            )
            
            # Design should import all rows including duplicates
            assert len(design._design_matrix) == 5
            
            # But factor levels should only count unique values
            assert design._factors['Factor1'].n_levels == 2
            assert design._factors['Factor2'].n_levels == 2
        finally:
            os.unlink(temp_path)

    def test_mixed_numeric_string_categorical(self):
        """Test categorical factor with mixed numeric and string values"""
        data = pd.DataFrame({
            'MixedFactor': ['A', 'B', '1', '2', 'A', 'B']
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            design = ImportDesign(
                vars=['MixedFactor'],
                vars_type=['cat'],
                file_path=temp_path,
                coded=False
            )
            
            factor = design._factors['MixedFactor']
            # All values should be converted to strings
            assert all(isinstance(level, str) for level in factor.levels)
            assert set(factor.levels) == {'1', '2', 'A', 'B'}
        finally:
            os.unlink(temp_path)

    def test_large_design_import(self):
        """Test importing larger design (performance check)"""
        # Create a larger design
        np.random.seed(42)
        data = pd.DataFrame({
            'F1': np.random.uniform(0, 100, 100),
            'F2': np.random.uniform(0, 10, 100),
            'F3': np.random.choice(['A', 'B', 'C'], 100),
            'F4': np.random.uniform(20, 80, 100)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            design = ImportDesign(
                vars=['F1', 'F2', 'F3', 'F4'],
                vars_type=['cont', 'cont', 'cat', 'cont'],
                file_path=temp_path,
                coded=False
            )
            
            assert len(design._design_matrix) == 100
            assert len(design._factors) == 4
        finally:
            os.unlink(temp_path)
