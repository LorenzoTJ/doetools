import numpy as np
import pandas as pd
import pytest

from doetools import (
    ConstrainedMixtureDesign,
    ContinuousFactor,
    DOptAddDesign,
    DOptDesign,
    FullFactorialDesign,
    ImportDesign,
    MixtureFactor,
    ModelTerms,
    SimplexCentroidDesign,
    SimplexLatticeDesign,
)
from doetools.utils.pdf_report import DesignReportPDF
from doetools.utils.summary import DesignSummaryMixin


class DummyFactor:
    def __init__(self, factor_type, levels, coded_levels, decimals=2):
        self.type = factor_type
        self.levels = levels
        self.coded_levels = np.array(coded_levels)
        self.decimals = decimals


class DummyWrapper:
    def __init__(self):
        self.results = {}
        self.vif = pd.DataFrame({"VIF": [1.0, 1.2]}, index=["A", "B"])


class DummyDesign(DesignSummaryMixin):
    def __init__(self):
        # placeholders; tests will set attributes as needed
        self._design_type = ""
        self._coded_design_matrix = pd.DataFrame()
        self._design_matrix = pd.DataFrame()
        self._factors = {}
        self._model_spec = None
        self._response_conditions = None
        self._responses = pd.DataFrame()
        self._response_list = []
        self._mlr_wrapper = DummyWrapper()
        self._model_matrix = pd.DataFrame()

    # helper stubs
    def _group_by_replicates(self):
        return []

    def _number_of_center_points(self, coded_design_matrix):
        return 0


@pytest.fixture
def dummy_design():
    return DummyDesign()


# ==============================================================================
#                    Test PDF Report to_multicol_df Logic
# ==============================================================================

def test_to_multicol_df_basic_splitting(dummy_design):
    """Test basic dataframe splitting into multiple column blocks."""
    pdf = DesignReportPDF(dummy_design)
    
    # Create dataframe with 25 rows
    df = pd.DataFrame({
        'A': range(1, 26),
        'B': range(101, 126)
    })
    
    # Split into blocks of 10 rows
    result = pdf.to_multicol_df(df, max_rows=10, runs=False)
    
    # Should create 3 blocks: 10 + 10 + 5 (padded to 10)
    # Result should have 10 rows (max_rows) and 6 columns (3 blocks × 2 original columns)
    assert result.shape[0] == 10
    assert result.shape[1] == 6  # 3 blocks × 2 columns


def test_to_multicol_df_with_run_column(dummy_design):
    """Test that Run column is correctly added and numbered."""
    pdf = DesignReportPDF(dummy_design)
    
    df = pd.DataFrame({
        'Value': [10, 20, 30, 40, 50]
    })
    
    result = pdf.to_multicol_df(df, max_rows=3, runs=True)
    
    # Should have Run column in each block
    # First block: Run 1-3, Second block: Run 4-5
    assert 'Run' in result.columns
    
    # Check first block Run values
    assert result.iloc[0, 0] == '1'  # First Run column, first value
    assert result.iloc[1, 0] == '2'
    assert result.iloc[2, 0] == '3'
    
    # Check second block Run values (starts at column 2, index 0-based)
    assert result.iloc[0, 2] == '4'  # Second Run column, first value
    assert result.iloc[1, 2] == '5'


def test_to_multicol_df_padding(dummy_design):
    """Test that blocks are padded with empty strings to equal length."""
    pdf = DesignReportPDF(dummy_design)
    
    # Create dataframe with 22 rows
    df = pd.DataFrame({'X': range(22)})
    
    # Split into blocks of 10: will have 3 blocks (10, 10, 2)
    # Last block should be padded to 10 rows
    result = pdf.to_multicol_df(df, max_rows=10, runs=False)
    
    # All blocks should have same height (10 rows)
    assert result.shape[0] == 10
    
    # Last block (columns 2) should have empty strings in rows 2-9
    for row_idx in range(2, 10):
        assert result.iloc[row_idx, 2] == ''


def test_to_multicol_df_empty_dataframe(dummy_design):
    """Test handling of empty dataframe."""
    pdf = DesignReportPDF(dummy_design)
    
    df = pd.DataFrame({'A': [], 'B': []})
    
    result = pdf.to_multicol_df(df, max_rows=10, runs=False)
    
    # Should return empty dataframe
    assert len(result) == 0
    assert result.shape[0] == 0


def test_to_multicol_df_single_block(dummy_design):
    """Test when data fits in a single block."""
    pdf = DesignReportPDF(dummy_design)
    
    df = pd.DataFrame({
        'Col1': [1, 2, 3],
        'Col2': [4, 5, 6]
    })
    
    result = pdf.to_multicol_df(df, max_rows=10, runs=False)
    
    # Should have only one block (original columns)
    assert result.shape[1] == 2
    assert result.shape[0] == 3


def test_to_multicol_df_exact_multiple(dummy_design):
    """Test when number of rows is exact multiple of max_rows."""
    pdf = DesignReportPDF(dummy_design)
    
    # Exactly 20 rows with max_rows=10 should create 2 blocks with no padding
    df = pd.DataFrame({'Value': range(20)})
    
    result = pdf.to_multicol_df(df, max_rows=10, runs=False)
    
    # Should have 10 rows and 2 columns (2 blocks)
    assert result.shape == (10, 2)
    
    # No padding needed - all values should be numeric strings
    for col_idx in range(result.shape[1]):
        for row_idx in range(10):
            assert result.iloc[row_idx, col_idx] != ''


def test_to_multicol_df_run_numbering_continuity(dummy_design):
    """Test that Run numbers continue across blocks correctly."""
    pdf = DesignReportPDF(dummy_design)
    
    df = pd.DataFrame({'Data': range(15)})
    
    result = pdf.to_multicol_df(df, max_rows=6, runs=True)
    
    # Should have 3 blocks: 6 + 6 + 3
    # Block 1: Run 1-6 (columns 0-1)
    # Block 2: Run 7-12 (columns 2-3)
    # Block 3: Run 13-15 (columns 4-5)
    
    # Check continuity
    assert result.iloc[0, 0] == '1'   # Block 1, row 1
    assert result.iloc[5, 0] == '6'   # Block 1, row 6
    assert result.iloc[0, 2] == '7'   # Block 2, row 1
    assert result.iloc[5, 2] == '12'  # Block 2, row 6
    assert result.iloc[0, 4] == '13'  # Block 3, row 1
    assert result.iloc[2, 4] == '15'  # Block 3, row 3
    
    # Padded rows in block 3 should have empty Run values
    assert result.iloc[3, 4] == ''
    assert result.iloc[4, 4] == ''
    assert result.iloc[5, 4] == ''


def test_to_multicol_df_preserves_column_order(dummy_design):
    """Test that original column order is preserved within each block."""
    pdf = DesignReportPDF(dummy_design)
    
    df = pd.DataFrame({
        'First': [1, 2, 3, 4, 5],
        'Second': [10, 20, 30, 40, 50],
        'Third': [100, 200, 300, 400, 500]
    })
    
    result = pdf.to_multicol_df(df, max_rows=3, runs=False)
    
    # Should have 2 blocks
    # Block 1: columns 0-2 (First, Second, Third for rows 1-3)
    # Block 2: columns 3-5 (First, Second, Third for rows 4-5)
    
    # Check that column names appear in correct order
    col_names = list(result.columns)
    assert col_names[0] == 'First'
    assert col_names[1] == 'Second'
    assert col_names[2] == 'Third'
    assert col_names[3] == 'First'
    assert col_names[4] == 'Second'
    assert col_names[5] == 'Third'


# ==============================================================================
#                    Test PDF Report Smoke Tests
# ==============================================================================

def test_pdf_report_initialization(dummy_design):
    """Test that DesignReportPDF initializes correctly."""
    pdf = DesignReportPDF(dummy_design)
    
    assert pdf.design is dummy_design
    assert pdf.title == "Design and Model Summary Report"
    assert pdf.auto_page_break is True


def test_pdf_report_custom_title(dummy_design):
    """Test initialization with custom title."""
    custom_title = "My Custom Report Title"
    pdf = DesignReportPDF(dummy_design, title=custom_title)
    
    assert pdf.title == custom_title


def test_report_title_is_only_rendered_on_the_first_page(dummy_design):
    pdf = DesignReportPDF(dummy_design)
    pdf.add_page()
    pdf.add_page()

    assert pdf.pages[1].contents.count(pdf.title.encode("latin-1")) == 1
    assert pdf.pages[2].contents.count(pdf.title.encode("latin-1")) == 0
    assert pdf.get_y() >= pdf.t_margin


def test_pdf_report_formatting_methods(dummy_design):
    """Test that formatting methods can be called without errors."""
    pdf = DesignReportPDF(dummy_design)
    pdf.add_page()
    
    # These should not raise exceptions
    pdf.add_title("Test Title")
    pdf.add_subtitle("Test Subtitle")
    pdf.add_paragraph("Test paragraph text")
    pdf.add_separator()


def test_pdf_report_table_from_df(dummy_design):
    """Test that add_table_from_df handles DataFrames correctly."""
    pdf = DesignReportPDF(dummy_design)
    pdf.add_page()
    
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    # Should not raise exceptions
    pdf.add_table_from_df(df)


# ==============================================================================
#                 Model diagnostics and interpretation helpers
# ==============================================================================

def test_model_matrix_diagnostics_full_rank_with_intercept():
    matrix = pd.DataFrame({
        "Intercept": [1.0, 1.0, 1.0, 1.0],
        "A": [-1.0, 0.0, 1.0, 2.0],
    })

    diagnostics = DesignReportPDF._model_matrix_diagnostics(matrix)

    assert diagnostics["Rank"] == 2
    assert diagnostics["Aliased columns"] == 0
    assert diagnostics["Intercept"] == "Yes"
    assert diagnostics["Regression DF"] == 1
    assert diagnostics["Residual DF"] == 2
    assert diagnostics["Status"] == "Full rank"
    assert "Condition number" not in diagnostics


def test_model_matrix_diagnostics_rank_deficient_and_saturated():
    deficient = pd.DataFrame({
        "Intercept": [1.0, 1.0, 1.0],
        "A": [-1.0, 0.0, 1.0],
        "A alias": [-2.0, 0.0, 2.0],
    })
    saturated_mixture = pd.DataFrame({
        "A": [1.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0],
        "C": [0.0, 0.0, 1.0],
    })

    deficient_diagnostics = DesignReportPDF._model_matrix_diagnostics(deficient)
    saturated_diagnostics = DesignReportPDF._model_matrix_diagnostics(saturated_mixture)

    assert deficient_diagnostics["Rank"] == 2
    assert deficient_diagnostics["Aliased columns"] == 1
    assert "Rank deficient" in deficient_diagnostics["Status"]
    assert saturated_diagnostics["Intercept"] == "Yes"
    assert saturated_diagnostics["Regression DF"] == 2
    assert saturated_diagnostics["Residual DF"] == 0
    assert "Saturated" in saturated_diagnostics["Status"]


@pytest.mark.parametrize(
    ("p_value", "expected_title"),
    [(0.009, "Highly significant"), (0.01, "Significant"),
     (0.049, "Significant"), (0.05, "Not significant")],
)
def test_model_f_interpretation_thresholds(p_value, expected_title):
    f_test = pd.DataFrame({"F_value": [4.0], "p_value": [p_value]})
    anova = pd.DataFrame({
        "Source": ["Regression", "Residuals"],
        "df": [2.0, 8.0],
    })

    title, _, _ = DesignReportPDF._interpret_model_f_test(f_test, anova)

    assert title == expected_title


def test_model_f_interpretation_uses_the_correct_null_hypothesis():
    f_test = pd.DataFrame({"F_value": [4.0], "p_value": [0.02]})
    anova = pd.DataFrame({"Source": ["Regression", "Residuals"], "df": [2.0, 8.0]})

    _, with_constant, _ = DesignReportPDF._interpret_model_f_test(f_test, anova, has_constant=True)
    _, without_constant, _ = DesignReportPDF._interpret_model_f_test(f_test, anova, has_constant=False)

    assert "beyond a constant response" in with_constant
    assert "joint null hypothesis that all coefficients are zero" in without_constant


def test_lack_of_fit_interpretation_does_not_claim_adequacy():
    f_test = pd.DataFrame({"F_value": [1.0], "p_value": [0.3]})
    anova = pd.DataFrame({
        "Source": ["Lack of Fit", "Pure Error"],
        "df": [2.0, 4.0],
    })

    title, message, kind = DesignReportPDF._interpret_lof_f_test(f_test, anova)

    assert title == "No significant lack of fit"
    assert kind == "success"
    assert "does not prove model adequacy" in message


def test_design_overview_cards_only_include_present_factor_types():
    summary = {
        "Design": "Simplex Lattice",
        "Runs": 9,
        "Replicates": 2,
        "Center Points": 0,
        "Continuous": 0,
        "Categorical": 0,
        "Mixture Components": 3,
    }

    cards = DesignReportPDF._overview_card_values(summary)

    assert list(cards) == ["Design", "Runs", "Replicates", "Center Points", "Mixture Components"]
    assert "Continuous" not in cards
    assert "Categorical" not in cards


def test_f_interpretations_handle_undefined_statistics():
    undefined = pd.DataFrame({"F_value": [np.inf], "p_value": [np.nan]})
    anova = pd.DataFrame({
        "Source": ["Regression", "Residuals", "Lack of Fit", "Pure Error"],
        "df": [2.0, 0.0, 0.0, 0.0],
    })

    assert DesignReportPDF._interpret_model_f_test(undefined, anova)[0] == "Not interpretable"
    assert DesignReportPDF._interpret_lof_f_test(undefined, anova)[0] == "Not interpretable"


@pytest.mark.parametrize(
    ("value", "assessment"),
    [(1.0, "Low"), (5.0, "Elevated"), (10.0, "Elevated"),
     (10.01, "High"), (np.inf, "Not estimable"), (None, "Not defined")],
)
def test_vif_assessment(value, assessment):
    assert DesignReportPDF._assess_vif(value) == assessment


def test_safe_text_normalizes_unicode_and_special_values():
    assert DesignReportPDF._safe_text(True) == "Yes"
    assert DesignReportPDF._safe_text(False) == "No"
    assert DesignReportPDF._safe_text(pd.NA) == "N/A"
    assert DesignReportPDF._safe_text(float("inf")) == "inf"
    assert DesignReportPDF._safe_text("R² × ≤ β") == "R^2 x <= ?"


def test_wide_design_matrix_is_split_without_losing_columns():
    matrix = pd.DataFrame({"Run": range(1, 4)})
    factor_columns = [f"Long process factor {index}" for index in range(18)]
    for index, column in enumerate(factor_columns):
        matrix[column] = index

    groups = DesignReportPDF._split_design_matrix(matrix, max_width=120)

    assert len(groups) > 1
    assert all(group.columns[0] == "Run" for group in groups)
    assert [column for group in groups for column in group.columns[1:]] == factor_columns
    assert all(len(group) == len(matrix) for group in groups)


def test_atomic_write_preserves_existing_file_on_replace_failure(tmp_path, monkeypatch):
    output = tmp_path / "report.pdf"
    output.write_bytes(b"existing-pdf")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("doetools.utils.pdf_report.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        DesignReportPDF._atomic_write(output, b"new-pdf")

    assert output.read_bytes() == b"existing-pdf"
    assert list(tmp_path.glob("*.tmp")) == []


def test_expected_optional_absence_uses_configured_notice(dummy_design):
    pdf = DesignReportPDF(dummy_design)

    section = pdf._capture_optional(
        "Response conditions",
        lambda: (_ for _ in ()).throw(ValueError("internal getter message")),
        empty_notice="Response conditions are not defined.",
        missing_is_expected=True,
    )

    assert section.applicable is False
    assert section.notice == "Response conditions are not defined."
    assert pdf._report_notices == []


# ==============================================================================
#                         End-to-end report smoke tests
# ==============================================================================

def _fit_design_for_pdf(design, model_terms):
    design.set_model_terms(model_terms)
    matrix = design.get_model_matrix().to_numpy(dtype=float)
    coefficients = np.arange(1, matrix.shape[1] + 1, dtype=float)
    response = matrix @ coefficients + np.linspace(0.0, 0.03, len(matrix))
    design._response_list = ["Yield β"]
    design._responses = pd.DataFrame({"Yield β": response})
    design.compute_mlr_model()
    return design


def _process_design():
    design = FullFactorialDesign({
        "Temperature °C": ContinuousFactor(3, 20.0, 80.0),
        "Pressure": ContinuousFactor(3, 1.0, 5.0),
    })
    return _fit_design_for_pdf(
        design,
        ModelTerms(pro_main="all", pro_int2=None, pro_quadratic=None),
    )


def _mixture_design(kind):
    factors = {name: MixtureFactor(0.0, 1.0) for name in ("A", "B", "C")}
    if kind == "lattice":
        design = SimplexLatticeDesign(factors, m=2)
    elif kind == "centroid":
        design = SimplexCentroidDesign(factors)
    else:
        design = ConstrainedMixtureDesign(factors, center_points=1)
    return _fit_design_for_pdf(
        design,
        ModelTerms(
            intercept=False, pro_main=None, pro_int2=None,
            pro_quadratic=None, scheffe_pol_order=1,
        ),
    )


def _d_optimal_design(kind):
    if kind == "process":
        factors = {
            "Temperature": ContinuousFactor(3, 20.0, 80.0),
            "Pressure": ContinuousFactor(3, 1.0, 5.0),
        }
        design = DOptDesign(factors, process_strategy="grid", mixture_include=None)
        terms = ModelTerms(pro_main="all", pro_int2=None, pro_quadratic=None)
        runs = 5
    elif kind == "mixture":
        factors = {name: MixtureFactor(0.0, 1.0) for name in ("A", "B", "C")}
        design = DOptDesign(factors, process_strategy=None, mixture_include="all")
        terms = ModelTerms(
            intercept=False, pro_main=None, pro_int2=None,
            pro_quadratic=None, scheffe_pol_order=1,
        )
        runs = 5
    else:
        factors = {
            "Temperature": ContinuousFactor(3, 20.0, 80.0),
            "A": MixtureFactor(0.0, 1.0),
            "B": MixtureFactor(0.0, 1.0),
        }
        design = DOptDesign(factors, process_strategy="grid", mixture_include="all")
        terms = ModelTerms(
            intercept=False, pro_main="all", pro_int2=None,
            pro_quadratic=None, mix_main="all",
        )
        runs = 6
    design.set_model_terms(terms)
    design.compute_d_optimal(runs, runs, trials=3, max_no_improve=2, graph=False)
    design.select_design(runs)
    matrix = design.get_model_matrix().to_numpy(dtype=float)
    response = matrix @ np.arange(1, matrix.shape[1] + 1) + np.linspace(0.0, 0.02, runs)
    design._response_list = ["Yield"]
    design._responses = pd.DataFrame({"Yield": response})
    design.compute_mlr_model()
    return design


@pytest.mark.parametrize(
    "design_factory",
    [
        _process_design,
        lambda: _mixture_design("lattice"),
        lambda: _mixture_design("centroid"),
        lambda: _mixture_design("constrained"),
        lambda: _d_optimal_design("process"),
        lambda: _d_optimal_design("mixture"),
        lambda: _d_optimal_design("mixed"),
    ],
    ids=["process", "simplex-lattice", "simplex-centroid", "constrained-mixture",
         "d-opt-process", "d-opt-mixture", "d-opt-mixed"],
)
def test_complete_pdf_smoke_for_supported_designs(design_factory, tmp_path):
    design = design_factory()
    output = tmp_path / "report.pdf"

    result = design.get_model_summary_pdf(str(output))

    assert result is None
    assert output.stat().st_size > 500
    assert output.read_bytes().startswith(b"%PDF-")


def test_complete_pdf_smoke_for_d_optimal_augmentation(tmp_path):
    source = tmp_path / "existing_design.csv"
    pd.DataFrame({
        "X1": [0.0, 10.0, 0.0, 10.0, 5.0],
        "X2": [0.0, 0.0, 10.0, 10.0, 5.0],
    }).to_csv(source, index=False)
    design = DOptAddDesign(
        factors={
            "X1": ContinuousFactor(3, 0.0, 10.0),
            "X2": ContinuousFactor(3, 0.0, 10.0),
        },
        source=str(source),
    )
    design.set_model_terms(ModelTerms(pro_main="all", pro_int2=None, pro_quadratic=None))
    design.generate_cp(
        process_strategy="grid",
    )
    design.compute_d_optimal(
        n_min=1, n_max=1, trials=3, max_no_improve=2,
        random_state=42, graph=False,
    )
    design.select_design(1)
    matrix = design.get_model_matrix().to_numpy(dtype=float)
    design._response_list = ["Yield"]
    design._responses = pd.DataFrame({
        "Yield": matrix @ np.arange(1, matrix.shape[1] + 1) + np.linspace(0.0, 0.02, len(matrix))
    })
    design.compute_mlr_model()
    output = tmp_path / "augmentation_report.pdf"

    design.get_model_summary_pdf(str(output))

    assert output.stat().st_size > 500
    assert output.read_bytes().startswith(b"%PDF-")


def test_complete_pdf_smoke_for_imported_design(tmp_path):
    source = tmp_path / "imported_design.csv"
    pd.DataFrame({
        "Temperature": np.repeat([20.0, 50.0, 80.0], 3),
        "Pressure": np.tile([1.0, 3.0, 5.0], 3),
    }).to_csv(source, index=False)
    design = ImportDesign(
        factors={
            "Temperature": ContinuousFactor(3, 20.0, 80.0),
            "Pressure": ContinuousFactor(3, 1.0, 5.0),
        },
        source=str(source),
    )
    design = _fit_design_for_pdf(
        design,
        ModelTerms(pro_main="all", pro_int2=None, pro_quadratic=None),
    )
    output = tmp_path / "imported_report.pdf"

    design.get_model_summary_pdf(str(output))

    assert output.stat().st_size > 500
    assert output.read_bytes().startswith(b"%PDF-")
