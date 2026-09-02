import numpy as np
import pandas as pd
import pytest

from doetools.utils.factors import CategoricalFactor
from doetools.utils.summary import DesignSummaryMixin


class DummyFactor:
    def __init__(self, factor_type, levels, coded_levels, decimals=2):
        self.type = factor_type
        self.levels = levels
        self.coded_levels = np.array(coded_levels)
        self.decimals = decimals


class DummyMixtureFactor:
    type = "mix"

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

class DummyResult:
    def __init__(self, y_hat=None, coef=None, anova=None, dispersion_matrix=None, replicates=None, metrics=None):
        self.y_hat = y_hat if y_hat is not None else pd.Series(dtype=float)
        self.coef = coef if coef is not None else pd.DataFrame()
        self.anova = anova if anova is not None else {}
        self.dispersion_matrix = dispersion_matrix if dispersion_matrix is not None else pd.DataFrame()
        self.replicates = replicates if replicates is not None else {}
        self.metrics = metrics if metrics is not None else {}


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

    def _compute_leverage(self, points):
        return np.zeros(points.shape[0])


@pytest.fixture
def dummy_design():
    return DummyDesign()


def test_get_design_summary(dummy_design):
    dummy_design._design_type = "CCD"
    dummy_design._coded_design_matrix = pd.DataFrame(np.zeros((5, 2)), columns=["A", "B"])
    dummy_design._factors = {
        "A": DummyFactor("cont", [0, 1], [0.0, 1.0]),
        "B": DummyFactor("cat", ["L", "H"], [-1, 1]),
    }
    dummy_design._number_of_center_points = lambda df: 1
    dummy_design._group_by_replicates = lambda: [[0, 1], [2]]

    df = dummy_design.get_design_summary()

    assert df.loc["", "Design"] == "CCD"
    assert df.loc["", "Runs"] == 5
    assert df.loc["", "Replicates"] == 1  # (2-1)+(1-1)
    assert df.loc["", "Center Points"] == 1
    assert df.loc["", "Continuous"] == 1
    assert df.loc["", "Categorical"] == 1


def test_get_design_summary_counts_mixture_components(dummy_design):
    dummy_design._design_type = "D-Optimal"
    dummy_design._coded_design_matrix = pd.DataFrame(np.zeros((4, 2)), columns=["A", "B"])
    dummy_design._factors = {
        "A": DummyFactor("cont", [0, 1], [0.0, 1.0]),
        "B": DummyMixtureFactor(0.1, 0.9),
    }

    summary = dummy_design.get_design_summary()

    assert summary.loc["", "Continuous"] == 1
    assert summary.loc["", "Mixture Components"] == 1


def test_get_factor_summary(dummy_design):
    
    dummy_design._factors = {
        "A": DummyFactor("cont", [0, 1], [0.1234, 0.5678], decimals=2),
        "B": DummyFactor("cat", ["L", "M", "H"], [-1, 0, 1]),
    }

    df = dummy_design.get_factor_summary()

    assert set(df["Factor"]) == {"A", "B"}
    a_row = df[df["Factor"] == "A"].iloc[0]
    assert a_row["Type"] == "cont"
    assert a_row["Levels Count"] == 2
    assert np.allclose(a_row["Coded Levels"], np.array([0.12, 0.57]))


def test_get_factor_summary_includes_categorical_reference(dummy_design):
    dummy_design._factors = {
        "Catalyst": CategoricalFactor(
            levels=["A", "B", "C"], reference_level="B"
        )
    }

    summary = dummy_design.get_factor_summary()

    assert summary.loc[0, "Reference Level"] == "B"


def test_get_factor_summary_supports_mixture_components(dummy_design):
    dummy_design._factors = {
        "A": DummyFactor("cont", [0, 1], [0.0, 1.0]),
        "Solvent": DummyMixtureFactor(0.1, 0.8),
    }

    df = dummy_design.get_factor_summary()
    mixture = df.loc[df["Factor"] == "Solvent"].iloc[0]

    assert mixture["Type"] == "mix"
    assert mixture["Lower Bound"] == 0.1
    assert mixture["Upper Bound"] == 0.8
    assert pd.isna(mixture["Levels Count"])


def test_get_model_term_summary(dummy_design):
    class MS:
        intercept = True
        main = ["A", "B"]
        interaction2 = [("A", "B")]
        interaction3 = None
        quadratic = ["A"]
    dummy_design._model_spec = MS()

    df = dummy_design.get_model_term_summary()

    assert df.shape[0] == 5
    assert df.loc[df["Terms"] == "Intercept", "Included"].iloc[0] is True
    linear_terms = df.loc[df["Terms"] == "Linear Terms", "Included"].iloc[0]
    assert linear_terms == ["A", "B"]

    int2_terms = df.loc[df["Terms"] == "2-Term Int.", "Included"].iloc[0]
    assert "A : B" in int2_terms

    quad_terms = df.loc[df["Terms"] == "Quadratic Terms", "Included"].iloc[0]
    assert quad_terms == ["A^2"]


def test_get_model_term_count(dummy_design):
    class MS:
        model_terms = 4

    dummy_design._model_spec = MS()

    assert dummy_design.get_model_term_count() == 4


def test_get_model_term_count_requires_model_spec(dummy_design):
    dummy_design._model_spec = None

    with pytest.raises(ValueError, match="No model specification defined"):
        dummy_design.get_model_term_count()


def test_get_response_condition_summary(dummy_design):
    dummy_design._responses = pd.DataFrame({"y1": [1, 2], "y2": [3, 4]})
    dummy_design._response_conditions = {
        "y1": {"lower_limit": 0, "upper_limit": None, "maximize": True},
        "y2": {"lower_limit": None, "upper_limit": 10, "maximize": False},
    }

    df = dummy_design.get_response_condition_summary()

    assert set(df["Response"]) == {"y1", "y2"}
    assert "No" in df["Upper Limit"].values
    assert "No" in df["Lower Limit"].values
    assert "Maximize" in df["Goal"].values
    assert "Minimize" in df["Goal"].values


def test_get_response_condition_summary_requires_complete_conditions(dummy_design):
    dummy_design._responses = pd.DataFrame({"y": [1, 2]})
    dummy_design._response_conditions = {}

    with pytest.raises(ValueError, match="No response conditions"):
        dummy_design.get_response_condition_summary()


def test_getters_design_and_responses(dummy_design):
    dummy_design._design_matrix = pd.DataFrame({"A": [1]})
    dummy_design._coded_design_matrix = pd.DataFrame({"A": [0]})
    dummy_design._responses = pd.DataFrame({"y": [1.23]})

    assert dummy_design.get_design_matrix().equals(dummy_design._design_matrix)
    assert dummy_design.get_coded_design_matrix().equals(dummy_design._coded_design_matrix)
    assert dummy_design.get_responses().equals(dummy_design._responses)

    design_copy = dummy_design.get_design_matrix()
    design_copy.iloc[0, 0] = 99
    assert dummy_design._design_matrix.iloc[0, 0] == 1


def test_get_predicted_responses(dummy_design):
    dummy_design._response_list = ["y"]
    dummy_design._responses = pd.DataFrame({"y": [1.234, 2.5]})
    dummy_design._mlr_wrapper.results["y"] = DummyResult(y_hat=pd.Series([1.2, 2.6]))

    preds = dummy_design.get_predicted_responses()

    assert list(preds.columns) == ["y"]
    assert np.allclose(preds["y"], [1.2, 2.6], atol=1e-3)


def test_get_leverages(dummy_design):
    dummy_design._coded_design_matrix = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    dummy_design._compute_leverage = lambda points: np.array([0.1235, 0.33333])

    lev = dummy_design.get_leverages()
    
    assert list(lev.columns) == ["Leverage"]
    assert np.allclose(lev["Leverage"], [0.1235, 0.33333])


def test_get_coefficient_summary_and_model_matrix(dummy_design):
    coef_df = pd.DataFrame({"coef": [1.0, 2.0]}, index=["Intercept", "A"])
    dummy_design._mlr_wrapper.results["y"] = DummyResult(coef=coef_df)
    dummy_design._model_matrix = pd.DataFrame({"Intercept": [1], "A": [0.5]})

    assert dummy_design.get_coefficient_summary("y").equals(coef_df)
    assert dummy_design.get_model_matrix().equals(dummy_design._model_matrix)


def test_get_anova_summary_with_lof(dummy_design):
    anova = {
        "SS_tot": 10.0,
        "SS_reg": 8.0,
        "SS_res": 2.0,
        "df_tot": 9,
        "df_reg": 4,
        "df_res": 5,
        "MS_tot": 1.111,
        "MS_reg": 2.0,
        "MS_res": 0.4,
        "SS_pe": 0.6,
        "SS_lof": 1.4,
        "df_pe": 2,
        "df_lof": 3,
        "MS_pe": 0.3,
        "MS_lof": 0.4667,
    }
    dummy_design._mlr_wrapper.results["y"] = DummyResult(anova=anova)
    dummy_design._group_by_replicates = lambda: [[0, 1], [2]]

    df = dummy_design.get_anova_summary("y")

    assert set(df["Source"]) >= {"Total", "Regression", "Residuals", "Pure Error", "Lack of Fit"}
    assert np.isclose(df.loc[df["Source"] == "Regression", "SS"].iloc[0], 8.0)


def test_get_vif(dummy_design):
    vif = dummy_design.get_vif()
    assert "VIF" in vif.columns
    assert len(vif) == 2


def test_get_dispersion_matrix(dummy_design):
    disp = pd.DataFrame([[0, 1e-9], [1.23456, 0]])
    dummy_design._mlr_wrapper.results["y"] = DummyResult(dispersion_matrix=disp)

    out = dummy_design.get_dispersion_matrix("y", tol=1e-8)

    assert (out.values == np.array([[0.0, 0.0], [1.23456, 0.0]])).all()
    assert out.index.name is None


def test_get_replicate_summary(dummy_design):
    repl = {
        0: {"n_replicates": 2, "mean": 1.0, "var": 0.1, "std": 0.316, "dof": 1},
        1: {"n_replicates": 3, "mean": 2.0, "var": 0.2, "std": 0.447, "dof": 2},
    }
    dummy_design._mlr_wrapper.results["y"] = DummyResult(replicates=repl)

    df = dummy_design.get_replicate_summary("y")

    assert set(df.columns) == {"Run Index", "n*", "Mean", "Var", "Std Dev", "dof"}
    assert df.shape[0] == 2


def test_get_metric_summary(dummy_design):
    metrics = {
        "R2": 0.9123456789,
        "R2_adj": 0.9,
        "Q2": 0.85,
        "PRESS": 1.25,
        "RMSE": 0.5,
        "RMSE_CV": 0.6,
    }
    dummy_design._mlr_wrapper.results["y"] = DummyResult(metrics=metrics)

    df = dummy_design.get_metric_summary("y")

    assert list(df.columns) == ["R2", "R2_adj", "Q2", "PRESS", "RMSE", "RMSE_CV"]
    assert df.loc[0, "R2"] == metrics["R2"]
    assert "Q2_adj" not in df.columns

    df.loc[0, "R2"] = -1.0
    assert dummy_design._mlr_wrapper.results["y"].metrics["R2"] == metrics["R2"]


def test_model_and_lof_f_tests(monkeypatch, dummy_design):
    dummy_design._mlr_wrapper.results["y"] = DummyResult(anova={
        "MS_lof": 1.0, "MS_pe": 1.0, "df_lof": 1, "df_pe": 1,
    })

    def fake_model_f_test(resp, wrapper):
        return pd.DataFrame({"pvalue": [0.01]})

    def fake_lof_f_test(resp, wrapper):
        return pd.DataFrame({"pvalue": [0.2]})

    monkeypatch.setattr("doetools.utils.summary.RegressionAnalyzer.compute_model_f_test", fake_model_f_test)
    monkeypatch.setattr("doetools.utils.summary.RegressionAnalyzer.compute_lof_f_test", fake_lof_f_test)

    model_df = dummy_design.get_model_f_test("y")
    lof_df = dummy_design.get_lof_f_test("y")

    assert model_df.loc[0, "pvalue"] == 0.01
    assert lof_df.loc[0, "pvalue"] == 0.2


def test_get_lof_f_test_returns_empty_dataframe_when_not_applicable(dummy_design):
    dummy_design._mlr_wrapper.results["y"] = DummyResult(anova={})

    result = dummy_design.get_lof_f_test("y")

    assert result.empty
    assert list(result.columns) == ["F_value", "F_crit_95%", "F_crit_99%", "p_value"]


def test_get_lof_f_test_returns_empty_dataframe_with_insufficient_dof(dummy_design):
    dummy_design._mlr_wrapper.results["y"] = DummyResult(anova={
        "MS_lof": np.nan, "MS_pe": 1.0, "df_lof": 0, "df_pe": 2,
    })

    result = dummy_design.get_lof_f_test("y")

    assert result.empty
    assert list(result.columns) == ["F_value", "F_crit_95%", "F_crit_99%", "p_value"]

def test_model_summary_pdf(monkeypatch, dummy_design, tmp_path):
    called = {}

    class DummyPDF:
        def __init__(self, *args, **kwargs):
            called["init"] = True

        def build(self, filename):
            called["filename"] = filename

    monkeypatch.setattr("doetools.utils.summary.DesignReportPDF", DummyPDF)

    outfile = tmp_path / "report.pdf"
    dummy_design.get_model_summary_pdf(filename=str(outfile))

    assert called.get("init") is True
    assert called.get("filename") == str(outfile)
