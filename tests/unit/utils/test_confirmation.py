"""Tests for external OLS confirmation runs and PIMean intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from doetools import (
    CategoricalFactor,
    ContinuousFactor,
    FullFactorialDesign,
    MixtureFactor,
    ModelTerms,
    SimplexCentroidDesign,
)


def _fitted_process_design():
    design = FullFactorialDesign(
        {
            "A": ContinuousFactor(3, 0.0, 10.0, decimals=2),
            "B": ContinuousFactor(2, 20.0, 40.0, decimals=2),
        }
    )
    design.set_model_terms(
        ModelTerms(
            intercept=True,
            pro_main="all",
            pro_int2=None,
            pro_quadratic=None,
        )
    )
    coded = design._coded_design_matrix
    design._response_list = ["Yield", "Purity"]
    design._responses = pd.DataFrame(
        {
            "Yield": 10.0 + 2.0 * coded["A"] - coded["B"] + 0.3 * (coded.index % 3),
            "Purity": 80.0 - coded["A"] + 0.4 * (coded.index % 2),
        }
    )
    design.compute_mlr_model()
    return design


def _confirmation_frame():
    return pd.DataFrame(
        {
            "Exp. Order": [3, 1, 2],
            "A": [2.0, 2.004, 8.0],
            "B": [25.0, 25.0, 35.0],
            "Yield": [10.1, 10.3, 12.4],
            "Purity": [80.2, 80.4, 79.3],
            "Lab Note": ["a", "b", "c"],
        }
    )


def test_load_groups_runs_and_getters_return_defensive_copies(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "confirmation.xlsx"
    _confirmation_frame().to_excel(path, index=False)

    design.load_confirmation_runs(path)

    points = design.get_confirmation_points()
    responses = design.get_confirmation_responses()
    assert list(points.columns) == ["A", "B"]
    assert points.loc[0, "A"] == points.loc[1, "A"] == 2.0
    assert list(responses.columns) == ["Yield", "Purity"]
    assert list(design.get_confirmation_responses("Yield").columns) == ["Yield"]

    points.loc[0, "A"] = 999.0
    responses.loc[0, "Yield"] = 999.0
    assert design.get_confirmation_points().loc[0, "A"] == 2.0
    assert design.get_confirmation_responses().loc[0, "Yield"] == 10.1

    results = design.get_confirmation_results("Yield")
    assert list(results.columns) == [
        "Setting", "A", "B", "n", "Observed", "Predicted", "Residual",
        "PI Lower", "PI Upper", "Within PI",
    ]
    assert list(results["Setting"]) == [1, 2]
    assert list(results["n"]) == [2, 1]
    assert np.isclose(results.loc[0, "Observed"], 10.2)
    assert np.allclose(
        results["Residual"], results["Observed"] - results["Predicted"]
    )


def test_pimean_matches_independent_statsmodels_calculation(tmp_path):
    design = _fitted_process_design()
    frame = _confirmation_frame()
    path = tmp_path / "confirmation.csv"
    frame.to_csv(path, index=False)
    design.load_confirmation_runs(path)

    alpha = 0.1
    results = design.get_confirmation_results("Yield", alpha=alpha)
    fitted = design._mlr_wrapper.results["Yield"]
    unique_coded = design.get_confirmation_points(coded=True).iloc[[0, 2]].reset_index(drop=True)
    x0 = design._build_model_matrix(unique_coded, design._model_spec).to_numpy(dtype=float)
    covariance = np.asarray(fitted.model.normalized_cov_params, dtype=float)
    leverage = np.einsum("ij,jk,ik->i", x0, covariance, x0)
    counts = np.array([2.0, 1.0])
    expected_half_width = t.ppf(
        1 - alpha / 2, fitted.model.df_resid
    ) * np.sqrt(fitted.model.mse_resid * (1 / counts + leverage))

    assert np.allclose(
        results["PI Upper"] - results["Predicted"], expected_half_width
    )
    assert np.allclose(
        results["Predicted"] - results["PI Lower"], expected_half_width
    )
    assert results["Within PI"].dtype == bool


def test_actual_and_coded_loads_are_equivalent(tmp_path):
    actual_design = _fitted_process_design()
    coded_design = _fitted_process_design()
    frame = _confirmation_frame().drop(columns=["Exp. Order", "Lab Note"])
    actual_path = tmp_path / "actual.csv"
    coded_path = tmp_path / "coded.csv"
    frame.to_csv(actual_path, index=False)

    coded_frame = frame.copy()
    coded_frame[["A", "B"]] = coded_design._code_matrix(frame[["A", "B"]])
    coded_frame.to_csv(coded_path, index=False)

    actual_design.load_confirmation_runs(actual_path, coded=False)
    coded_design.load_confirmation_runs(coded_path, coded=True)

    pd.testing.assert_frame_equal(
        actual_design.get_confirmation_points(),
        coded_design.get_confirmation_points(),
    )
    pd.testing.assert_frame_equal(
        actual_design.get_confirmation_results("Yield"),
        coded_design.get_confirmation_results("Yield"),
    )


def test_failed_load_preserves_existing_confirmation_data(tmp_path):
    design = _fitted_process_design()
    valid_path = tmp_path / "valid.csv"
    invalid_path = tmp_path / "invalid.csv"
    _confirmation_frame().to_csv(valid_path, index=False)
    _confirmation_frame().drop(columns="Purity").to_csv(invalid_path, index=False)
    design.load_confirmation_runs(valid_path)
    original = design.get_confirmation_points()

    with pytest.raises(ValueError, match="Response columns not found"):
        design.load_confirmation_runs(invalid_path)

    pd.testing.assert_frame_equal(design.get_confirmation_points(), original)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("A", np.nan, "missing or non-finite"),
        ("A", 100.0, "outside the allowed domain"),
        ("Yield", np.inf, "missing or non-finite"),
    ],
)
def test_invalid_numeric_confirmation_values_are_rejected(
    tmp_path, column, value, message
):
    design = _fitted_process_design()
    frame = _confirmation_frame()
    frame.loc[0, column] = value
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        design.load_confirmation_runs(path)


def test_domain_filters_are_applied_in_actual_units(tmp_path):
    design = _fitted_process_design()
    design.set_domain_filters([lambda frame: frame["A"] + frame["B"] <= 30.0])
    path = tmp_path / "filtered.csv"
    _confirmation_frame().to_csv(path, index=False)

    with pytest.raises(ValueError, match="outside the configured domain"):
        design.load_confirmation_runs(path)


def test_missing_factor_and_unfitted_model_are_rejected(tmp_path):
    frame = _confirmation_frame()
    path = tmp_path / "confirmation.csv"
    frame.to_csv(path, index=False)
    unfitted = FullFactorialDesign(
        {"A": ContinuousFactor(2, 0.0, 1.0), "B": ContinuousFactor(2, 0.0, 1.0)}
    )
    with pytest.raises(ValueError, match="compute_mlr_model"):
        unfitted.load_confirmation_runs(path)

    design = _fitted_process_design()
    frame.drop(columns="B").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Factor columns not found"):
        design.load_confirmation_runs(path)


def test_categorical_actual_and_coded_levels_are_validated(tmp_path):
    design = FullFactorialDesign(
        {
            "A": ContinuousFactor(3, 0.0, 10.0),
            "Type": CategoricalFactor(["Low", "High"]),
        }
    )
    design.set_model_terms(
        ModelTerms(
            intercept=True,
            pro_main="all",
            pro_int2=None,
            pro_quadratic=None,
        )
    )
    coded = design._coded_design_matrix
    design._response_list = ["Yield"]
    design._responses = pd.DataFrame(
        {"Yield": 5 + coded["A"] + 2 * coded["Type"] + 0.1 * coded.index}
    )
    design.compute_mlr_model()
    path = tmp_path / "categorical.csv"

    pd.DataFrame({"A": [5.0], "Type": ["Other"], "Yield": [6.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="Unknown level"):
        design.load_confirmation_runs(path)

    pd.DataFrame({"A": [0.0], "Type": [0.25], "Yield": [6.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="Unknown coded level"):
        design.load_confirmation_runs(path, coded=True)

    pd.DataFrame({"A": [0.0, 0.0], "Type": [-1.0, 1.0], "Yield": [4.0, 8.0]}).to_csv(
        path, index=False
    )
    design.load_confirmation_runs(path, coded=True)
    assert list(design.get_confirmation_points()["Type"]) == ["Low", "High"]


def test_duplicate_columns_are_rejected(monkeypatch):
    design = _fitted_process_design()
    duplicate = pd.DataFrame(
        [[1.0, 2.0, 25.0, 10.0, 80.0]],
        columns=["A", "A", "B", "Yield", "Purity"],
    )
    monkeypatch.setattr(design, "upload_file", lambda _: duplicate)
    with pytest.raises(ValueError, match="duplicate columns"):
        design.load_confirmation_runs("unused.csv")


def test_mixture_bounds_and_sum_are_validated(tmp_path):
    design = SimplexCentroidDesign(
        {
            name: MixtureFactor(0.0, 1.0, decimals=3)
            for name in ("A", "B", "C")
        }
    )
    design.set_model_terms(
        ModelTerms(
            intercept=False,
            pro_main=None,
            pro_int2=None,
            pro_quadratic=None,
            mix_main="all",
        )
    )
    matrix = design._coded_design_matrix
    design._response_list = ["Yield"]
    design._responses = pd.DataFrame(
        {"Yield": 2 * matrix["A"] + 3 * matrix["B"] + 5 * matrix["C"] + 0.1 * matrix.index}
    )
    design.compute_mlr_model()
    path = tmp_path / "mixture.csv"
    pd.DataFrame(
        {"A": [0.2], "B": [0.2], "C": [0.2], "Yield": [3.0]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="must sum to 1"):
        design.load_confirmation_runs(path)


def test_clear_removes_only_confirmation_state(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "confirmation.csv"
    _confirmation_frame().to_csv(path, index=False)
    design.load_confirmation_runs(path)
    fitted_model = design._mlr_wrapper
    training_responses = design._responses.copy()

    design.clear_confirmation_runs()

    with pytest.raises(ValueError, match="No confirmation runs"):
        design.get_confirmation_points()
    assert design._mlr_wrapper is fitted_model
    pd.testing.assert_frame_equal(design._responses, training_responses)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan])
def test_invalid_alpha_is_rejected(tmp_path, alpha):
    design = _fitted_process_design()
    path = tmp_path / "confirmation.csv"
    _confirmation_frame().to_csv(path, index=False)
    design.load_confirmation_runs(path)
    with pytest.raises(ValueError, match="alpha must be between"):
        design.get_confirmation_results("Yield", alpha=alpha)


def test_saturated_model_cannot_compute_pimean(tmp_path):
    design = FullFactorialDesign(
        {
            "A": ContinuousFactor(2, 0.0, 1.0),
            "B": ContinuousFactor(2, 0.0, 1.0),
        }
    )
    design.set_model_terms(
        ModelTerms(
            intercept=True,
            pro_main="all",
            pro_int2="all",
            pro_quadratic=None,
        )
    )
    design._response_list = ["Yield"]
    design._responses = pd.DataFrame({"Yield": [1.0, 2.0, 3.0, 4.0]})
    design.compute_mlr_model()
    path = tmp_path / "confirmation.csv"
    pd.DataFrame({"A": [0.5], "B": [0.5], "Yield": [2.5]}).to_csv(path, index=False)
    design.load_confirmation_runs(path)

    with pytest.raises(ValueError, match="PIMean is unavailable"):
        design.get_confirmation_results("Yield")


def test_confirmation_plots_use_grouped_results(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "confirmation.csv"
    _confirmation_frame().to_csv(path, index=False)
    design.load_confirmation_runs(path)
    results = design.get_confirmation_results("Yield")

    predicted = design.plot_confirmation(
        "Yield", view="observed_vs_predicted"
    )
    residuals = design.plot_confirmation("Yield", view="residuals")

    assert len(predicted.data) == len(residuals.data) == 2
    assert np.allclose(predicted.data[0].x, results["Observed"])
    assert np.allclose(predicted.data[0].y, results["Predicted"])
    assert np.allclose(predicted.data[1].x, predicted.data[1].y)
    assert np.allclose(residuals.data[0].x, results["Observed"])
    assert np.allclose(residuals.data[0].y, results["Residual"])
    assert np.allclose(residuals.data[1].y, 0.0)
    assert "Confirmation" in predicted.layout.title.text
    assert "Observed Mean" in residuals.layout.xaxis.title.text


def test_confirmation_plot_rejects_unknown_view(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "confirmation.csv"
    _confirmation_frame().to_csv(path, index=False)
    design.load_confirmation_runs(path)

    with pytest.raises(ValueError, match="view must be either"):
        design.plot_confirmation("Yield", view="unknown")
