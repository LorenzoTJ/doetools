"""Tests for predictions at externally loaded factor settings."""

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


def _prediction_frame():
    return pd.DataFrame(
        {
            "A": [2.0, 8.0, 2.0],
            "B": [25.0, 35.0, 25.0],
            "Comment": ["first", "second", "duplicate"],
        },
        index=[10, 20, 30],
    )


def test_load_xlsx_and_getters_preserve_points_and_return_copies(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "prediction.xlsx"
    _prediction_frame().to_excel(path, index=False)

    design.load_prediction_points(path)

    actual = design.get_prediction_points()
    coded = design.get_prediction_points(coded=True)
    assert list(actual.columns) == ["A", "B"]
    assert list(actual.index) == [0, 1, 2]
    assert actual.iloc[0].equals(actual.iloc[2])
    assert np.allclose(coded["A"], [-0.6, 0.6, -0.6])
    assert np.allclose(coded["B"], [-0.5, 0.5, -0.5])

    actual.loc[0, "A"] = 999.0
    coded.loc[0, "A"] = 999.0
    assert design.get_prediction_points().loc[0, "A"] == 2.0
    assert np.isclose(design.get_prediction_points(coded=True).loc[0, "A"], -0.6)


def test_loader_accepts_dataframe_and_preserves_source_index():
    design = _fitted_process_design()
    frame = _prediction_frame()

    design.load_prediction_points(source=frame)
    frame.loc[10, "A"] = 999.0

    assert list(design.get_prediction_points().index) == [10, 20, 30]
    assert design.get_prediction_points().loc[10, "A"] == 2.0
    assert list(design.get_prediction_results("Yield").index) == [10, 20, 30]


def test_prediction_results_match_independent_mean_ci_calculation(tmp_path):
    design = _fitted_process_design()
    frame = _prediction_frame().drop(columns="Comment")
    path = tmp_path / "prediction.csv"
    frame.to_csv(path, index=False)
    design.load_prediction_points(path)

    alpha = 0.1
    results = design.get_prediction_results("Yield", alpha=alpha)
    fitted = design._mlr_wrapper.results["Yield"]
    coded = design.get_prediction_points(coded=True)
    x0 = design._build_model_matrix(coded, design._model_spec)
    expected_prediction = np.asarray(fitted.model.predict(x0), dtype=float)
    covariance = np.asarray(fitted.model.normalized_cov_params, dtype=float)
    x_values = x0.to_numpy(dtype=float)
    leverage = np.einsum("ij,jk,ik->i", x_values, covariance, x_values)
    half_width = t.ppf(1 - alpha / 2, fitted.model.df_resid) * np.sqrt(
        fitted.model.mse_resid * leverage
    )

    assert list(results.columns) == [
        "A", "B", "Predicted", "CI Lower", "CI Upper"
    ]
    assert np.allclose(results["Predicted"], expected_prediction)
    assert np.allclose(results["CI Lower"], expected_prediction - half_width)
    assert np.allclose(results["CI Upper"], expected_prediction + half_width)


def test_coded_result_changes_only_factor_representation(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "prediction.csv"
    _prediction_frame().to_csv(path, index=False)
    design.load_prediction_points(path)

    actual = design.get_prediction_results("Purity")
    coded = design.get_prediction_results("Purity", coded=True)

    assert not np.allclose(actual[["A", "B"]], coded[["A", "B"]])
    assert np.allclose(
        actual[["Predicted", "CI Lower", "CI Upper"]],
        coded[["Predicted", "CI Lower", "CI Upper"]],
    )


def test_actual_and_coded_loads_are_equivalent(tmp_path):
    actual_design = _fitted_process_design()
    coded_design = _fitted_process_design()
    actual = _prediction_frame().drop(columns="Comment")
    coded = actual.copy()
    coded[["A", "B"]] = coded_design._code_matrix(actual[["A", "B"]])
    actual_path = tmp_path / "actual.csv"
    coded_path = tmp_path / "coded.csv"
    actual.to_csv(actual_path, index=False)
    coded.to_csv(coded_path, index=False)

    actual_design.load_prediction_points(actual_path)
    coded_design.load_prediction_points(coded_path, coded=True)

    pd.testing.assert_frame_equal(
        actual_design.get_prediction_points(),
        coded_design.get_prediction_points(),
    )
    pd.testing.assert_frame_equal(
        actual_design.get_prediction_results("Yield"),
        coded_design.get_prediction_results("Yield"),
    )


def test_results_use_the_current_refitted_model(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "prediction.csv"
    _prediction_frame().to_csv(path, index=False)
    design.load_prediction_points(path)
    first = design.get_prediction_results("Yield")

    design._responses["Yield"] = design._responses["Yield"] + 20.0
    design.compute_mlr_model()
    second = design.get_prediction_results("Yield")

    assert np.allclose(second["Predicted"] - first["Predicted"], 20.0)
    pd.testing.assert_frame_equal(first[["A", "B"]], second[["A", "B"]])


def test_failed_load_preserves_existing_prediction_points(tmp_path):
    design = _fitted_process_design()
    valid_path = tmp_path / "valid.csv"
    invalid_path = tmp_path / "invalid.csv"
    frame = _prediction_frame()
    frame.to_csv(valid_path, index=False)
    frame.drop(columns="B").to_csv(invalid_path, index=False)
    design.load_prediction_points(valid_path)
    original = design.get_prediction_points()

    with pytest.raises(ValueError, match="Factor columns not found"):
        design.load_prediction_points(invalid_path)

    pd.testing.assert_frame_equal(design.get_prediction_points(), original)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("A", np.nan, "missing or non-finite"),
        ("A", 100.0, "outside the allowed domain"),
        ("B", "invalid", "numeric values only"),
    ],
)
def test_invalid_continuous_values_are_rejected(
    tmp_path, column, value, message
):
    design = _fitted_process_design()
    frame = _prediction_frame()
    if isinstance(value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[10, column] = value
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        design.load_prediction_points(path)


def test_duplicate_columns_are_rejected(monkeypatch):
    design = _fitted_process_design()
    duplicate = pd.DataFrame(
        [[2.0, 3.0, 25.0]],
        columns=["A", "A", "B"],
    )
    monkeypatch.setattr(design, "upload_file", lambda _: duplicate)

    with pytest.raises(ValueError, match="duplicate columns"):
        design.load_prediction_points("unused.csv")


def test_categorical_mixture_and_domain_constraints_are_shared(tmp_path):
    categorical = FullFactorialDesign(
        {
            "A": ContinuousFactor(3, 0.0, 10.0),
            "Type": CategoricalFactor(["Low", "High"]),
        }
    )
    categorical.set_domain_filters([lambda frame: frame["A"] <= 8.0])
    path = tmp_path / "external.csv"
    pd.DataFrame({"A": [5.0], "Type": ["Other"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unknown level"):
        categorical.load_prediction_points(path)
    pd.DataFrame({"A": [9.0], "Type": ["Low"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="outside the configured domain"):
        categorical.load_prediction_points(path)

    mixture = SimplexCentroidDesign(
        {name: MixtureFactor(0.0, 1.0) for name in ("A", "B", "C")}
    )
    pd.DataFrame({"A": [0.2], "B": [0.2], "C": [0.2]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="must sum to 1"):
        mixture.load_prediction_points(path)


def test_missing_model_response_and_invalid_arguments_are_clear(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "prediction.csv"
    _prediction_frame().to_csv(path, index=False)

    with pytest.raises(ValueError, match="No prediction points"):
        design.get_prediction_results("Yield")

    unfitted = FullFactorialDesign({"A": ContinuousFactor(2, 0.0, 1.0)})
    unfitted_path = tmp_path / "unfitted.csv"
    pd.DataFrame({"A": [0.5]}).to_csv(unfitted_path, index=False)
    unfitted.load_prediction_points(unfitted_path)
    with pytest.raises(ValueError, match="compute_mlr_model"):
        unfitted.get_prediction_results("Yield")

    design.load_prediction_points(path)
    with pytest.raises(ValueError, match="not available for prediction"):
        design.get_prediction_results("Unknown")
    with pytest.raises(TypeError, match="alpha must be a numeric"):
        design.get_prediction_results("Yield", alpha=True)
    with pytest.raises(ValueError, match="alpha must be between"):
        design.get_prediction_results("Yield", alpha=1.0)
    with pytest.raises(TypeError, match="coded must be a boolean"):
        design.get_prediction_results("Yield", coded=1)
    with pytest.raises(TypeError, match="coded must be a boolean"):
        design.load_prediction_points(source=path, coded=1)


def test_saturated_model_returns_predictions_without_confidence_intervals(tmp_path):
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
    path = tmp_path / "prediction.csv"
    pd.DataFrame({"A": [0.5], "B": [0.5]}).to_csv(path, index=False)
    design.load_prediction_points(path)

    results = design.get_prediction_results("Yield")

    assert np.isfinite(results["Predicted"]).all()
    assert results["CI Lower"].isna().all()
    assert results["CI Upper"].isna().all()


def test_clear_and_breaking_api_surface(tmp_path):
    design = _fitted_process_design()
    path = tmp_path / "prediction.csv"
    _prediction_frame().to_csv(path, index=False)
    design.load_prediction_points(path)
    fitted_model = design._mlr_wrapper

    design.clear_prediction_points()

    with pytest.raises(ValueError, match="No prediction points"):
        design.get_prediction_points()
    assert design._mlr_wrapper is fitted_model
    assert not hasattr(design, "predict")
    assert not hasattr(design, "get_predicted_responses")
    assert hasattr(design, "get_fitted_values")
