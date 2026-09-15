"""Independent numerical references for interval and error definitions."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from doetools.utils.intervals import interval_half_width, validate_interval_options
from doetools.utils.regression import RegressionAnalyzer
from doetools.utils.summary import DesignSummaryMixin
from types import SimpleNamespace


@pytest.fixture
def fit():
    x = np.repeat([-1.0, 0.0, 1.0], 3)
    matrix = pd.DataFrame({"Int": np.ones(9), "x": x})
    y = pd.Series(2 + x + np.array([0, .2, -.1, .4, -.2, .1, -.3, .2, .1]))
    return RegressionAnalyzer()._fit_single_response(
        matrix, y, replicate_groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    )


@pytest.mark.parametrize("interval,obs", [("confidence", False), ("prediction", True)])
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_intervals_match_statsmodels(fit, interval, obs, alpha):
    points = pd.DataFrame({"Int": [1., 1., 1.], "x": [-.7, 0., 1.4]})
    expected = fit.model.get_prediction(points).conf_int(obs=obs, alpha=alpha)
    actual = interval_half_width(fit, points, interval=interval, alpha=alpha)
    np.testing.assert_allclose(actual, (expected[:, 1] - expected[:, 0]) / 2)
    # Reordering input columns must not change which coefficients they multiply.
    np.testing.assert_allclose(actual, interval_half_width(
        fit, points[["x", "Int"]], interval=interval, alpha=alpha,
    ))


@pytest.mark.parametrize("interval,extra", [("confidence", 0), ("prediction", 1)])
def test_pure_error_uses_its_own_variance_and_df(fit, interval, extra):
    points = pd.DataFrame({"Int": [1., 1.], "x": [0., .75]})
    # The symmetric training design has X'X = diag(9, 6).
    h = 1 / 9 + points.x.to_numpy() ** 2 / 6
    expected = t.ppf(.975, 6) * np.sqrt(fit.anova["SS_pe"] / 6 * (h + extra))
    np.testing.assert_allclose(interval_half_width(
        fit, points, interval=interval, variance_source="pure_error",
    ), expected)


@pytest.mark.parametrize("variant", ["no_intercept", "rank_deficient", "constant", "saturated"])
def test_rmse_and_anova_sd_use_distinct_denominators(variant):
    x = np.arange(6, dtype=float)
    matrix = pd.DataFrame({"Int": np.ones(6), "x": x})
    y = pd.Series([1., 2.4, 2.8, 4.2, 5., 5.7])
    if variant == "no_intercept":
        matrix = matrix[["x"]]
    elif variant == "rank_deficient":
        matrix["twice_x"] = 2 * x
    elif variant == "constant":
        y[:] = 3.
    else:
        matrix = pd.DataFrame(np.eye(6))
    fitted = RegressionAnalyzer()._fit_single_response(matrix, y, replicate_groups=[])
    np.testing.assert_allclose(fitted.metrics["RMSE"], np.sqrt(np.mean(fitted.residuals ** 2)))
    assert fitted.anova["df_res"] == len(y) - np.linalg.matrix_rank(matrix)
    summary = DesignSummaryMixin()
    summary._mlr_wrapper = SimpleNamespace(results={"y": fitted})
    table = summary.get_anova_summary("y")
    np.testing.assert_allclose(table.SD, np.sqrt(table.MS), equal_nan=True)
    assert set(fitted.metrics) == {"R2", "R2_adj", "Q2", "PRESS", "RMSE", "RMSE_CV"}
    width = interval_half_width(fitted, matrix, interval="prediction")
    if variant == "saturated":
        assert np.isnan(width).all()
        with pytest.raises(ValueError, match="residual degrees"):
            interval_half_width(fitted, matrix, require_estimable=True)
    else:
        expected = fitted.model.get_prediction(matrix).conf_int(obs=True)
        np.testing.assert_allclose(width, (expected[:, 1] - expected[:, 0]) / 2, atol=1e-14)


def test_anova_sd_includes_pure_error_and_lof(fit):
    summary = DesignSummaryMixin()
    summary._mlr_wrapper = SimpleNamespace(results={"y": fit})
    table = summary.get_anova_summary("y")
    assert list(table.columns) == ["Source", "SS", "df", "MS", "SD"]
    assert list(table.Source) == ["Total", "Regression", "Residuals", "Pure Error", "Lack of Fit"]
    np.testing.assert_allclose(table.SD, np.sqrt(table.MS))


def test_zero_variance_and_invalid_leverage(fit):
    points = pd.DataFrame({"Int": [1.], "x": [0.]})
    fit.anova["MS_res"] = 0.
    np.testing.assert_array_equal(interval_half_width(fit, points, interval="prediction"), [0.])
    fit.dispersion_matrix = np.diag([-1e-16, 1.])
    np.testing.assert_array_equal(interval_half_width(fit, points), [0.])
    fit.dispersion_matrix = np.diag([-0.1, 1.])
    with pytest.raises(ValueError, match="negative"):
        interval_half_width(fit, points)
    fit.dispersion_matrix = np.diag([np.nan, 1.])
    with pytest.raises(ValueError, match="finite"):
        interval_half_width(fit, points)


@pytest.mark.parametrize("change", [{"df_pe": 0}, {"MS_pe": np.nan}, {"MS_pe": -1}])
def test_invalid_pure_error_never_falls_back_to_residuals(fit, change):
    fit = deepcopy(fit)
    fit.anova.update(change)
    with pytest.raises(ValueError, match="Pure-error"):
        interval_half_width(fit, pd.DataFrame({"Int": [1.], "x": [0.]}), variance_source="pure_error")


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        validate_interval_options("prediction", "residuals", alpha)


@pytest.mark.parametrize("alpha", [True, np.bool_(False), "0.05", 1j])
def test_nonnumeric_alpha(alpha):
    with pytest.raises(TypeError, match="alpha"):
        validate_interval_options("confidence", "residuals", alpha)


@pytest.mark.parametrize("interval,variance", [(None, "residuals"), ("PI", "residuals"), ("prediction", "replicates")])
def test_invalid_interval_options(interval, variance):
    with pytest.raises(ValueError):
        validate_interval_options(interval, variance, .05)
