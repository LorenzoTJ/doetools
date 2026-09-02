import numpy as np
import pandas as pd
import pytest

from doetools import FullFactorialDesign, SimplexCentroidDesign
from doetools.utils.factors import (
    CategoricalFactor,
    ContinuousFactor,
    MixtureFactor,
)
from doetools.utils.model_spec import ModelTerms


def _fitted_process_design(reference_level=None):
    design = FullFactorialDesign(
        {
            "A": ContinuousFactor(
                n_levels=3, lower_bound=0, upper_bound=10
            ),
            "B": ContinuousFactor(
                n_levels=2, lower_bound=20, upper_bound=40
            ),
            "Type": CategoricalFactor(
                levels=["Low", "High"], reference_level=reference_level
            ),
        }
    )
    # Make the design deliberately unbalanced.
    design._coded_design_matrix = (
        design._coded_design_matrix.iloc[:-1].reset_index(drop=True)
    )
    design._design_matrix = design._decode_matrix(
        design._coded_design_matrix
    )
    design.set_model_terms(
        ModelTerms(
            intercept=True,
            pro_main="all",
            pro_int2=[("A", "B"), ("A", "Type")],
            pro_quadratic=None,
        )
    )
    matrix = design._coded_design_matrix
    design._response_list = ["Yield", "Purity"]
    design._responses = pd.DataFrame(
        {
            "Yield": (
                12
                + 2.0 * matrix["A"]
                - 1.2 * matrix["B"]
                + 0.8 * matrix["Type"]
                + 2.5 * matrix["A"] * matrix["B"]
                + 0.5 * (matrix.index % 3)
            ),
            "Purity": (
                80
                - 1.5 * matrix["A"]
                + 2.0 * matrix["Type"]
                + 0.2 * (matrix.index % 2)
            ),
        }
    )
    design.compute_mlr_model()
    return design


def _fitted_mixture_design():
    design = SimplexCentroidDesign(
        {
            name: MixtureFactor(lower_bound=0.0, upper_bound=1.0)
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
            mix_int2="all",
        )
    )
    matrix = design._coded_design_matrix
    design._response_list = ["Yield"]
    design._responses = pd.DataFrame(
        {
            "Yield": (
                2 * matrix["A"]
                + 3 * matrix["B"]
                + 5 * matrix["C"]
                + 4 * matrix["A"] * matrix["B"]
            )
        }
    )
    design.compute_mlr_model()
    return design


@pytest.mark.parametrize(
    "method_name",
    [
        "plot_regression_coefficients",
        "plot_exp_vs_pred",
        "plot_residuals",
        "plot_model_diagnostics",
    ],
)
def test_multi_response_diagnostics_keep_dropdown(method_name):
    design = _fitted_process_design()

    figure = getattr(design, method_name)()

    assert len(figure.layout.updatemenus) == 1
    labels = [
        button.label
        for button in figure.layout.updatemenus[0].buttons
    ]
    assert labels == ["Yield", "Purity"]


@pytest.mark.parametrize(
    "method_name",
    [
        "plot_regression_coefficients",
        "plot_exp_vs_pred",
        "plot_residuals",
        "plot_model_diagnostics",
        "plot_residuals_vs_fitted",
        "plot_qq_residuals",
        "plot_residuals_histogram",
    ],
)
def test_response_specific_diagnostics_have_no_dropdown(method_name):
    design = _fitted_process_design()

    figure = getattr(design, method_name)("Yield")

    assert len(figure.layout.updatemenus or []) == 0
    assert "Yield" in str(figure.layout.title.text)


def test_cv_diagnostics_use_predictions_stored_by_regression_result():
    design = _fitted_process_design()
    result = design._mlr_wrapper.results["Yield"]
    sentinel = np.linspace(101.0, 102.0, len(design._responses))
    result.y_hat_cv = sentinel
    result.residuals_cv = design._responses["Yield"].to_numpy() - sentinel

    predicted_figure = design.plot_exp_vs_pred("Yield", cv=True)
    residual_figure = design.plot_residuals("Yield", cv=True, x_axis="sequence")

    assert np.allclose(np.asarray(predicted_figure.data[0].y, dtype=float), sentinel)
    assert np.allclose(
        np.asarray(residual_figure.data[0].y, dtype=float),
        result.residuals_cv,
    )


def test_residuals_by_run_uses_imported_experimental_order():
    design = _fitted_process_design()
    n_runs = len(design._responses)
    exp_order = np.arange(n_runs)[::-1] + 10
    design._experimental_metadata = pd.DataFrame(
        {
            "Exp. Order": exp_order,
            "Exp. Idx": np.arange(n_runs),
        }
    )

    figure = design.plot_residuals_by_run("Yield")

    assert np.array_equal(np.asarray(figure.data[0].x), exp_order)
    assert len(figure.data[0].x) == len(figure.data[0].y) == n_runs
    assert figure.layout.xaxis.title.text == "<b>Experimental Order</b>"


def test_residuals_by_run_falls_back_to_one_based_sequence():
    design = _fitted_process_design()
    design._experimental_metadata = None

    figure = design.plot_residuals_by_run("Yield")

    assert np.array_equal(
        np.asarray(figure.data[0].x),
        np.arange(1, len(design._responses) + 1),
    )
    assert figure.layout.xaxis.title.text == "<b>Experimental Run</b>"


def test_main_effect_is_model_based_on_dense_prediction_grid():
    design = _fitted_process_design()

    figure = design.plot_main_effect(
        "Yield", "A", coded=True, n_points=27
    )

    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 27
    assert np.isclose(figure.data[0].x[0], -1.0)
    assert np.isclose(figure.data[0].x[-1], 1.0)
    assert "Predicted Yield" in figure.data[0].hovertemplate


def test_categorical_main_effect_uses_available_levels():
    design = _fitted_process_design()

    figure = design.plot_main_effect(
        "Yield", "Type", coded=False, n_points=20
    )

    assert list(figure.data[0].x) == ["Low", "High"]
    assert figure.data[0].mode == "lines+markers"


def test_model_profile_base_uses_categorical_reference():
    design = _fitted_process_design(reference_level="High")

    baseline = design._base_prediction_row()

    assert baseline["Type"] == 1.0


def test_continuous_interaction_uses_three_fitted_profiles():
    design = _fitted_process_design()

    figure = design.plot_interaction(
        "Yield", "A", "B", coded=True, n_points=21
    )

    assert len(figure.data) == 3
    assert all(len(trace.x) == 21 for trace in figure.data)
    assert all("Predicted Yield" in trace.hovertemplate for trace in figure.data)


def test_interactions_collection_has_one_dropdown_entry_per_pair():
    design = _fitted_process_design()

    figure = design.plot_interactions("Yield", n_points=12)

    assert len(figure.layout.updatemenus) == 1
    assert len(figure.layout.updatemenus[0].buttons) == 3


@pytest.mark.parametrize("method_name", ["plot_main_effects", "plot_interactions"])
def test_classical_effect_plots_reject_mixture_designs(method_name):
    design = _fitted_mixture_design()

    with pytest.raises(ValueError, match="plot_mixture_trace"):
        getattr(design, method_name)("Yield")


def test_mixture_trace_uses_valid_bound_respecting_paths():
    design = _fitted_mixture_design()
    captured_grids = []
    original_predict = design.predict

    def capturing_predict(matrix_to_pred, responses):
        captured_grids.append(matrix_to_pred.copy())
        return original_predict(matrix_to_pred, responses)

    design.predict = capturing_predict
    figure = design.plot_mixture_trace("Yield", n_points=19)

    assert [trace.name for trace in figure.data] == ["A", "B", "C"]
    assert all(len(trace.x) == 19 for trace in figure.data)
    for grid in captured_grids[:3]:
        mixture = grid[["A", "B", "C"]].to_numpy(dtype=float)
        assert np.allclose(mixture.sum(axis=1), 1.0)
        assert np.all(mixture >= -1e-12)
        assert np.all(mixture <= 1.0 + 1e-12)


@pytest.mark.parametrize("response", [None, "Yield"])
def test_regression_coefficients_keep_first_mixture_component_without_intercept(response):
    design = _fitted_mixture_design()

    figure = design.plot_regression_coefficients(response)

    plotted_terms = list(figure.data[0].x)
    assert "A" in plotted_terms
    assert "B" in plotted_terms
    assert "C" in plotted_terms
