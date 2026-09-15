"""Prediction bounds propagate through process and mixture surfaces."""

import inspect

import numpy as np
import pandas as pd
import pytest

from doetools import ContinuousFactor, FullFactorialDesign, MixtureFactor, ModelTerms, SimplexCentroidDesign


def fitted_design(mixture=False):
    if mixture:
        design = SimplexCentroidDesign({name: MixtureFactor(0., 1.) for name in ("A", "B", "C")})
        terms = ModelTerms(intercept=False, mix_main="all", pro_main=None)
    else:
        design = FullFactorialDesign({name: ContinuousFactor(3, -1., 1.) for name in ("A", "B")})
        terms = ModelTerms(intercept=True, pro_main="all", pro_int2=None, pro_quadratic=None)
    design.add_replicates(type="all", n_replicates=1)
    design.set_model_terms(terms)
    coded = design._coded_design_matrix
    noise = .2 * np.cos(np.arange(len(coded)))
    design._response_list = ["Yield", "Purity"]
    design._responses = pd.DataFrame({
        "Yield": 5 + coded.A + 2 * coded.B + noise,
        "Purity": 8 - coded.A + coded.B + 2 * noise,
    })
    design.compute_mlr_model()
    design.set_response_conditions(
        lower_limits=[4.5, None],
        upper_limits=[None, 8.5], maximize=[True, False],
    )
    return design


@pytest.mark.parametrize("mixture", [False, True])
@pytest.mark.parametrize("interval", ["confidence", "prediction"])
@pytest.mark.parametrize("variance_source", ["residuals", "pure_error"])
def test_graph_bounds_and_external_predictions_agree(mixture, interval, variance_source):
    design = fitted_design(mixture)
    options = dict(ax3="C" if mixture else None, resolution=7)
    grid, _ = design._create_grid("A", "B", options["ax3"], {}, resolution=7)
    design.load_prediction_points(grid, coded=True)
    grid = design.get_prediction_points(coded=True)
    table = design.get_prediction_results("Yield", interval=interval, variance_source=variance_source)
    prefix = "CI" if interval == "confidence" else "PI"
    half = design._calculate_interval(grid, "Yield", interval=interval, variance_source=variance_source)
    np.testing.assert_allclose(half, (table[f"{prefix} Upper"] - table[f"{prefix} Lower"]) / 2)
    for method in (design.plot_response, design.plot_interval):
        figures = method("A", "B", "Yield", interval=interval, variance_source=variance_source, **options)
        expected_label = "Yield" if method == design.plot_response else (
            "Conf. Interval" if interval == "confidence" else "Pred. Interval"
        )
        for figure in figures:
            hover = [trace.hovertemplate for trace in figure.data if getattr(trace, "hovertemplate", None)]
            assert any(expected_label in text for text in hover)
            assert any(expected_label == (trace.name or "") for trace in figure.data)
        assert np.isfinite(np.asarray(figures[1].data[0].z, dtype=float)).any()


@pytest.mark.parametrize("mixture", [False, True])
def test_two_responses_correct_in_opposite_directions_and_feasibility(mixture, monkeypatch):
    design = fitted_design(mixture)
    ax3 = "C" if mixture else None
    captured = {}
    original = design._compute_feasible_region

    def record(**kwargs):
        captured.update(kwargs)
        result = original(**kwargs)
        expected = (np.asarray(kwargs["response1"]) >= 4.5) & (np.asarray(kwargs["response2"]) <= 8.5)
        np.testing.assert_array_equal(np.isfinite(result), expected)
        return result

    monkeypatch.setattr(design, "_compute_feasible_region", record)
    figures = design.plot_response(
        "A", "B", "Yield", second_response="Purity", ax3=ax3,
        interval="prediction", feasible_region=True, resolution=7,
    )
    # Read factor settings from hover data to check values actually plotted.
    surface = figures[1]
    traces = [trace for trace in surface.data if trace.name in {"Yield", "Purity"}]
    assert len(traces) == 2
    for trace, response, side in zip(traces, ["Yield", "Purity"], [0, 1]):
        data = np.asarray(trace.customdata, dtype=float)
        factors = list(design._factors)
        points = pd.DataFrame(data.reshape(-1, data.shape[-1])[:, :len(factors)], columns=factors)
        matrix = design._build_model_matrix(points, design._model_spec)
        fitted = design._mlr_wrapper.results[response].model
        bounds = fitted.get_prediction(matrix).conf_int(obs=True)
        np.testing.assert_allclose(np.asarray(trace.z).ravel(), bounds[:, side])
    assert [trace.name for trace in traces] == ["Yield", "Purity"]
    for figure in figures:
        titles = [str(item.text) for item in figure.layout.annotations]
        if figure.layout.title.text:
            titles.append(str(figure.layout.title.text))
        assert any("Yield & Purity" in title for title in titles)
        summary = next(
            str(item.text)
            for item in figure.layout.annotations
            if "Max. Yield:" in str(item.text)
        )
        response_traces = {
            trace.name: trace for trace in figure.data
            if trace.name in {"Yield", "Purity"}
        }
        for response in ("Yield", "Purity"):
            plotted = np.asarray(response_traces[response].z, dtype=float)
            assert f"Max. {response}: {np.nanmax(plotted):.3f}" in summary
            assert f"Min. {response}: {np.nanmin(plotted):.3f}" in summary
        hovers = [trace.hovertemplate for trace in figure.data if trace.hovertemplate]
        assert any("Yield:" in hover and "Purity:" in hover for hover in hovers)
    assert captured


def test_public_plot_api_and_invalid_options():
    design = fitted_design()
    assert not hasattr(design, "plot_confidence_interval")
    for method in (design.plot_response, design.plot_interval):
        signature = inspect.signature(method)
        for option in ("interval", "variance_source"):
            assert signature.parameters[option].kind == inspect.Parameter.KEYWORD_ONLY
        for kwargs in ({"interval": "bad"}, {"variance_source": "replicates"}, {"alpha": 0}):
            with pytest.raises(ValueError):
                method("A", "B", "Yield", **kwargs)
    with pytest.raises(TypeError):
        design.plot_response("A", "B", "Yield", corrected="residuals")
    with pytest.raises(TypeError):
        design.plot_interval("A", "B", "Yield", type="residuals")
    design._response_conditions.clear()
    with pytest.raises(ValueError, match="response conditions"):
        design.plot_response("A", "B", "Yield", interval="prediction")
    design._mlr_wrapper.results["Yield"].anova["df_res"] = 0
    with pytest.raises(ValueError, match="residual degrees"):
        design.plot_interval("A", "B", "Yield", resolution=5)
