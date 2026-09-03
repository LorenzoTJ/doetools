import inspect

import numpy as np
import pandas as pd
import pytest

from doetools import FullFactorialDesign
from doetools.graphs.plot_api_mixin import GraphsMixin
from doetools.graphs.renderers import _RendererMixin
from doetools.utils.factors import CategoricalFactor, ContinuousFactor
from doetools.utils.grid_builder import rectangular_grid
from doetools.utils.model_spec import ModelTerms


def _grid(constant_level=0.0, resolution=11):
    reference = pd.DataFrame(columns=["X1", "X2", "X3"])
    return rectangular_grid(
        reference,
        x="X1",
        y="X2",
        constants={"X3": constant_level},
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        resolution=resolution,
    )


def _constraint(frame):
    return frame["X1"] + frame["X2"] <= 1.0


def test_surface_domain_defaults_to_full():
    for method_name in (
        "plot_leverage",
        "plot_response",
        "plot_confidence_interval",
    ):
        parameter = inspect.signature(
            getattr(GraphsMixin, method_name)
        ).parameters["domain"]
        assert parameter.default == "full"


def test_old_surface_domain_keywords_are_removed():
    design = FullFactorialDesign(
        {
            "X1": ContinuousFactor(2, 0.0, 1.0),
            "X2": ContinuousFactor(2, 0.0, 1.0),
        }
    )

    with pytest.raises(TypeError):
        design.plot_design("X1", "X2", mixture_domain_mode="experimental")
    with pytest.raises(TypeError):
        design.plot_leverage("X1", "X2", surface_domain_mode="experimental")
    with pytest.raises(TypeError):
        design.plot_response("X1", "X2", "Yield", mixture_domain_mode="experimental")
    with pytest.raises(TypeError):
        design.plot_confidence_interval(
            "X1", "X2", "Yield", surface_domain_mode="experimental"
        )


def test_design_stores_and_validates_domain_filters():
    design = FullFactorialDesign(
        {
            "X1": ContinuousFactor(2, 0.0, 1.0),
            "X2": ContinuousFactor(2, 0.0, 1.0),
        }
    )

    design.set_domain_filters([_constraint])
    assert design._domain_filters == [_constraint]

    design.set_domain_filters(None)
    assert design._domain_filters == []

    with pytest.raises(TypeError, match="list of callable"):
        design.set_domain_filters([None])


def test_experimental_process_contour_masks_grid_and_keeps_axes():
    renderer = _RendererMixin()
    grid = _grid()
    domain = renderer._build_process_surface_domain(
        grid,
        "X1",
        "X2",
        mode="allowed",
        filters=[_constraint],
        filter_grid=grid,
    )
    response = grid["X1"].to_numpy() + 2 * grid["X2"].to_numpy()

    figure = renderer._render_contour_process(
        grid,
        "X1",
        "X2",
        "Response",
        response,
        process_surface_domain=domain,
    )

    z = np.asarray(figure.data[0].z, dtype=float)
    assert np.isfinite(z).any()
    assert np.isnan(z).any()
    assert any(trace.name == "Allowed domain" for trace in figure.data)
    assert figure.layout.xaxis.showline is True
    assert figure.layout.yaxis.showline is True
    assert np.allclose(figure.layout.xaxis.range, [0.0, 1.0])
    assert np.allclose(figure.layout.yaxis.range, [0.0, 1.0])
    assert figure.layout.meta["domain"] == "allowed"


def test_experimental_process_surface_masks_grid_and_keeps_scene_axes():
    renderer = _RendererMixin()
    grid = _grid()
    domain = renderer._build_process_surface_domain(
        grid,
        "X1",
        "X2",
        mode="allowed",
        filters=[_constraint],
        filter_grid=grid,
    )
    response = grid["X1"].to_numpy() + 2 * grid["X2"].to_numpy()

    figure = renderer._render_surface_process(
        grid,
        "X1",
        "X2",
        "Response",
        response,
        process_surface_domain=domain,
    )

    z = np.asarray(figure.data[0].z, dtype=float)
    assert np.isfinite(z).any()
    assert np.isnan(z).any()
    assert any(
        trace.type == "scatter3d" and trace.name == "Allowed domain"
        for trace in figure.data
    )
    assert figure.layout.scene.xaxis.showgrid is True
    assert figure.layout.scene.yaxis.showgrid is True
    assert np.allclose(figure.layout.scene.xaxis.range, [0.0, 1.0])
    assert np.allclose(figure.layout.scene.yaxis.range, [0.0, 1.0])
    assert figure.layout.meta["domain"] == "allowed"


def test_fixed_factor_level_changes_process_domain_mask():
    renderer = _RendererMixin()

    def sliced_constraint(frame):
        return frame["X1"] + frame["X2"] + frame["X3"] <= 1.2

    low_grid = _grid(constant_level=0.1)
    high_grid = _grid(constant_level=0.7)
    low_domain = renderer._build_process_surface_domain(
        low_grid,
        "X1",
        "X2",
        mode="allowed",
        filters=[sliced_constraint],
        filter_grid=low_grid,
    )
    high_domain = renderer._build_process_surface_domain(
        high_grid,
        "X1",
        "X2",
        mode="allowed",
        filters=[sliced_constraint],
        filter_grid=high_grid,
    )

    assert low_domain.mask.sum() > high_domain.mask.sum()


def test_process_domain_degeneracies_do_not_raise():
    renderer = _RendererMixin()
    grid = _grid(resolution=5)
    empty_domain = renderer._build_process_surface_domain(
        grid,
        "X1",
        "X2",
        mode="allowed",
        filters=[lambda frame: pd.Series(False, index=frame.index)],
        filter_grid=grid,
    )

    contour = renderer._render_contour_process(
        grid,
        "X1",
        "X2",
        "Response",
        np.arange(len(grid), dtype=float),
        process_surface_domain=empty_domain,
    )
    surface = renderer._render_surface_process(
        grid,
        "X1",
        "X2",
        "Response",
        np.arange(len(grid), dtype=float),
        process_surface_domain=empty_domain,
    )

    assert np.isnan(np.asarray(contour.data[0].z, dtype=float)).all()
    assert np.isnan(np.asarray(surface.data[0].z, dtype=float)).all()


def test_process_domain_falls_back_to_available_points_or_full_grid():
    renderer = _RendererMixin()
    grid = _grid(resolution=5)
    triangle = pd.DataFrame(
        {"X1": [0.0, 1.0, 0.0], "X2": [0.0, 0.0, 1.0]}
    )

    candidate_domain = renderer._build_process_surface_domain(
        grid,
        "X1",
        "X2",
        mode="allowed",
        fallback_points=triangle,
    )
    insufficient_domain = renderer._build_process_surface_domain(
        grid,
        "X1",
        "X2",
        mode="allowed",
        fallback_points=triangle.iloc[:2],
    )

    assert candidate_domain.has_area is True
    assert candidate_domain.mask.sum() < candidate_domain.mask.size
    assert len(candidate_domain.boundary) == 4
    assert insufficient_domain.has_area is False
    assert insufficient_domain.mask.all()


def test_plot_leverage_uses_saved_process_filters():
    design = FullFactorialDesign(
        {
            "X1": ContinuousFactor(2, 0.0, 1.0),
            "X2": ContinuousFactor(2, 0.0, 1.0),
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
    design.set_domain_filters([_constraint])

    contour, surface = design.plot_leverage(
        "X1",
        "X2",
        resolution=11,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        domain="allowed",
    )

    assert contour.layout.meta["domain"] == "allowed"
    assert surface.layout.meta["domain"] == "allowed"
    assert np.isnan(np.asarray(contour.data[0].z, dtype=float)).any()


def test_process_hover_shows_constant_categorical_levels():
    design = FullFactorialDesign(
        {
            "T": ContinuousFactor(2, 15.0, 40.0),
            "H": ContinuousFactor(2, 15.0, 70.0),
            "Type": CategoricalFactor(["A", "B"]),
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
    design.set_domain_filters([lambda frame: frame["T"] + frame["H"] <= 90.0])

    contour, _ = design.plot_leverage(
        "T",
        "H",
        constant_levels={"Type": "A"},
        resolution=5,
        domain="allowed",
    )

    assert "Type: %{customdata[2]}" in contour.data[0].hovertemplate
    assert "Type: %{customdata[2]:.3f}" not in contour.data[0].hovertemplate
    assert contour.data[0].customdata[0][0][2] == "A"


def test_response_and_confidence_interval_use_filtered_process_slice():
    design = FullFactorialDesign(
        {
            "X1": ContinuousFactor(2, 0.0, 1.0),
            "X2": ContinuousFactor(2, 0.0, 1.0),
            "X3": ContinuousFactor(2, 0.0, 1.0),
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
    design._response_list = ["Yield"]
    design._responses = pd.DataFrame(
        {
            "Yield": (
                1.0
                + design._design_matrix["X1"]
                + 2.0 * design._design_matrix["X2"]
                + 3.0 * design._design_matrix["X3"]
            )
        }
    )
    design.compute_mlr_model()
    design.set_domain_filters(
        [lambda frame: frame["X1"] + frame["X2"] + frame["X3"] <= 1.2]
    )

    response_figures = design.plot_response(
        "X1",
        "X2",
        "Yield",
        constant_levels={"X3": 0.2},
        resolution=11,
        domain="allowed",
    )
    confidence_figures = design.plot_confidence_interval(
        "X1",
        "X2",
        "Yield",
        constant_levels={"X3": 0.2},
        resolution=11,
        domain="allowed",
    )

    for contour, surface in (response_figures, confidence_figures):
        assert contour.layout.meta["domain"] == "allowed"
        assert surface.layout.meta["domain"] == "allowed"
        assert np.isnan(np.asarray(contour.data[0].z, dtype=float)).any()
        assert any(trace.name == "Allowed domain" for trace in contour.data)
