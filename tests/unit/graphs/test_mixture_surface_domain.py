import inspect

import numpy as np
import pandas as pd

from doetools import ConstrainedMixtureDesign, MixtureFactor
from doetools.graphs.plot_api_mixin import GraphsMixin
from doetools.graphs.renderers import Renderer
from doetools.utils.model_spec import ModelTerms


def _constrained_grid():
    return pd.DataFrame(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.50, 0.30, 0.20],
            [0.60, 0.20, 0.20],
            [0.60, 0.25, 0.15],
        ],
        columns=["A", "B", "C"],
    )


def _render_contour(renderer, grid, response, mode):
    return renderer.render_contour_mixture(
        grid_df_scaled=grid,
        grid_df=grid,
        response=response,
        a_title="A",
        b_title="B",
        c_title="C",
        z_title="Response",
        domain=mode,
        resolution=40,
        show_grid=True,
        show_border=True,
    )


def test_surface_plot_api_defaults_to_full():
    for method_name in (
        "plot_leverage",
        "plot_response",
        "plot_confidence_interval",
    ):
        parameter = inspect.signature(
            getattr(GraphsMixin, method_name)
        ).parameters["domain"]
        assert parameter.default == "full"


def test_allowed_domain_is_propagated_by_all_public_surface_methods():
    factors = {
        name: MixtureFactor(lower_bound=0.1, upper_bound=0.7)
        for name in ("A", "B", "C")
    }
    design = ConstrainedMixtureDesign(factors)
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
        {
            "Yield": (
                2 * matrix["A"]
                + 3 * matrix["B"]
                + 5 * matrix["C"]
                + 0.01 * (matrix.index % 2)
            )
        }
    )
    design.compute_mlr_model()

    figure_pairs = (
        design.plot_leverage(
            "A", "B", "C", resolution=8, domain="allowed"
        ),
        design.plot_response(
            "A",
            "B",
            "Yield",
            ax3="C",
            resolution=8,
            domain="allowed",
        ),
        design.plot_confidence_interval(
            "A",
            "B",
            "Yield",
            ax3="C",
            resolution=8,
            domain="allowed",
        ),
    )

    assert all(
        figure.layout.meta["domain"] == "allowed"
        for figures in figure_pairs
        for figure in figures
    )


def test_allowed_contour_uses_hull_mask_border_and_zoom():
    renderer = Renderer()
    grid = _constrained_grid()
    response = np.linspace(1.0, 2.0, len(grid))

    figure = _render_contour(renderer, grid, response, "allowed")

    assert figure.layout.meta["domain"] == "allowed"
    contour_z = np.asarray(figure.data[0].z, dtype=float)
    assert np.isfinite(contour_z).any()
    assert np.isnan(contour_z).any()

    domain_trace = next(
        trace for trace in figure.data if trace.name == "Allowed domain"
    )
    domain = renderer.build_mixture_surface_domain(
        grid, "A", "B", "C", resolution=40, mode="allowed"
    )
    assert np.allclose(domain_trace.x, domain.boundary[:, 0])
    assert np.allclose(domain_trace.y, domain.boundary[:, 1])

    x_range = tuple(figure.layout.xaxis.range)
    y_range = tuple(figure.layout.yaxis.range)
    assert x_range[0] > 0.0
    assert x_range[1] < 1.0
    assert y_range[1] < np.sqrt(3) / 2.0


def test_full_contour_remains_the_default():
    renderer = Renderer()
    grid = _constrained_grid()

    figure = _render_contour(
        renderer,
        grid,
        np.linspace(1.0, 2.0, len(grid)),
        "full",
    )

    assert figure.layout.meta["domain"] == "full"
    assert figure.layout.xaxis.range is None
    assert figure.layout.yaxis.range is None
    assert any(
        trace.type == "scatter"
        and np.allclose(np.asarray(trace.x, dtype=float), [0.0, 1.0, 0.5, 0.0])
        for trace in figure.data
        if trace.x is not None and len(trace.x) == 4
    )


def test_allowed_surface_mesh_is_limited_to_hull():
    renderer = Renderer()
    grid = _constrained_grid()
    response = np.linspace(1.0, 2.0, len(grid))

    figure = renderer.render_surface_mixture(
        grid_df_scaled=grid,
        grid_df=grid,
        response=response,
        a_title="A",
        b_title="B",
        c_title="C",
        z_title="Response",
        domain="allowed",
    )

    mesh = next(trace for trace in figure.data if trace.type == "mesh3d")
    points = np.column_stack([mesh.x, mesh.y])
    simplices = np.column_stack([mesh.i, mesh.j, mesh.k])
    centroids = points[simplices].mean(axis=1)

    domain = renderer.build_mixture_surface_domain(
        grid, "A", "B", "C", mode="allowed"
    )
    canonical_centroid = np.array([0.5, np.sqrt(3) / 6.0])
    hull_points = domain.boundary[:-1] - canonical_centroid
    from scipy.spatial import ConvexHull

    hull = ConvexHull(hull_points)
    assert np.all(
        centroids @ hull.equations[:, :-1].T + hull.equations[:, -1]
        <= 1e-10
    )
    assert figure.layout.scene.xaxis.range is not None
    assert figure.layout.scene.yaxis.range is not None
    assert figure.layout.scene.zaxis.range is not None


def test_degenerate_allowed_domain_does_not_raise():
    renderer = Renderer()
    grid = pd.DataFrame(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
            [0.50, 0.30, 0.20],
        ],
        columns=["A", "B", "C"],
    )
    response = np.array([1.0, 1.5, 2.0])

    contour = _render_contour(renderer, grid, response, "allowed")
    surface = renderer.render_surface_mixture(
        grid_df_scaled=grid,
        grid_df=grid,
        response=response,
        a_title="A",
        b_title="B",
        c_title="C",
        z_title="Response",
        domain="allowed",
    )

    assert np.isnan(np.asarray(contour.data[0].z, dtype=float)).all()
    assert any(trace.type == "scatter3d" for trace in surface.data)
    assert not any(trace.type == "mesh3d" for trace in surface.data)
