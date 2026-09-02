import numpy as np
import pandas as pd
import plotly.graph_objects as go

from doetools.design.mixture.simplex_centroid import SimplexCentroidDesign
from doetools.design.process.full_factorial import FullFactorialDesign
from doetools.graphs.design_plot_builder import (
    DesignPlotOptions,
    _add_experimental_hull_outline_3d,
    build_design_plot,
    infer_geometry,
    normalize_factors,
)
from doetools.utils.factors import ContinuousFactor, MixtureFactor


def test_builder_supports_dashboard_style_process_2d_from_serialized_factors():
    factors = [
        {
            "name": "Temperature",
            "symbol": "T",
            "type": "continuous",
            "unit": "degC",
            "lower_bound": 20,
            "upper_bound": 80,
        },
        {
            "name": "Pressure",
            "symbol": "P",
            "type": "continuous",
            "unit": "bar",
            "lower_bound": 1,
            "upper_bound": 5,
        },
    ]
    rows = [
        {"T": 20.0, "P": 1.0, "Run": 1, "point_type": "Factorial"},
        {"T": 80.0, "P": 1.0, "Run": 2, "point_type": "Factorial"},
        {"T": 20.0, "P": 5.0, "Run": 3, "point_type": "Factorial"},
        {"T": 20.0, "P": 1.0, "Run": 4, "point_type": "Replicate"},
    ]
    figure = build_design_plot(
        rows,
        rows,
        factors,
        "Full Factorial",
        ["T", "P"],
        options=DesignPlotOptions(show_title=True, marker_color="#e03131"),
    )

    assert figure.data[0].type == "scatter"
    assert figure.data[0].marker.color == "#e03131"
    assert "<b>Design Point</b>" in figure.data[0].hovertemplate
    assert "Coordinates" in figure.data[0].hovertemplate
    assert "Experimental runs" in figure.data[0].hovertemplate
    assert "Replicates" in figure.data[0].hovertemplate
    assert "Exp. numbers" in figure.data[0].hovertemplate
    assert "Unique full-factor combinations" not in figure.data[0].hovertemplate
    assert "Point types" not in figure.data[0].hovertemplate
    assert figure.layout.meta["geometry"] == "process_2d"
    assert figure.layout.meta["axis_labels"]["symbol_unit"] == ["T [degC]", "P [bar]"]
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.scaleratio == 1
    assert np.isclose(
        figure.layout.xaxis.range[1] - figure.layout.xaxis.range[0],
        figure.layout.yaxis.range[1] - figure.layout.yaxis.range[0],
    )
    assert "Runs: 4" in figure.layout.annotations[-1].text


def test_builder_supports_process_3d_from_backend_factor_objects():
    factors = {
        "A": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=1),
        "B": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=1),
        "C": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=1),
    }
    matrix = pd.DataFrame(
        [
            {"A": 0, "B": 0, "C": 0},
            {"A": 1, "B": 0, "C": 0},
            {"A": 0, "B": 1, "C": 1},
        ]
    )

    figure = build_design_plot(matrix, matrix, factors, "Process", ["A", "B", "C"])

    assert figure.data[0].type == "scatter3d"
    assert figure.layout.meta["geometry"] == "process_3d"
    assert figure.layout.scene.aspectmode == "cube"
    assert list(figure.data[0].marker.size) == [9, 9, 9]


def test_builder_supports_three_component_mixture_simplex_2d():
    factors = {
        "A": MixtureFactor(lower_bound=0, upper_bound=1),
        "B": MixtureFactor(lower_bound=0, upper_bound=1),
        "C": MixtureFactor(lower_bound=0, upper_bound=1),
    }
    matrix = pd.DataFrame(
        [
            {"A": 1.0, "B": 0.0, "C": 0.0},
            {"A": 0.0, "B": 1.0, "C": 0.0},
            {"A": 0.0, "B": 0.0, "C": 1.0},
            {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        ]
    )

    figure = build_design_plot(matrix, matrix, factors, "Simplex", ["A", "B", "C"])

    assert figure.data[0].type == "scatter"
    assert figure.layout.meta["geometry"] == "mixture_simplex_2d"
    assert figure.layout.meta["domain"] == "full"
    assert not figure.layout.xaxis.visible
    assert not figure.layout.yaxis.visible
    assert len(figure.layout.shapes) == 31
    assert any(shape.type == "path" for shape in figure.layout.shapes)
    assert len(figure.layout.annotations) == 4
    assert any("A: 1" in annotation.text for annotation in figure.layout.annotations)
    assert not any(annotation.text == "<b>A</b>" for annotation in figure.layout.annotations)
    assert not any(annotation.text == "<b>B</b>" for annotation in figure.layout.annotations)
    assert not any(annotation.text == "<b>C</b>" for annotation in figure.layout.annotations)
    upper_annotations = [
        annotation.text
        for annotation in figure.layout.annotations
        if "A: 0" in annotation.text and "B: 0" in annotation.text and "C: 1" in annotation.text
    ]
    assert upper_annotations
    assert "<br>" not in upper_annotations[0]
    assert figure.layout.hoverlabel.bordercolor == "#003153"


def test_builder_supports_three_component_experimental_mixture_domain():
    factors = {
        "A": MixtureFactor(lower_bound=0, upper_bound=0.7),
        "B": MixtureFactor(lower_bound=0, upper_bound=0.7),
        "C": MixtureFactor(lower_bound=0, upper_bound=0.7),
    }
    matrix = pd.DataFrame(
        [
            {"A": 0.7, "B": 0.3, "C": 0.0},
            {"A": 0.7, "B": 0.0, "C": 0.3},
            {"A": 0.3, "B": 0.7, "C": 0.0},
            {"A": 0.0, "B": 0.7, "C": 0.3},
            {"A": 0.3, "B": 0.0, "C": 0.7},
            {"A": 0.0, "B": 0.3, "C": 0.7},
        ]
    )

    figure = build_design_plot(
        matrix,
        matrix,
        factors,
        "Constrained Mixture",
        ["A", "B", "C"],
        options=DesignPlotOptions(domain="allowed"),
    )

    assert figure.layout.meta["geometry"] == "mixture_simplex_2d"
    assert figure.layout.meta["domain"] == "allowed"
    assert figure.data[0].type == "scatter"
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].type == "path"
    assert figure.layout.shapes[0].fillcolor == "rgba(0,0,0,0)"
    assert not any("A: 1" in str(annotation.text) for annotation in figure.layout.annotations)


def test_experimental_mixture_domain_zooms_to_constrained_3_component_design():
    factors = {
        "A": MixtureFactor(lower_bound=0, upper_bound=0.7),
        "B": MixtureFactor(lower_bound=0, upper_bound=0.7),
        "C": MixtureFactor(lower_bound=0, upper_bound=0.7),
    }
    matrix = pd.DataFrame(
        [
            {"A": 0.7, "B": 0.3, "C": 0.0},
            {"A": 0.7, "B": 0.0, "C": 0.3},
            {"A": 0.3, "B": 0.7, "C": 0.0},
            {"A": 0.0, "B": 0.7, "C": 0.3},
            {"A": 0.3, "B": 0.0, "C": 0.7},
            {"A": 0.0, "B": 0.3, "C": 0.7},
        ]
    )

    figure = build_design_plot(
        matrix,
        matrix,
        factors,
        "Constrained Mixture",
        ["A", "B", "C"],
        options=DesignPlotOptions(domain="allowed"),
    )

    x_range = list(figure.layout.xaxis.range)
    y_range = list(figure.layout.yaxis.range)
    assert x_range[0] > 0.0
    assert x_range[1] < 1.0
    assert y_range[1] < np.sqrt(3) / 2
    assert (y_range[1] - y_range[0]) < np.sqrt(3) / 2


def test_builder_supports_four_component_mixture_tetrahedron_3d():
    factors = {
        name: MixtureFactor(lower_bound=0, upper_bound=1)
        for name in ("A", "B", "C", "D")
    }
    matrix = pd.DataFrame(
        [
            {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
            {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0},
            {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0},
            {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0},
            {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        ]
    )

    figure = build_design_plot(matrix, matrix, factors, "Simplex", ["A", "B", "C", "D"])

    assert figure.data[0].type == "scatter3d"
    assert figure.layout.meta["geometry"] == "mixture_tetrahedron_3d"
    assert figure.layout.scene.aspectmode == "data"
    assert any(trace.mode == "markers" for trace in figure.data[1:])
    assert any(trace.mode == "text" for trace in figure.data[1:])
    assert figure.data[2].hovertemplate.startswith("A: %{customdata[0]")
    assert figure.data[-1].type == "scatter3d"
    assert any(trace.type == "scatter3d" and trace.mode == "lines" for trace in figure.data[1:])


def test_builder_supports_four_component_experimental_mixture_domain():
    factors = {
        name: MixtureFactor(lower_bound=0, upper_bound=1)
        for name in ("A", "B", "C", "D")
    }
    matrix = pd.DataFrame(
        [
            {"A": 0.7, "B": 0.3, "C": 0.0, "D": 0.0},
            {"A": 0.7, "B": 0.0, "C": 0.3, "D": 0.0},
            {"A": 0.7, "B": 0.0, "C": 0.0, "D": 0.3},
            {"A": 0.3, "B": 0.7, "C": 0.0, "D": 0.0},
            {"A": 0.0, "B": 0.7, "C": 0.3, "D": 0.0},
            {"A": 0.0, "B": 0.7, "C": 0.0, "D": 0.3},
            {"A": 0.3, "B": 0.0, "C": 0.7, "D": 0.0},
            {"A": 0.0, "B": 0.3, "C": 0.7, "D": 0.0},
            {"A": 0.0, "B": 0.0, "C": 0.7, "D": 0.3},
            {"A": 0.3, "B": 0.0, "C": 0.0, "D": 0.7},
            {"A": 0.0, "B": 0.3, "C": 0.0, "D": 0.7},
            {"A": 0.0, "B": 0.0, "C": 0.3, "D": 0.7},
        ]
    )

    figure = build_design_plot(
        matrix,
        matrix,
        factors,
        "Constrained Mixture",
        ["A", "B", "C", "D"],
        options=DesignPlotOptions(domain="allowed"),
    )

    assert figure.layout.meta["geometry"] == "mixture_tetrahedron_3d"
    assert figure.layout.meta["domain"] == "allowed"
    assert figure.data[0].type == "scatter3d"
    assert not any(trace.type == "mesh3d" for trace in figure.data[1:])
    assert any(trace.type == "scatter3d" and trace.mode == "lines" for trace in figure.data[1:])


def test_experimental_hull_outline_3d_omits_coplanar_face_diagonals():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    figure = go.Figure()

    _add_experimental_hull_outline_3d(figure, points)

    line_trace = figure.data[0]
    segments = [
        (
            np.array([line_trace.x[i], line_trace.y[i], line_trace.z[i]], dtype=float),
            np.array([line_trace.x[i + 1], line_trace.y[i + 1], line_trace.z[i + 1]], dtype=float),
        )
        for i in range(0, len(line_trace.x), 3)
    ]
    lengths = [np.linalg.norm(end - start) for start, end in segments]
    assert len(segments) == 12
    assert np.allclose(lengths, 1.0)


def test_experimental_mixture_domain_handles_degenerate_hull_without_error():
    factors = {
        "A": MixtureFactor(lower_bound=0, upper_bound=1),
        "B": MixtureFactor(lower_bound=0, upper_bound=1),
        "C": MixtureFactor(lower_bound=0, upper_bound=1),
    }
    matrix = pd.DataFrame(
        [
            {"A": 1.0, "B": 0.0, "C": 0.0},
            {"A": 0.5, "B": 0.5, "C": 0.0},
            {"A": 0.0, "B": 1.0, "C": 0.0},
        ]
    )

    figure = build_design_plot(
        matrix,
        matrix,
        factors,
        "Degenerate Mixture",
        ["A", "B", "C"],
        options=DesignPlotOptions(domain="allowed"),
    )

    assert figure.layout.meta["domain"] == "allowed"
    assert figure.data[0].type == "scatter"


def test_plot_design_wrapper_uses_builder_metadata_for_process_design():
    design = FullFactorialDesign(
        factors={
            "A": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=9),
        }
    )

    figure = design.plot_design("A", "B", coded=True)

    assert figure.layout.meta["source"] == "doetools.graphs.design_plot_builder"
    assert figure.layout.meta["coordinate_mode"] == "coded"
    assert figure.layout.meta["geometry"] == "process_2d"


def test_plot_design_wrapper_shows_design_name_title_by_default():
    design = FullFactorialDesign(
        factors={
            "A": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=9),
        }
    )

    figure = design.plot_design("A", "B")

    assert any(
        annotation.text == "<b>Full Factorial Design</b>"
        for annotation in figure.layout.annotations
    )
    assert figure.layout.title.text is None


def test_plot_design_wrapper_can_hide_design_name_title():
    design = FullFactorialDesign(
        factors={
            "A": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=9),
        }
    )

    figure = design.plot_design("A", "B", show_title=False)

    assert not any(
        annotation.text == "<b>Full Factorial Design</b>"
        for annotation in figure.layout.annotations
    )


def test_design_summary_annotation_is_offset_from_top_legend():
    design = FullFactorialDesign(
        factors={
            "A": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10),
            "B": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=9),
        }
    )

    figure = design.plot_design("A", "B")
    summary = next(
        annotation
        for annotation in figure.layout.annotations
        if str(annotation.text).startswith("Runs:")
    )

    assert summary.y == 0.91
    assert figure.layout.legend.y > summary.y


def test_plot_design_wrapper_uses_builder_metadata_for_mixture_design():
    design = SimplexCentroidDesign(
        factors={
            "A": MixtureFactor(lower_bound=0, upper_bound=1),
            "B": MixtureFactor(lower_bound=0, upper_bound=1),
            "C": MixtureFactor(lower_bound=0, upper_bound=1),
        }
    )

    figure = design.plot_design("A", "B", "C")

    assert figure.layout.meta["geometry"] == "mixture_simplex_2d"
    assert np.isclose(max(figure.data[0].y), np.sqrt(3) / 2)


def test_plot_design_wrapper_accepts_allowed_mixture_domain():
    design = SimplexCentroidDesign(
        factors={
            "A": MixtureFactor(lower_bound=0, upper_bound=0.7),
            "B": MixtureFactor(lower_bound=0, upper_bound=0.7),
            "C": MixtureFactor(lower_bound=0, upper_bound=0.7),
        }
    )

    figure = design.plot_design("A", "B", "C", domain="allowed")

    assert figure.layout.meta["geometry"] == "mixture_simplex_2d"
    assert figure.layout.meta["domain"] == "allowed"


def test_geometry_rejects_mixed_process_and_mixture_axes():
    factors = normalize_factors(
        {
            "A": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=1),
            "M": MixtureFactor(lower_bound=0, upper_bound=1),
        }
    )

    try:
        infer_geometry(["A", "M"], factors)
    except ValueError as error:
        assert "same family" in str(error)
    else:
        raise AssertionError("Expected mixed process/mixture axes to fail.")
