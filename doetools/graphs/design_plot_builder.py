from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, QhullError


SCIENTIFIC_FONT = "Arial, sans-serif"
DEFAULT_MARKER_COLOR = "#1971c2"
RUN_METADATA_COLUMNS = {
    "Run",
    "run_id",
    "point_type",
    "source_run_id",
    "is_center",
    "is_replicate",
    "is_manual",
}


@dataclass(frozen=True)
class DesignPlotOptions:
    coded: bool = False
    marker_color: str = DEFAULT_MARKER_COLOR
    show_title: bool = False
    show_summary: bool = True
    show_legend: bool = True
    show_grid: bool = True
    show_hover: bool = True
    show_run_labels: bool = False
    show_replicate_count: bool = True
    aggregate_projected_points: bool = True
    axis_label_mode: str = "symbol_unit"
    height: int | None = None
    mixture_simplex_3d: bool = False
    domain: Literal["full", "allowed"] = "full"


@dataclass(frozen=True)
class FactorInfo:
    key: str
    name: str
    type: str
    unit: str = ""
    levels: tuple[Any, ...] = ()


def _validate_design_plot_options(options: DesignPlotOptions) -> None:
    if options.domain not in {"full", "allowed"}:
        raise ValueError("domain must be either 'full' or 'allowed'.")


def build_design_plot(
    design_matrix: pd.DataFrame | Iterable[Mapping[str, Any]],
    coded_design_matrix: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    factors: Mapping[str, Any] | Iterable[Any],
    design_type: str,
    axes: Sequence[str],
    *,
    options: DesignPlotOptions | None = None,
    filters: Mapping[str, Sequence[Any]] | None = None,
) -> go.Figure:
    """Build a design-space Plotly figure from explicit data.

    The function is intentionally independent from Dash and from the design class
    instance so it can be used both by ``plot_design()`` and by a dashboard service.
    """
    options = options or DesignPlotOptions()
    _validate_design_plot_options(options)
    design_df = _as_dataframe(design_matrix)
    coded_df = _as_dataframe(coded_design_matrix) if coded_design_matrix is not None else design_df
    matrix = coded_df if options.coded else design_df
    factor_lookup = normalize_factors(factors)
    axes = resolve_axes(axes, factor_lookup)
    geometry = infer_geometry(axes, factor_lookup)
    if options.mixture_simplex_3d and geometry == "mixture_simplex_2d":
        geometry = "mixture_simplex_3d"
    plot_data = build_design_plot_data(
        matrix,
        factor_lookup,
        axes,
        filters=filters,
        aggregate=options.aggregate_projected_points,
    )
    return render_design_plot(
        plot_data,
        factor_lookup,
        design_type,
        axes=axes,
        geometry=geometry,
        coordinate_mode="coded" if options.coded else "actual",
        options=options,
    )


def normalize_factors(factors: Mapping[str, Any] | Iterable[Any]) -> dict[str, FactorInfo]:
    if isinstance(factors, Mapping):
        items = factors.items()
    else:
        items = ((_factor_key(factor), factor) for factor in factors or [])
    normalized: dict[str, FactorInfo] = {}
    for fallback_key, factor in items:
        key = _factor_key(factor, fallback=str(fallback_key))
        if not key:
            continue
        normalized[key] = FactorInfo(
            key=key,
            name=str(_factor_value(factor, "name", key) or key),
            type=normalize_factor_type(_factor_value(factor, "type")),
            unit=str(_factor_value(factor, "unit", "") or "").strip(),
            levels=_factor_levels(factor),
        )
    return normalized


def normalize_factor_type(value: Any) -> str:
    return {
        "continuous": "cont",
        "categorical": "cat",
        "mixture": "mix",
        "cont": "cont",
        "cat": "cat",
        "mix": "mix",
    }.get(str(value or "").lower(), str(value or ""))


def resolve_axes(axes: Sequence[str], factors: Mapping[str, FactorInfo]) -> list[str]:
    resolved = [axis for axis in axes if axis]
    missing = [axis for axis in resolved if axis not in factors]
    if missing:
        raise KeyError("Unknown factor(s): " + ", ".join(missing))
    if len(set(resolved)) != len(resolved):
        raise ValueError("Choose a different factor for each active axis.")
    if len(resolved) not in {2, 3, 4}:
        raise ValueError("Design plots require two, three, or four active factors.")
    return resolved


def infer_geometry(axes: Sequence[str], factors: Mapping[str, FactorInfo]) -> str:
    factor_types = [factors[axis].type for axis in axes]
    if all(kind == "mix" for kind in factor_types):
        if len(axes) == 2:
            return "mixture_line_2d"
        if len(axes) == 3:
            return "mixture_simplex_2d"
        if len(axes) == 4:
            return "mixture_tetrahedron_3d"
    if any(kind == "mix" for kind in factor_types):
        raise ValueError("Design plot supports only axes of the same family: process or mixture.")
    if len(axes) == 2:
        return "process_2d"
    if len(axes) == 3:
        return "process_3d"
    raise ValueError("Process design plots support two or three active factors.")


def build_design_plot_data(
    matrix: pd.DataFrame,
    factors: Mapping[str, FactorInfo],
    axes: Sequence[str],
    *,
    filters: Mapping[str, Sequence[Any]] | None = None,
    aggregate: bool = True,
) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    missing = [axis for axis in axes if axis not in matrix]
    if missing:
        raise KeyError("Design matrix is missing factor(s): " + ", ".join(missing))
    factor_columns = [key for key in factors if key in matrix]
    filtered = apply_projection_filters(matrix, filters)
    if filtered.empty:
        return pd.DataFrame()
    if not aggregate:
        return _unaggregated_plot_data(filtered, axes)
    return aggregate_projected_points(filtered, axes, factor_columns=factor_columns)


def apply_projection_filters(
    matrix: pd.DataFrame,
    filters: Mapping[str, Sequence[Any]] | None,
) -> pd.DataFrame:
    filtered = matrix.copy()
    for factor, selected in (filters or {}).items():
        if factor not in filtered or not selected:
            continue
        selected_strings = {str(value) for value in selected}
        filtered = filtered[filtered[factor].map(str).isin(selected_strings)]
    return filtered.reset_index(drop=True)


def aggregate_projected_points(
    matrix: pd.DataFrame,
    axes: Sequence[str],
    *,
    factor_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = matrix.groupby(list(axes), dropna=False, sort=False)
    for coordinates, group in grouped:
        if not isinstance(coordinates, tuple):
            coordinates = (coordinates,)
        unique_count = len(group[list(factor_columns)].drop_duplicates())
        run_count = len(group)
        rows.append(
            {
                **{axis: coordinates[index] for index, axis in enumerate(axes)},
                "_unique_count": unique_count,
                "_run_count": run_count,
                "_replicate_count": run_count - unique_count,
                "_point_types": ", ".join(sorted({_point_type(row) for row in group.to_dict("records")})),
                "_runs": _run_numbers(group),
            }
        )
    return pd.DataFrame(rows)


def barycentric_to_cartesian_2d(matrix: np.ndarray) -> np.ndarray:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ]
    )
    return np.asarray(matrix, dtype=float) @ vertices


def barycentric_to_cartesian_3d(matrix: np.ndarray) -> np.ndarray:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
        ]
    )
    return np.asarray(matrix, dtype=float) @ vertices


def render_design_plot(
    plot_data: pd.DataFrame,
    factors: Mapping[str, FactorInfo],
    design_type: str,
    *,
    axes: Sequence[str],
    geometry: str,
    coordinate_mode: str,
    options: DesignPlotOptions,
) -> go.Figure:
    if plot_data.empty:
        figure = empty_figure("No runs match the selected design view.", options=options)
    elif geometry == "process_2d":
        figure = render_process_2d(plot_data, axes, options)
    elif geometry == "process_3d":
        figure = render_process_3d(plot_data, axes, options)
    elif geometry == "mixture_line_2d":
        figure = render_mixture_line_2d(plot_data, axes, options)
    elif geometry == "mixture_simplex_2d":
        figure = render_mixture_simplex_2d(plot_data, axes, options)
    elif geometry == "mixture_simplex_3d":
        figure = render_mixture_simplex_3d(plot_data, axes, options)
    elif geometry == "mixture_tetrahedron_3d":
        figure = render_mixture_tetrahedron_3d(plot_data, axes, options)
    else:
        raise ValueError(f"Unsupported design plot geometry: {geometry}")

    axis_labels = _axis_label_sets(axes, factors)
    selected_labels = axis_labels.get(options.axis_label_mode, axis_labels["symbol"])
    _apply_common_layout(
        figure,
        design_type,
        plot_data,
        axes=axes,
        axis_labels=selected_labels,
        axis_label_sets=axis_labels,
        geometry=geometry,
        coordinate_mode=coordinate_mode,
        options=options,
    )
    return figure


def render_process_2d(plot_data: pd.DataFrame, axes: Sequence[str], options: DesignPlotOptions) -> go.Figure:
    labels = _point_labels(plot_data, options)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=plot_data[axes[0]],
            y=plot_data[axes[1]],
            mode="markers+text" if any(labels) else "markers",
            text=labels,
            textposition="top center",
            marker=_marker(plot_data, options, size_base=10),
            customdata=_customdata(plot_data),
            name="Design points",
            hovertemplate=_hovertemplate(axes, "process_2d", options),
            hoverinfo=None if options.show_hover else "skip",
        )
    )
    return figure


def render_process_3d(plot_data: pd.DataFrame, axes: Sequence[str], options: DesignPlotOptions) -> go.Figure:
    labels = _point_labels(plot_data, options)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=plot_data[axes[0]],
            y=plot_data[axes[1]],
            z=plot_data[axes[2]],
            mode="markers+text" if any(labels) else "markers",
            text=labels,
            textposition="top center",
            marker=_marker(plot_data, options, size_base=9),
            customdata=_customdata(plot_data),
            name="Design points",
            hovertemplate=_hovertemplate(axes, "process_3d", options),
            hoverinfo=None if options.show_hover else "skip",
        )
    )
    return figure


def render_mixture_line_2d(plot_data: pd.DataFrame, axes: Sequence[str], options: DesignPlotOptions) -> go.Figure:
    figure = render_process_2d(plot_data, axes, options)
    figure.add_shape(
        type="line",
        x0=0,
        y0=1,
        x1=1,
        y1=0,
        line={"color": "#adb5bd", "width": 1, "dash": "dot"},
        layer="below",
    )
    return figure


def render_mixture_simplex_2d(plot_data: pd.DataFrame, axes: Sequence[str], options: DesignPlotOptions) -> go.Figure:
    labels = _point_labels(plot_data, options)
    barycentric = plot_data[list(axes)].to_numpy(dtype=float)
    xy = barycentric_to_cartesian_2d(barycentric)
    figure = go.Figure()
    if options.domain == "full":
        _add_simplex_triangle(figure, options.show_grid)
    _add_experimental_hull_outline_2d(figure, xy)
    figure.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers+text" if any(labels) else "markers",
            text=labels,
            textposition="top center",
            marker=_marker(plot_data, options, size_base=10),
            customdata=_customdata(plot_data, extra=barycentric),
            name="Design points",
            hovertemplate=_hovertemplate(axes, "mixture_simplex_2d", options),
            hoverinfo=None if options.show_hover else "skip",
        )
    )
    return figure


def render_mixture_simplex_3d(plot_data: pd.DataFrame, axes: Sequence[str], options: DesignPlotOptions) -> go.Figure:
    labels = _point_labels(plot_data, options)
    barycentric = plot_data[list(axes)].to_numpy(dtype=float)
    xy = barycentric_to_cartesian_2d(barycentric)
    z = np.zeros(len(xy))
    figure = go.Figure()
    if options.domain == "full":
        _add_simplex_triangle_3d(figure, options.show_grid)
    _add_experimental_hull_outline_3d_plane(figure, xy)
    figure.add_trace(
        go.Scatter3d(
            x=xy[:, 0],
            y=xy[:, 1],
            z=z,
            mode="markers+text" if any(labels) else "markers",
            text=labels,
            textposition="top center",
            marker=_marker(plot_data, options, size_base=9),
            customdata=_customdata(plot_data, extra=barycentric),
            name="Design points",
            hovertemplate=_hovertemplate(axes, "mixture_simplex_2d", options),
            hoverinfo=None if options.show_hover else "skip",
        )
    )
    return figure


def render_mixture_tetrahedron_3d(plot_data: pd.DataFrame, axes: Sequence[str], options: DesignPlotOptions) -> go.Figure:
    labels = _point_labels(plot_data, options)
    barycentric = plot_data[list(axes)].to_numpy(dtype=float)
    xyz = barycentric_to_cartesian_3d(barycentric)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers+text" if any(labels) else "markers",
            text=labels,
            textposition="top center",
            marker=_marker(plot_data, options, size_base=6),
            customdata=_customdata(plot_data, extra=barycentric),
            name="Design points",
            hovertemplate=_hovertemplate(axes, "mixture_tetrahedron_3d", options),
            hoverinfo=None if options.show_hover else "skip",
        )
    )
    if options.domain == "full":
        _add_simplex_tetrahedron(figure, axes, options.show_grid)
    _add_experimental_hull_outline_3d(figure, xyz)
    return figure


def empty_figure(message: str, *, options: DesignPlotOptions) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#868e96", "size": 14},
    )
    return figure


def _apply_common_layout(
    figure: go.Figure,
    design_type: str,
    plot_data: pd.DataFrame,
    *,
    axes: Sequence[str],
    axis_labels: Sequence[str],
    axis_label_sets: Mapping[str, Sequence[str]],
    geometry: str,
    coordinate_mode: str,
    options: DesignPlotOptions,
) -> None:
    title = _design_plot_title(design_type) if options.show_title else None
    figure.update_layout(
        template="plotly_white",
        title=None,
        font={"family": SCIENTIFIC_FONT, "size": 13, "color": "#212529"},
        margin={"l": 64, "r": 28, "t": 46 if title else 24, "b": 58},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=options.show_legend,
        legend={"orientation": "h", "y": 1.03, "x": 1, "xanchor": "right"},
        uirevision="doetools-design-plot",
        meta={
            "source": "doetools.graphs.design_plot_builder",
            "geometry": geometry,
            "domain": options.domain,
            "axes": list(axes),
            "coordinate_mode": coordinate_mode,
            "axis_labels": {key: list(value) for key, value in axis_label_sets.items()},
        },
    )
    if title:
        figure.add_annotation(
            text=f"<b>{title}</b>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.09,
            showarrow=False,
            font={"size": 20, "family": SCIENTIFIC_FONT, "color": "black"},
        )
    if options.height is not None:
        figure.update_layout(height=options.height)

    if geometry in {"process_2d", "mixture_line_2d"}:
        figure.update_xaxes(title=axis_labels[0], showgrid=options.show_grid, gridcolor="#e9ecef", zeroline=False)
        figure.update_yaxes(title=axis_labels[1], showgrid=options.show_grid, gridcolor="#e9ecef", zeroline=False)
        if geometry == "process_2d":
            _apply_square_process_range_2d(figure, plot_data, axes)
        if geometry == "mixture_line_2d":
            figure.update_xaxes(range=[-0.03, 1.03], constrain="domain")
            figure.update_yaxes(range=[-0.03, 1.03], scaleanchor="x", scaleratio=1)
    elif geometry == "process_3d":
        figure.update_layout(
            scene={
                "xaxis": {"title": axis_labels[0], "showgrid": options.show_grid},
                "yaxis": {"title": axis_labels[1], "showgrid": options.show_grid},
                "zaxis": {"title": axis_labels[2], "showgrid": options.show_grid},
                "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
                "aspectmode": "cube",
            }
        )
    elif geometry == "mixture_simplex_2d":
        if options.domain == "full":
            _add_simplex_vertex_annotations(figure, axis_labels)
        figure.update_layout(
            margin={"l": 40, "r": 40, "t": 50 if title else 36, "b": 50},
            xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
            yaxis={"visible": False},
            hovermode="closest",
            hoverlabel={
                "bgcolor": "white",
                "bordercolor": "#003153",
                "font": {"size": 12, "family": SCIENTIFIC_FONT, "color": "black"},
            },
        )
        if options.domain == "allowed":
            xy = barycentric_to_cartesian_2d(plot_data[list(axes)].to_numpy(dtype=float))
            _apply_experimental_range_2d(figure, xy)
    elif geometry == "mixture_simplex_3d":
        if options.domain == "full":
            _add_simplex_vertex_labels_3d_plane(figure, axis_labels)
        xy = barycentric_to_cartesian_2d(plot_data[list(axes)].to_numpy(dtype=float))
        scene = {
            "xaxis": {"visible": False, "showgrid": False},
            "yaxis": {"visible": False, "showgrid": False},
            "zaxis": {"visible": False, "showgrid": False},
            "camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.05}},
            "aspectmode": "data",
        }
        if options.domain == "allowed":
            scene.update(_experimental_scene_ranges(np.column_stack([xy, np.zeros(len(xy))])))
        figure.update_layout(
            margin={"l": 8, "r": 8, "t": 20, "b": 8},
            scene=scene,
            hovermode="closest",
        )
    elif geometry == "mixture_tetrahedron_3d":
        if options.domain == "full":
            _add_simplex_vertex_labels_3d(figure, axis_labels)
        xyz = barycentric_to_cartesian_3d(plot_data[list(axes)].to_numpy(dtype=float))
        scene = {
            "xaxis": {"visible": False, "showgrid": False},
            "yaxis": {"visible": False, "showgrid": False},
            "zaxis": {"visible": False, "showgrid": False},
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
            "aspectmode": "data",
        }
        if options.domain == "allowed":
            scene.update(_experimental_scene_ranges(xyz))
        figure.update_layout(
            scene=scene
        )

    if options.show_summary and not plot_data.empty:
        total_runs = int(plot_data["_run_count"].sum()) if "_run_count" in plot_data else len(plot_data)
        replicate_runs = int(plot_data["_replicate_count"].sum()) if "_replicate_count" in plot_data else 0
        figure.add_annotation(
            text=f"Runs: {total_runs}<br>Replicates: {replicate_runs}",
            x=0.99,
            y=0.91,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            align="right",
            font={"size": 12, "family": SCIENTIFIC_FONT, "color": "#212529"},
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#dee2e6",
            borderwidth=1,
            borderpad=6,
        )


def _add_experimental_hull_outline_2d(figure: go.Figure, xy: np.ndarray) -> None:
    unique_xy = _unique_rows(xy)
    if len(unique_xy) < 2:
        return
    shapes = list(figure.layout.shapes or [])
    if len(unique_xy) >= 3:
        try:
            hull = ConvexHull(unique_xy)
            polygon = unique_xy[hull.vertices]
            path = "M " + " L ".join(f"{x},{y}" for x, y in polygon) + " Z"
            shapes.append(
                {
                    "type": "path",
                    "path": path,
                    "xref": "x",
                    "yref": "y",
                    "layer": "below",
                    "fillcolor": "rgba(0,0,0,0)",
                    "line": {"color": "#1971c2", "width": 2.4},
                }
            )
            figure.update_layout(shapes=shapes)
            return
        except QhullError:
            pass

    order = np.argsort(unique_xy[:, 0] + unique_xy[:, 1])
    line_points = unique_xy[order]
    shapes.append(
        _line_shape(
            line_points[0, 0],
            line_points[0, 1],
            line_points[-1, 0],
            line_points[-1, 1],
            color="#1971c2",
            width=2,
        )
    )
    figure.update_layout(shapes=shapes)


def _add_experimental_hull_outline_3d_plane(figure: go.Figure, xy: np.ndarray) -> None:
    unique_xy = _unique_rows(xy)
    if len(unique_xy) < 2:
        return
    if len(unique_xy) >= 3:
        try:
            hull = ConvexHull(unique_xy)
            polygon = unique_xy[hull.vertices]
            if len(polygon) >= 3:
                _add_3d_polyline(figure, np.column_stack([polygon, np.zeros(len(polygon))]), close=True)
                return
        except QhullError:
            pass

    points = np.column_stack([unique_xy, np.zeros(len(unique_xy))])
    _add_3d_polyline(figure, points, close=False)


def _add_experimental_hull_outline_3d(figure: go.Figure, xyz: np.ndarray) -> None:
    unique_xyz = _unique_rows(xyz)
    if len(unique_xyz) < 2:
        return
    if len(unique_xyz) >= 4:
        try:
            hull = ConvexHull(unique_xyz)
            _add_hull_edges_3d(figure, unique_xyz, hull.simplices, hull.equations)
            return
        except QhullError:
            pass

    _add_3d_polyline(figure, unique_xyz, close=False)


def _add_hull_edges_3d(
    figure: go.Figure,
    points: np.ndarray,
    simplices: np.ndarray,
    equations: np.ndarray,
) -> None:
    edges = _external_hull_edges_3d(simplices, equations)
    if not edges:
        edges = {
            tuple(sorted(edge))
            for a, b, c in simplices
            for edge in ((a, b), (b, c), (c, a))
        }
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in sorted(edges):
        x_values.extend([points[start, 0], points[end, 0], None])
        y_values.extend([points[start, 1], points[end, 1], None])
        z_values.extend([points[start, 2], points[end, 2], None])
    figure.add_trace(
        go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line={"color": "#1971c2", "width": 4},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _external_hull_edges_3d(
    simplices: np.ndarray,
    equations: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> set[tuple[int, int]]:
    """Return only polyhedron edges, excluding diagonals from triangulated faces."""
    face_groups: dict[tuple[int, ...], list[tuple[int, int, int]]] = {}
    for simplex, equation in zip(simplices, equations):
        key = _canonical_hull_plane_key(equation, tolerance=tolerance)
        face_groups.setdefault(key, []).append(tuple(int(index) for index in simplex))

    edges: set[tuple[int, int]] = set()
    for face_simplices in face_groups.values():
        face_edge_counts: dict[tuple[int, int], int] = {}
        for a, b, c in face_simplices:
            for edge in ((a, b), (b, c), (c, a)):
                sorted_edge = tuple(sorted(edge))
                face_edge_counts[sorted_edge] = face_edge_counts.get(sorted_edge, 0) + 1
        edges.update(edge for edge, count in face_edge_counts.items() if count == 1)
    return edges


def _canonical_hull_plane_key(equation: np.ndarray, *, tolerance: float) -> tuple[int, ...]:
    normal = np.asarray(equation[:-1], dtype=float)
    offset = float(equation[-1])
    norm = float(np.linalg.norm(normal))
    if norm > 0:
        plane = np.append(normal / norm, offset / norm)
    else:
        plane = np.asarray(equation, dtype=float)

    nonzero = np.flatnonzero(np.abs(plane[:-1]) > tolerance)
    if nonzero.size and plane[nonzero[0]] < 0:
        plane = -plane
    return tuple(np.round(plane / tolerance).astype(int))


def _add_3d_polyline(figure: go.Figure, points: np.ndarray, *, close: bool) -> None:
    if len(points) < 2:
        return
    ordered = points[np.argsort(points[:, 0] + points[:, 1] + points[:, 2])]
    if close and len(ordered) > 2:
        ordered = np.vstack([ordered, ordered[0]])
    figure.add_trace(
        go.Scatter3d(
            x=ordered[:, 0],
            y=ordered[:, 1],
            z=ordered[:, 2],
            mode="lines",
            line={"color": "#1971c2", "width": 4},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _apply_experimental_range_2d(figure: go.Figure, xy: np.ndarray) -> None:
    x_range = _padded_range(xy[:, 0])
    y_range = _padded_range(xy[:, 1])
    figure.update_xaxes(range=x_range)
    figure.update_yaxes(range=y_range)


def _apply_square_process_range_2d(figure: go.Figure, plot_data: pd.DataFrame, axes: Sequence[str]) -> None:
    x_values = pd.to_numeric(plot_data[axes[0]], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(plot_data[axes[1]], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x_values).any() or not np.isfinite(y_values).any():
        return

    x_range = _padded_range(x_values)
    y_range = _padded_range(y_values)
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    span = max(x_span, y_span)
    x_center = sum(x_range) / 2
    y_center = sum(y_range) / 2
    figure.update_xaxes(range=[x_center - span / 2, x_center + span / 2], constrain="domain")
    figure.update_yaxes(
        range=[y_center - span / 2, y_center + span / 2],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )


def _experimental_scene_ranges(xyz: np.ndarray) -> dict[str, Any]:
    return {
        "xaxis": {"visible": False, "showgrid": False, "range": _padded_range(xyz[:, 0])},
        "yaxis": {"visible": False, "showgrid": False, "range": _padded_range(xyz[:, 1])},
        "zaxis": {"visible": False, "showgrid": False, "range": _padded_range(xyz[:, 2])},
    }


def _padded_range(values: np.ndarray, *, fraction: float = 0.12) -> list[float]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [-1.0, 1.0]
    low = float(finite_values.min())
    high = float(finite_values.max())
    span = high - low
    if span <= 1e-12:
        padding = max(abs(low), 1.0) * 0.05
    else:
        padding = span * fraction
    return [low - padding, high + padding]


def _unique_rows(points: np.ndarray, decimals: int = 12) -> np.ndarray:
    if len(points) == 0:
        return np.asarray(points, dtype=float)
    rounded = np.round(np.asarray(points, dtype=float), decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return np.asarray(points, dtype=float)[np.sort(indices)]


def _add_simplex_triangle(figure: go.Figure, show_grid: bool) -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    shapes = list(figure.layout.shapes or [])
    if show_grid:
        for value in np.arange(0.1, 1.0, 0.1):
            for segment in (
                np.array([[value, 1 - value, 0], [value, 0, 1 - value]]),
                np.array([[1 - value, value, 0], [0, value, 1 - value]]),
                np.array([[1 - value, 0, value], [0, 1 - value, value]]),
            ):
                xy = barycentric_to_cartesian_2d(segment)
                shapes.append(_line_shape(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1], color="#dee2e6", width=1))
    edges = [(0, 1), (1, 2), (2, 0)]
    for start, end in edges:
        shapes.append(
            _line_shape(
                vertices[start, 0],
                vertices[start, 1],
                vertices[end, 0],
                vertices[end, 1],
                color="#212529",
                width=1.8,
            )
        )
    figure.update_layout(shapes=shapes)


def _add_simplex_triangle_3d(figure: go.Figure, show_grid: bool) -> None:
    vertices_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    vertices = np.column_stack([vertices_2d, np.zeros(3)])
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[0],
            j=[1],
            k=[2],
            color="#1971c2",
            opacity=0.08,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    edges = [(0, 1), (1, 2), (2, 0)]
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in edges:
        x_values.extend([vertices[start, 0], vertices[end, 0], None])
        y_values.extend([vertices[start, 1], vertices[end, 1], None])
        z_values.extend([0.0, 0.0, None])
    figure.add_trace(
        go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line={"color": "#495057", "width": 4},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if not show_grid:
        return

    grid_x: list[float | None] = []
    grid_y: list[float | None] = []
    grid_z: list[float | None] = []
    for value in np.arange(0.1, 1.0, 0.1):
        for segment in (
            np.array([[value, 1 - value, 0], [value, 0, 1 - value]]),
            np.array([[1 - value, value, 0], [0, value, 1 - value]]),
            np.array([[1 - value, 0, value], [0, 1 - value, value]]),
        ):
            xy = barycentric_to_cartesian_2d(segment)
            grid_x.extend([xy[0, 0], xy[1, 0], None])
            grid_y.extend([xy[0, 1], xy[1, 1], None])
            grid_z.extend([0.0, 0.0, None])
    figure.add_trace(
        go.Scatter3d(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            mode="lines",
            line={"color": "rgba(0,0,0,0.18)", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _simplex_tetrahedron_vertices() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
        ]
    )


def _add_simplex_tetrahedron(
    figure: go.Figure,
    axis_labels: Sequence[str],
    show_grid: bool,
    *,
    grid_n: int = 5,
) -> None:
    vertices = _simplex_tetrahedron_vertices()
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in edges:
        x_values.extend([vertices[start, 0], vertices[end, 0], None])
        y_values.extend([vertices[start, 1], vertices[end, 1], None])
        z_values.extend([vertices[start, 2], vertices[end, 2], None])
    figure.add_trace(
        go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line={"color": "#495057", "width": 3},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker={"size": 6, "color": "#003153", "opacity": 0.95},
            customdata=np.eye(4),
            hovertemplate="<br>".join(
                f"{axis}: %{{customdata[{index}]:.0f}}"
                for index, axis in enumerate(axis_labels)
            ) + "<extra></extra>",
            showlegend=False,
        )
    )
    if show_grid and grid_n > 0:
        _add_tetrahedral_grid(figure, vertices, grid_n=grid_n)


def _add_tetrahedral_grid(figure: go.Figure, vertices: np.ndarray, *, grid_n: int) -> None:
    def project(weights):
        return np.asarray(weights, dtype=float) @ vertices

    points = {}
    for i in range(grid_n + 1):
        for j in range(grid_n + 1 - i):
            for k in range(grid_n + 1 - i - j):
                fourth = grid_n - i - j - k
                points[(i, j, k, fourth)] = project(
                    (i / grid_n, j / grid_n, k / grid_n, fourth / grid_n)
                )

    moves = [
        (1, -1, 0, 0),
        (1, 0, -1, 0),
        (1, 0, 0, -1),
        (0, 1, -1, 0),
        (0, 1, 0, -1),
        (0, 0, 1, -1),
    ]
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for index, point in points.items():
        for move in moves:
            neighbor = tuple(index[dimension] + move[dimension] for dimension in range(4))
            if neighbor in points:
                other = points[neighbor]
                x_values.extend([point[0], other[0], None])
                y_values.extend([point[1], other[1], None])
                z_values.extend([point[2], other[2], None])
    figure.add_trace(
        go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line={"color": "rgba(0,0,0,0.18)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _add_simplex_vertex_annotations(figure: go.Figure, axis_labels: Sequence[str]) -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    texts = [
        f"{axis_labels[0]}: 1<br>{axis_labels[1]}: 0<br>{axis_labels[2]}: 0",
        f"{axis_labels[0]}: 0<br>{axis_labels[1]}: 1<br>{axis_labels[2]}: 0",
        f"{axis_labels[0]}: 0&nbsp;&nbsp;{axis_labels[1]}: 0&nbsp;&nbsp;{axis_labels[2]}: 1",
    ]
    placements = [
        (vertices[0], texts[0], -25, 0),
        (vertices[1], texts[1], 25, 0),
        (vertices[2], texts[2], 0, 30),
    ]
    for point, text, xshift, yshift in placements:
        figure.add_annotation(
            x=point[0],
            y=point[1],
            text=text,
            showarrow=False,
            font={"size": 12, "family": SCIENTIFIC_FONT, "color": "#212529"},
            xshift=xshift,
            yshift=yshift,
        )


def _add_simplex_vertex_labels_3d(figure: go.Figure, axis_labels: Sequence[str]) -> None:
    vertices = _simplex_tetrahedron_vertices()
    figure.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="text",
            text=[f"<b>{label}</b>" for label in axis_labels],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _add_simplex_vertex_labels_3d_plane(figure: go.Figure, axis_labels: Sequence[str]) -> None:
    vertices_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    figure.add_trace(
        go.Scatter3d(
            x=vertices_2d[:, 0],
            y=vertices_2d[:, 1],
            z=[0.0, 0.0, 0.0],
            mode="text",
            text=[f"<b>{label}</b>" for label in axis_labels],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _line_2d(x, y, *, color: str, width: float) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": color, "width": width},
        showlegend=False,
        hoverinfo="skip",
    )


def _line_shape(x0, y0, x1, y1, *, color: str, width: float) -> dict[str, Any]:
    return {
        "type": "line",
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "xref": "x",
        "yref": "y",
        "layer": "below",
        "line": {"color": color, "width": width},
    }


def _as_dataframe(data: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(list(data or []))


def _unaggregated_plot_data(matrix: pd.DataFrame, axes: Sequence[str]) -> pd.DataFrame:
    data = matrix[list(axes)].copy()
    data["_unique_count"] = 1
    data["_run_count"] = 1
    data["_replicate_count"] = 0
    data["_point_types"] = [_point_type(row) for row in matrix.to_dict("records")]
    data["_runs"] = _run_numbers(matrix, per_row=True)
    return data.reset_index(drop=True)


def _factor_key(factor: Any, *, fallback: str = "") -> str:
    if isinstance(factor, Mapping):
        return str(factor.get("symbol") or factor.get("name") or fallback or "")
    return str(getattr(factor, "symbol", None) or getattr(factor, "name", None) or fallback or "")


def _factor_value(factor: Any, name: str, default: Any = None) -> Any:
    if isinstance(factor, Mapping):
        return factor.get(name, default)
    return getattr(factor, name, default)


def _factor_levels(factor: Any) -> tuple[Any, ...]:
    levels = _factor_value(factor, "levels", None)
    if levels is None:
        return ()
    if isinstance(levels, np.ndarray):
        return tuple(levels.tolist())
    try:
        return tuple(levels)
    except TypeError:
        return (levels,)


def _factor_label(factor: FactorInfo, mode: str) -> str:
    if mode == "name":
        label = factor.name
    else:
        label = factor.key
    return f"{label} [{factor.unit}]" if factor.unit and mode == "symbol_unit" else label


def _axis_label_sets(axes: Sequence[str], factors: Mapping[str, FactorInfo]) -> dict[str, list[str]]:
    return {
        mode: [_factor_label(factors[axis], mode) for axis in axes]
        for mode in ("symbol", "name", "symbol_unit")
    }


def _design_plot_title(design_type: str) -> str:
    cleaned = " ".join(str(design_type or "").split()) or "Experimental"
    if cleaned.lower().endswith("design"):
        return cleaned
    return f"{cleaned} Design"


def _point_type(row: Mapping[str, Any]) -> str:
    raw = str(row.get("point_type") or "Factorial").strip().lower()
    return {
        "factorial": "Factorial",
        "center": "Center",
        "replicate": "Replicate",
        "manual": "Manually added",
        "manually added": "Manually added",
    }.get(raw, str(row.get("point_type") or "Factorial"))


def _run_numbers(matrix: pd.DataFrame, *, per_row: bool = False) -> list[str] | list[list[str]]:
    if "Run" in matrix:
        values = [str(value) for value in matrix["Run"].tolist()]
    else:
        values = [str(index + 1) for index in matrix.index]
    if per_row:
        return [[value] for value in values]
    return values


def _point_labels(plot_data: pd.DataFrame, options: DesignPlotOptions) -> list[str]:
    labels = []
    for row in plot_data.to_dict("records"):
        if options.show_run_labels:
            labels.append(", ".join(row.get("_runs") or []))
        elif options.show_replicate_count and row.get("_run_count", 1) > 1:
            labels.append(f"x{int(row.get('_run_count', 1))}")
        else:
            labels.append("")
    return labels


def _marker(plot_data: pd.DataFrame, options: DesignPlotOptions, *, size_base: int) -> dict[str, Any]:
    sizes = [
        size_base + 2.5 * min(max(int(row.get("_run_count", 1)) - 1, 0), 6)
        for row in plot_data.to_dict("records")
    ]
    return {
        "size": sizes,
        "color": _marker_color(options.marker_color),
        "symbol": "circle",
        "line": {"color": "white", "width": 1},
        "opacity": 0.9,
    }


def _marker_color(value: str | None) -> str:
    value = str(value or "").strip()
    if value.startswith("#") and len(value) in {4, 7, 9}:
        return value
    return DEFAULT_MARKER_COLOR


def _customdata(plot_data: pd.DataFrame, extra: np.ndarray | None = None) -> list[list[Any]]:
    rows = []
    for index, row in enumerate(plot_data.to_dict("records")):
        values = [
            row.get("_unique_count", 1),
            row.get("_run_count", 1),
            row.get("_replicate_count", 0),
            row.get("_point_types", "Factorial"),
            ", ".join(row.get("_runs") or []),
        ]
        if extra is not None:
            values.extend(extra[index].tolist())
        rows.append(values)
    return rows


def _hovertemplate(axes: Sequence[str], geometry: str, options: DesignPlotOptions) -> str | None:
    if not options.show_hover:
        return None
    if geometry == "process_2d":
        coordinate_lines = f"{axes[0]}: %{{x}}<br>{axes[1]}: %{{y}}"
    elif geometry == "process_3d":
        coordinate_lines = f"{axes[0]}: %{{x}}<br>{axes[1]}: %{{y}}<br>{axes[2]}: %{{z}}"
    elif geometry == "mixture_simplex_2d":
        coordinate_lines = "<br>".join(
            f"{axis}: %{{customdata[{5 + index}]:.4g}}"
            for index, axis in enumerate(axes)
        )
    elif geometry == "mixture_tetrahedron_3d":
        coordinate_lines = "<br>".join(
            f"{axis}: %{{customdata[{5 + index}]:.4g}}"
            for index, axis in enumerate(axes)
        )
    else:
        coordinate_lines = f"{axes[0]}: %{{x}}<br>{axes[1]}: %{{y}}"
    return (
        f"<b>Design Point</b><br><br>Coordinates<br>{coordinate_lines}"
        "<br><br>Experimental runs: %{customdata[1]}"
        "<br>Replicates: %{customdata[2]}"
        "<br>Exp. numbers: %{customdata[4]}<extra></extra>"
    )


__all__ = [
    "DesignPlotOptions",
    "FactorInfo",
    "aggregate_projected_points",
    "apply_projection_filters",
    "barycentric_to_cartesian_2d",
    "barycentric_to_cartesian_3d",
    "build_design_plot",
    "build_design_plot_data",
    "infer_geometry",
    "normalize_factors",
    "render_design_plot",
    "resolve_axes",
]
