# Imports
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from math import comb
from typing import Any, Collection, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# Constants and type aliases

_DEFAULT_INCLUDE = (
    "vertices",
    "edge_midpoints",
    "face_centroids",
    "global_centroid",
)
_INCLUDE_ALIASES = {
    "vertices": "vertices",
    "vertex": "vertices",
    "edge_midpoints": "edge_midpoints",
    "edge_midpoint": "edge_midpoints",
    "face_centroids": "face_centroids",
    "face_centroid": "face_centroids",
    "global_centroid": "global_centroid",
}
_SUPPORTED_INCLUDE = frozenset(_INCLUDE_ALIASES)
_DEFAULT_GRID = {"max_candidates": 100_000}


@dataclass(frozen=True)
class _Bounds:
    lower: np.ndarray
    upper: np.ndarray
    component_names: tuple[str, ...]
    total: float
    tolerance: float
    dedup_decimals: int


@dataclass(frozen=True)
class _PolytopeGeometry:
    vertices: np.ndarray
    projected_vertices: np.ndarray
    vertex_indices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]
    face_triangles: tuple[tuple[tuple[int, int, int], ...], ...]
    projection_origin: np.ndarray
    projection_basis: np.ndarray


# Input validation

def _validate_bounds(
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    component_names: Sequence[str] | None,
    total: float,
    tolerance: float,
    dedup_decimals: int,
) -> _Bounds:
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)

    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower_bounds and upper_bounds must be one-dimensional sequences.")
    if lower.size != upper.size:
        raise ValueError("lower_bounds and upper_bounds must have the same length.")
    if lower.size < 2:
        raise ValueError("At least two mixture components are required.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Bounds must contain only finite numeric values.")
    if not np.isfinite(total):
        raise ValueError("total must be a finite numeric value.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if dedup_decimals < 0:
        raise ValueError("dedup_decimals must be non-negative.")
    if np.any(lower > upper + tolerance):
        raise ValueError("Each lower bound must be less than or equal to its upper bound.")
    if lower.sum() > total + tolerance:
        raise ValueError("Infeasible mixture domain: sum(lower_bounds) exceeds total.")
    if upper.sum() < total - tolerance:
        raise ValueError("Infeasible mixture domain: sum(upper_bounds) is below total.")

    if component_names is None:
        names = tuple(f"x{i + 1}" for i in range(lower.size))
    else:
        names = tuple(str(name) for name in component_names)
        if len(names) != lower.size:
            raise ValueError("component_names must have the same length as bounds.")
        if len(set(names)) != len(names):
            raise ValueError("component_names must be unique.")
        if any(not name for name in names):
            raise ValueError("component_names cannot contain empty names.")

    return _Bounds(
        lower=lower,
        upper=upper,
        component_names=names,
        total=float(total),
        tolerance=float(tolerance),
        dedup_decimals=int(dedup_decimals),
    )


def _validate_include(include: Collection[str]) -> tuple[str, ...]:
    selected = tuple(str(item) for item in include)
    unknown = sorted(set(selected) - _SUPPORTED_INCLUDE)
    if unknown:
        raise ValueError(
            "Unsupported candidate category: "
            + ", ".join(unknown)
            + ". Supported categories are: "
            + ", ".join(sorted(_SUPPORTED_INCLUDE))
        )
    normalized = []
    for item in selected:
        category = _INCLUDE_ALIASES[item]
        if category not in normalized:
            normalized.append(category)
    return tuple(normalized)


def _normalize_grid_config(grid: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if grid is None:
        return None
    if not isinstance(grid, Mapping):
        raise TypeError("grid must be None or a mapping.")

    config = dict(_DEFAULT_GRID)
    config.update(grid)
    allowed_options = {"degree", "max_candidates"}
    unknown = set(config) - allowed_options
    if unknown:
        raise ValueError("Unsupported grid option(s): " + ", ".join(sorted(unknown)))
    if "degree" not in config:
        raise ValueError("grid['degree'] is required.")
    if (
        not isinstance(config["degree"], int)
        or isinstance(config["degree"], bool)
        or config["degree"] < 1
    ):
        raise ValueError("grid['degree'] must be an integer greater than or equal to 1.")
    if (
        not isinstance(config["max_candidates"], int)
        or isinstance(config["max_candidates"], bool)
        or config["max_candidates"] < 1
    ):
        raise ValueError("grid['max_candidates'] must be a positive integer.")
    return config


def _require_four_component_geometry(bounds: _Bounds, category: str) -> None:
    if bounds.lower.size not in {2, 3, 4}:
        raise NotImplementedError(
            f"{category} generation is currently supported only for two-, "
            "three-, or four-component mixtures."
        )


# Vertex generation

def _generate_vertices_from_bounds(bounds: _Bounds) -> np.ndarray:
    n_components = bounds.lower.size
    rows = []

    for dependent in range(n_components):
        independent = [idx for idx in range(n_components) if idx != dependent]
        for choices in itertools.product((0, 1), repeat=n_components - 1):
            point = np.empty(n_components, dtype=float)
            for idx, choice in zip(independent, choices):
                point[idx] = bounds.upper[idx] if choice else bounds.lower[idx]
            point[dependent] = bounds.total - point[independent].sum()
            if _is_feasible_point(point, bounds):
                rows.append(_clip_near_bounds(point, bounds))

    if not rows:
        raise ValueError("No extreme vertices were found for the requested mixture domain.")
    return _unique_points(np.vstack(rows), bounds.dedup_decimals)


# Polytope projection and geometry

def _project_to_affine_space(vertices: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = vertices.mean(axis=0)
    centered = vertices - origin
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        rank = 0
    else:
        threshold = tolerance * max(1.0, singular_values[0])
        rank = int(np.sum(singular_values > threshold))
    basis = vh[:rank].T
    projected = centered @ basis
    return projected, origin, basis


def _build_polytope_geometry(vertices: np.ndarray, bounds: _Bounds) -> _PolytopeGeometry:
    _require_four_component_geometry(bounds, "Polytope geometry")
    projected, origin, basis = _project_to_affine_space(vertices, bounds.tolerance)

    if projected.shape[1] == 1:
        return _build_segment_geometry(vertices, projected, origin, basis, bounds)
    if projected.shape[1] == 2:
        return _build_polygon_geometry(vertices, projected, origin, basis, bounds)
    if projected.shape[1] != 3:
        raise NotImplementedError(
            "Edge, face, and centroid generation requires a two- or "
            "three-dimensional affine mixture domain."
        )
    if len(vertices) < 4:
        raise ValueError("At least four non-coplanar vertices are required for 3D geometry.")

    hull = ConvexHull(projected)
    face_triangles = _group_coplanar_hull_triangles(hull, bounds.tolerance)
    faces = tuple(tuple(sorted({idx for triangle in group for idx in triangle})) for group in face_triangles)
    edges = _extract_true_edges_from_faces(face_triangles)

    return _PolytopeGeometry(
        vertices=vertices,
        projected_vertices=projected,
        vertex_indices=tuple(range(len(vertices))),
        edges=edges,
        faces=faces,
        face_triangles=face_triangles,
        projection_origin=origin,
        projection_basis=basis,
    )


def _build_segment_geometry(
    vertices: np.ndarray,
    projected: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    bounds: _Bounds,
) -> _PolytopeGeometry:
    if len(vertices) < 2:
        raise ValueError("At least two vertices are required for 1D mixture geometry.")

    order = np.argsort(projected[:, 0])
    edge = tuple(int(idx) for idx in order[[0, -1]])
    return _PolytopeGeometry(
        vertices=vertices,
        projected_vertices=projected,
        vertex_indices=tuple(range(len(vertices))),
        edges=(tuple(sorted(edge)),),
        faces=(),
        face_triangles=(),
        projection_origin=origin,
        projection_basis=basis,
    )


def _build_polygon_geometry(
    vertices: np.ndarray,
    projected: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    bounds: _Bounds,
) -> _PolytopeGeometry:
    if len(vertices) < 3:
        raise ValueError("At least three vertices are required for 2D mixture geometry.")

    hull = ConvexHull(projected)
    ordered_vertices = tuple(int(idx) for idx in hull.vertices)
    face_triangles = (_triangulate_polygon_face(ordered_vertices),)
    edges = tuple(sorted(tuple(sorted((int(i), int(j)))) for i, j in hull.simplices))

    return _PolytopeGeometry(
        vertices=vertices,
        projected_vertices=projected,
        vertex_indices=tuple(range(len(vertices))),
        edges=edges,
        faces=(ordered_vertices,),
        face_triangles=face_triangles,
        projection_origin=origin,
        projection_basis=basis,
    )


def _triangulate_polygon_face(vertices: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    if len(vertices) < 3:
        raise ValueError("A polygon face requires at least three vertices.")
    anchor = vertices[0]
    return tuple((anchor, vertices[i], vertices[i + 1]) for i in range(1, len(vertices) - 1))


def _group_coplanar_hull_triangles(
    hull: ConvexHull,
    tolerance: float,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    plane_tolerance = max(tolerance * 100.0, 1e-9)
    groups: list[dict[str, Any]] = []

    for simplex, equation in zip(hull.simplices, hull.equations):
        normal, offset = _canonical_plane(equation[:-1], equation[-1], plane_tolerance)
        triangle = tuple(int(idx) for idx in simplex)
        matched = False
        for group in groups:
            if (
                np.linalg.norm(group["normal"] - normal) <= plane_tolerance
                and abs(group["offset"] - offset) <= plane_tolerance
            ):
                group["triangles"].append(triangle)
                matched = True
                break
        if not matched:
            groups.append({"normal": normal, "offset": offset, "triangles": [triangle]})

    groups.sort(key=lambda item: min(min(triangle) for triangle in item["triangles"]))
    return tuple(tuple(group["triangles"]) for group in groups)


def _canonical_plane(
    normal: np.ndarray,
    offset: float,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    norm = np.linalg.norm(normal)
    if norm <= tolerance:
        raise ValueError("Degenerate hull plane encountered while building polytope geometry.")
    normal = normal / norm
    offset = float(offset / norm)

    pivot_candidates = np.flatnonzero(np.abs(normal) > tolerance)
    if pivot_candidates.size and normal[pivot_candidates[0]] < 0:
        normal = -normal
        offset = -offset
    return normal, offset


def _extract_true_edges_from_faces(
    face_triangles: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for triangles in face_triangles:
        counts: Counter[tuple[int, int]] = Counter()
        for a, b, c in triangles:
            for edge in ((a, b), (b, c), (c, a)):
                counts[tuple(sorted(edge))] += 1
        for edge, count in counts.items():
            if count == 1:
                edges.add(edge)
    return tuple(sorted(edges))


# Geometric candidate generators

def _generate_vertex_candidates(vertices: np.ndarray, bounds: _Bounds) -> pd.DataFrame:
    return _points_to_frame(vertices, "vertex", "V", bounds.component_names)


def _generate_edge_midpoints(geometry: _PolytopeGeometry, bounds: _Bounds) -> pd.DataFrame:
    if not geometry.edges:
        return _points_to_frame(
            np.empty((0, len(bounds.component_names))),
            "edge_midpoint",
            "E",
            bounds.component_names,
        )

    points = np.array(
        [(geometry.vertices[i] + geometry.vertices[j]) / 2.0 for i, j in geometry.edges],
        dtype=float,
    )
    return _points_to_frame(points, "edge_midpoint", "E", bounds.component_names)


def _generate_face_centroids(geometry: _PolytopeGeometry, bounds: _Bounds) -> pd.DataFrame:
    if not geometry.face_triangles:
        return _points_to_frame(
            np.empty((0, len(bounds.component_names))),
            "face_centroid",
            "F",
            bounds.component_names,
        )

    points = []
    for triangles in geometry.face_triangles:
        weighted_sum = np.zeros(geometry.vertices.shape[1], dtype=float)
        total_area = 0.0
        for triangle in triangles:
            projected_triangle = geometry.projected_vertices[list(triangle)]
            area = _triangle_area(projected_triangle)
            centroid = geometry.vertices[list(triangle)].mean(axis=0)
            weighted_sum += area * centroid
            total_area += area
        if total_area <= bounds.tolerance:
            raise ValueError("Degenerate face encountered while computing face centroids.")
        points.append(weighted_sum / total_area)
    return _points_to_frame(np.vstack(points), "face_centroid", "F", bounds.component_names)


def _generate_global_centroid(geometry: _PolytopeGeometry, bounds: _Bounds) -> pd.DataFrame:
    if geometry.projected_vertices.shape[1] == 1:
        point = geometry.vertices.mean(axis=0)
        return _points_to_frame(
            point.reshape(1, -1),
            "global_centroid",
            "C",
            bounds.component_names,
            start=0,
        )

    if geometry.projected_vertices.shape[1] == 2:
        point = _polygon_centroid_from_triangles(geometry, bounds)
        return _points_to_frame(
            point.reshape(1, -1),
            "global_centroid",
            "C",
            bounds.component_names,
            start=0,
        )

    origin_projected = geometry.projected_vertices.mean(axis=0)
    origin_original = geometry.vertices.mean(axis=0)
    weighted_sum = np.zeros(geometry.vertices.shape[1], dtype=float)
    total_volume = 0.0

    for triangles in geometry.face_triangles:
        for triangle in triangles:
            simplex = geometry.projected_vertices[list(triangle)]
            volume = _tetrahedron_volume(origin_projected, simplex)
            centroid = (origin_original + geometry.vertices[list(triangle)].sum(axis=0)) / 4.0
            weighted_sum += volume * centroid
            total_volume += volume

    if total_volume <= bounds.tolerance:
        raise ValueError("Degenerate polytope encountered while computing the global centroid.")
    point = weighted_sum / total_volume
    return _points_to_frame(point.reshape(1, -1), "global_centroid", "C", bounds.component_names, start=0)


def _polygon_centroid_from_triangles(
    geometry: _PolytopeGeometry,
    bounds: _Bounds,
) -> np.ndarray:
    weighted_sum = np.zeros(geometry.vertices.shape[1], dtype=float)
    total_area = 0.0
    for triangle in geometry.face_triangles[0]:
        projected_triangle = geometry.projected_vertices[list(triangle)]
        area = _triangle_area_2d(projected_triangle)
        centroid = geometry.vertices[list(triangle)].mean(axis=0)
        weighted_sum += area * centroid
        total_area += area
    if total_area <= bounds.tolerance:
        raise ValueError("Degenerate polygon encountered while computing the centroid.")
    return weighted_sum / total_area


def _triangle_area(triangle: np.ndarray) -> float:
    if triangle.shape[1] == 2:
        return _triangle_area_2d(triangle)
    a, b, c = triangle
    return float(np.linalg.norm(np.cross(b - a, c - a)) / 2.0)


def _triangle_area_2d(triangle: np.ndarray) -> float:
    a, b, c = triangle
    ab = b - a
    ac = c - a
    return float(abs(ab[0] * ac[1] - ab[1] * ac[0]) / 2.0)


def _tetrahedron_volume(origin: np.ndarray, triangle: np.ndarray) -> float:
    a, b, c = triangle
    return float(abs(np.dot(a - origin, np.cross(b - origin, c - origin))) / 6.0)


# Grid generation

def _integer_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for suffix in _integer_compositions(total - value, parts - 1):
            yield (value, *suffix)


def _generate_feasible_grid(bounds: _Bounds, grid_config: Mapping[str, Any]) -> pd.DataFrame:
    n_components = bounds.lower.size
    degree = int(grid_config["degree"])
    max_candidates = int(grid_config["max_candidates"])
    candidate_count = comb(degree + n_components - 1, n_components - 1)
    if candidate_count > max_candidates:
        raise ValueError(
            f"Requested lattice may generate {candidate_count} raw candidates, "
            f"which exceeds max_candidates={max_candidates}."
        )

    slack = bounds.total - bounds.lower.sum()
    rows = []
    for composition in _integer_compositions(degree, n_components):
        z = np.asarray(composition, dtype=float) / degree
        point = bounds.lower + slack * z
        if _is_feasible_point(point, bounds):
            rows.append(_clip_near_bounds(point, bounds))

    if not rows:
        return _points_to_frame(np.empty((0, n_components)), "grid", "G", bounds.component_names)
    points = _unique_points(np.vstack(rows), bounds.dedup_decimals)
    return _points_to_frame(points, "grid", "G", bounds.component_names)


# Candidate deduplication

def _points_to_frame(
    points: np.ndarray,
    point_type: str,
    source_prefix: str,
    component_names: Sequence[str],
    *,
    start: int = 1,
) -> pd.DataFrame:
    frame = pd.DataFrame(points, columns=component_names)
    frame.insert(0, "source_id", [f"{source_prefix}{idx}" for idx in range(start, start + len(frame))])
    frame.insert(0, "point_type", point_type)
    return frame


def _deduplicate_candidate_points(
    candidates: pd.DataFrame,
    component_names: Sequence[str],
    dedup_decimals: int,
) -> pd.DataFrame:
    columns = ["point_id", "point_type", "source_id", *component_names]
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    index_by_key: dict[tuple[float, ...], int] = {}

    for _, row in candidates.iterrows():
        coords = row[list(component_names)].to_numpy(dtype=float)
        key = tuple(np.round(coords, dedup_decimals))
        if key not in index_by_key:
            index_by_key[key] = len(rows)
            rows.append(row.to_dict())
            continue

        existing = rows[index_by_key[key]]
        existing["point_type"] = _merge_pipe_values(existing["point_type"], row["point_type"])
        existing["source_id"] = _merge_pipe_values(existing["source_id"], row["source_id"])

    deduped = pd.DataFrame(rows)
    deduped.insert(0, "point_id", [f"P{idx}" for idx in range(1, len(deduped) + 1)])
    return deduped[columns]


def _merge_pipe_values(left: Any, right: Any) -> str:
    values = []
    for item in (left, right):
        for value in str(item).split("|"):
            if value not in values:
                values.append(value)
    return "|".join(values)


def _validate_candidate_points(candidates: pd.DataFrame, bounds: _Bounds) -> None:
    points = candidates[list(bounds.component_names)].to_numpy(dtype=float)
    if len(points) == 0:
        raise ValueError("Candidate set is empty.")
    if not np.all(points >= bounds.lower - bounds.tolerance):
        raise RuntimeError("Internal error: candidate set contains points below lower bounds.")
    if not np.all(points <= bounds.upper + bounds.tolerance):
        raise RuntimeError("Internal error: candidate set contains points above upper bounds.")
    if not np.allclose(points.sum(axis=1), bounds.total, atol=bounds.tolerance, rtol=0.0):
        raise RuntimeError("Internal error: candidate set contains points that do not sum to total.")


def _is_feasible_point(point: np.ndarray, bounds: _Bounds) -> bool:
    return bool(
        np.all(point >= bounds.lower - bounds.tolerance)
        and np.all(point <= bounds.upper + bounds.tolerance)
        and np.isclose(point.sum(), bounds.total, atol=bounds.tolerance, rtol=0.0)
    )


def _clip_near_bounds(point: np.ndarray, bounds: _Bounds) -> np.ndarray:
    clipped = np.clip(point, bounds.lower, bounds.upper)
    if abs(clipped.sum() - bounds.total) <= bounds.tolerance:
        return clipped
    return point


def _unique_points(points: np.ndarray, dedup_decimals: int) -> np.ndarray:
    if len(points) == 0:
        return points
    rounded = np.round(points, dedup_decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


# Public wrapper

def build_candidate_points(
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    *,
    component_names: Sequence[str] | None = None,
    total: float = 1.0,
    include: Collection[str] = _DEFAULT_INCLUDE,
    grid: Mapping[str, Any] | None = None,
    tolerance: float = 1e-10,
    dedup_decimals: int = 12,
) -> pd.DataFrame:
    """
    Build geometric and lattice candidate points for a bounded mixture design.

    The feasible region is defined by component bounds and the mixture-sum
    constraint ``sum(x_i) = total``. The function only builds a unique candidate
    set; D-optimal point selection and experimental replicates are handled
    elsewhere.

    ``grid={"degree": n}`` adds a simplex-lattice candidate set. Lattice points
    are generated in slack coordinates and then filtered by upper bounds.
    """

    bounds = _validate_bounds(
        lower_bounds,
        upper_bounds,
        component_names,
        total,
        tolerance,
        dedup_decimals,
    )
    selected = _validate_include(include)
    grid_config = _normalize_grid_config(grid)

    vertices = _generate_vertices_from_bounds(bounds)
    geometry = None
    if any(category in selected for category in ("edge_midpoints", "face_centroids", "global_centroid")):
        geometry = _build_polytope_geometry(vertices, bounds)

    frames = []
    if "vertices" in selected:
        frames.append(_generate_vertex_candidates(vertices, bounds))
    if "edge_midpoints" in selected:
        frames.append(_generate_edge_midpoints(geometry, bounds))
    if "face_centroids" in selected:
        frames.append(_generate_face_centroids(geometry, bounds))
    if "global_centroid" in selected:
        frames.append(_generate_global_centroid(geometry, bounds))
    if grid_config is not None:
        frames.append(_generate_feasible_grid(bounds, grid_config))

    if not frames:
        raise ValueError("No candidate categories were requested.")

    candidates = pd.concat(frames, ignore_index=True)
    candidates = _deduplicate_candidate_points(
        candidates,
        bounds.component_names,
        bounds.dedup_decimals,
    )
    _validate_candidate_points(candidates, bounds)
    return candidates
