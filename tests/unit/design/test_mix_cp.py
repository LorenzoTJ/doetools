from math import comb

import numpy as np
import pytest

from doetools.design.mixture.mixture_cp_generator import build_candidate_points


OCTAHEDRON_LOWER = [0.098, 0.302, 0.140, 0.420]
OCTAHEDRON_UPPER = [0.118, 0.322, 0.160, 0.440]
NAMES = ["A", "B", "C", "D"]


def _coordinates(frame):
    return frame[NAMES].to_numpy(dtype=float)


def test_octahedron_full_geometric_candidate_set():
    candidates = build_candidate_points(
        OCTAHEDRON_LOWER,
        OCTAHEDRON_UPPER,
        component_names=NAMES,
    )

    assert len(candidates) == 27
    assert candidates.groupby("point_type").size().to_dict() == {
        "edge_midpoint": 12,
        "face_centroid": 8,
        "global_centroid": 1,
        "vertex": 6,
    }
    assert np.allclose(_coordinates(candidates).sum(axis=1), 1.0)
    assert np.all(_coordinates(candidates) >= np.asarray(OCTAHEDRON_LOWER) - 1e-10)
    assert np.all(_coordinates(candidates) <= np.asarray(OCTAHEDRON_UPPER) + 1e-10)


def test_octahedron_vertices_match_expected_points():
    candidates = build_candidate_points(
        OCTAHEDRON_LOWER,
        OCTAHEDRON_UPPER,
        component_names=NAMES,
        include={"vertices"},
    )
    expected = np.array(
        [
            [0.098, 0.302, 0.160, 0.440],
            [0.098, 0.322, 0.140, 0.440],
            [0.098, 0.322, 0.160, 0.420],
            [0.118, 0.302, 0.140, 0.440],
            [0.118, 0.302, 0.160, 0.420],
            [0.118, 0.322, 0.140, 0.420],
        ]
    )

    actual = {tuple(row) for row in np.round(_coordinates(candidates), 12)}
    expected = {tuple(row) for row in np.round(expected, 12)}
    assert actual == expected


def test_untruncated_tetrahedron_counts():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        component_names=NAMES,
    )

    assert candidates.groupby("point_type").size().to_dict() == {
        "edge_midpoint": 6,
        "face_centroid": 4,
        "global_centroid": 1,
        "vertex": 4,
    }
    assert len(candidates) == 15


def test_truncated_tetrahedron_has_non_triangular_faces():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0, 0.0],
        [0.7, 0.7, 0.7, 0.7],
        component_names=NAMES,
    )

    counts = candidates.groupby("point_type").size().to_dict()
    assert counts["vertex"] == 12
    assert counts["edge_midpoint"] == 18
    assert counts["face_centroid"] == 8
    assert counts["global_centroid"] == 1


def test_vertices_plus_edge_midpoints_only():
    candidates = build_candidate_points(
        OCTAHEDRON_LOWER,
        OCTAHEDRON_UPPER,
        component_names=NAMES,
        include={"vertices", "edge_midpoints"},
    )

    assert candidates.groupby("point_type").size().to_dict() == {
        "edge_midpoint": 12,
        "vertex": 6,
    }
    assert len(candidates) == 18


def test_three_component_full_simplex_geometry():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        component_names=["A", "B", "C"],
        include={"vertices", "edge_midpoints", "face_centroid"},
    )

    assert candidates.groupby("point_type").size().to_dict() == {
        "edge_midpoint": 3,
        "face_centroid": 1,
        "vertex": 3,
    }
    assert len(candidates) == 7
    assert np.allclose(candidates[["A", "B", "C"]].sum(axis=1), 1.0)


def test_three_component_truncated_polygon_geometry():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0],
        [0.7, 0.7, 0.7],
        component_names=["A", "B", "C"],
        include={"vertices", "edge_midpoints", "face_centroids"},
    )

    assert candidates.groupby("point_type").size().to_dict() == {
        "edge_midpoint": 6,
        "face_centroid": 1,
        "vertex": 6,
    }
    assert len(candidates) == 13
    assert np.allclose(candidates[["A", "B", "C"]].sum(axis=1), 1.0)


def test_two_component_full_simplex_geometry():
    candidates = build_candidate_points(
        [0.0, 0.0],
        [1.0, 1.0],
        component_names=["A", "B"],
    )

    assert len(candidates) == 3
    assert candidates.groupby("point_type").size().to_dict() == {
        "edge_midpoint|global_centroid": 1,
        "vertex": 2,
    }
    assert np.allclose(candidates[["A", "B"]].sum(axis=1), 1.0)


def test_grid_is_feasible_and_deduplicated_with_geometric_points():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        component_names=NAMES,
        grid={"degree": 2, "max_candidates": 1_000},
    )

    coords = _coordinates(candidates)
    rounded = np.round(coords, 12)
    assert len(np.unique(rounded, axis=0)) == len(candidates)
    assert np.allclose(coords.sum(axis=1), 1.0)
    assert any("grid" in point_type for point_type in candidates["point_type"])


def test_three_component_grid_degree_three_is_simplex_lattice():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        component_names=["A", "B", "C"],
        include=(),
        grid={"degree": 3, "max_candidates": 1_000},
    )

    coords = candidates[["A", "B", "C"]].to_numpy(dtype=float)
    assert len(candidates) == comb(3 + 3 - 1, 3 - 1)
    assert np.allclose(coords.sum(axis=1), 1.0)
    assert np.allclose(coords * 3, np.round(coords * 3))


def test_three_component_grid_filters_upper_bounds_symmetrically():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0],
        [0.7, 0.7, 0.7],
        component_names=["A", "B", "C"],
        include=(),
        grid={"degree": 3, "max_candidates": 1_000},
    )

    coords = candidates[["A", "B", "C"]].to_numpy(dtype=float)
    expected = {
        (0.0, 1 / 3, 2 / 3),
        (0.0, 2 / 3, 1 / 3),
        (1 / 3, 0.0, 2 / 3),
        (1 / 3, 1 / 3, 1 / 3),
        (1 / 3, 2 / 3, 0.0),
        (2 / 3, 0.0, 1 / 3),
        (2 / 3, 1 / 3, 0.0),
    }
    actual = {tuple(row) for row in np.round(coords, 12)}
    assert actual == {tuple(np.round(row, 12)) for row in expected}
    assert np.all(coords <= 0.7 + 1e-10)


def test_four_component_grid_degree_two_is_simplex_lattice():
    candidates = build_candidate_points(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        component_names=NAMES,
        include=(),
        grid={"degree": 2, "max_candidates": 1_000},
    )

    coords = _coordinates(candidates)
    assert len(candidates) == comb(2 + 4 - 1, 4 - 1)
    assert np.allclose(coords.sum(axis=1), 1.0)
    assert np.allclose(coords * 2, np.round(coords * 2))


def test_four_component_grid_uses_slack_lattice_with_bounds():
    lower = np.array([0.1, 0.2, 0.3, 0.1])
    candidates = build_candidate_points(
        lower,
        [0.4, 0.5, 0.6, 0.4],
        component_names=NAMES,
        include=(),
        grid={"degree": 2, "max_candidates": 1_000},
    )

    coords = _coordinates(candidates)
    slack = 1.0 - lower.sum()
    z = (coords - lower) / slack
    assert np.allclose(coords.sum(axis=1), 1.0)
    assert np.all(coords >= lower - 1e-10)
    assert np.all(coords <= np.array([0.4, 0.5, 0.6, 0.4]) + 1e-10)
    assert np.allclose(z * 2, np.round(z * 2))


def test_grid_limit_raises_clear_error():
    with pytest.raises(ValueError, match="exceeds max_candidates"):
        build_candidate_points(
            OCTAHEDRON_LOWER,
            OCTAHEDRON_UPPER,
            component_names=NAMES,
            include={"vertices"},
            grid={"degree": 5, "max_candidates": 10},
        )


def test_grid_requires_mapping_with_degree():
    with pytest.raises(TypeError, match="grid must be None or a mapping"):
        build_candidate_points(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            include=(),
            grid=True,
        )

    with pytest.raises(ValueError, match="Unsupported grid option"):
        build_candidate_points(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            include=(),
            grid={"n_levels": 4},
        )

    with pytest.raises(ValueError, match="Unsupported grid option"):
        build_candidate_points(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            include=(),
            grid={"method": "symmetric", "degree": 4},
        )


def test_inconsistent_bounds_raise_error():
    with pytest.raises(ValueError, match="less than or equal"):
        build_candidate_points([0.2, 0.5], [0.1, 0.9])


def test_empty_domain_raises_error():
    with pytest.raises(ValueError, match="sum\\(lower_bounds\\) exceeds total"):
        build_candidate_points([0.6, 0.6], [1.0, 1.0])


def test_unsupported_geometry_dimension_raises_error():
    with pytest.raises(NotImplementedError, match="two-, three-, or four-component mixtures"):
        build_candidate_points(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            include={"vertices", "edge_midpoints"},
        )
