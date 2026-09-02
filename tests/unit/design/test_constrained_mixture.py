import numpy as np
import pytest

from doetools import ConstrainedMixtureDesign, MixtureFactor
from doetools.design.mixture import ConstrainedMixtureDesign as MixtureExport


def _octahedron_factors():
    return {
        "A": MixtureFactor(lower_bound=0.098, upper_bound=0.118),
        "B": MixtureFactor(lower_bound=0.302, upper_bound=0.322),
        "C": MixtureFactor(lower_bound=0.140, upper_bound=0.160),
        "D": MixtureFactor(lower_bound=0.420, upper_bound=0.440),
    }


def test_constrained_mixture_design_replaces_extreme_vertices_export():
    assert MixtureExport is ConstrainedMixtureDesign


def test_octahedron_design_contains_boundary_geometry_without_centroid():
    design = ConstrainedMixtureDesign(_octahedron_factors())
    matrix = design._coded_design_matrix

    assert design._design_type == "Constrained Mixture"
    assert len(matrix) == 26
    assert np.allclose(matrix[["A", "B", "C", "D"]].sum(axis=1), 1.0)
    coords = matrix[["A", "B", "C", "D"]].to_numpy(dtype=float)
    assert np.all(coords >= np.array([0.098, 0.302, 0.140, 0.420]) - 1e-10)
    assert np.all(coords <= np.array([0.118, 0.322, 0.160, 0.440]) + 1e-10)


def test_center_points_add_global_center_after_boundary_geometry():
    design = ConstrainedMixtureDesign(_octahedron_factors(), center_points=2)
    matrix = design._coded_design_matrix

    center = np.array([0.108, 0.312, 0.150, 0.430])
    assert len(matrix) == 28
    assert np.allclose(matrix.tail(2)[["A", "B", "C", "D"]].to_numpy(), center)


def test_center_points_use_constrained_centroid_with_asymmetric_upper_bounds():
    factors = {
        "A": MixtureFactor(lower_bound=0.0, upper_bound=0.1),
        "B": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        "C": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        "D": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
    }

    design = ConstrainedMixtureDesign(factors, center_points=1)
    center = design._coded_design_matrix.tail(1)[["A", "B", "C", "D"]].to_numpy(dtype=float)

    assert center[0, 0] <= 0.1 + 1e-10
    assert np.allclose(center.sum(axis=1), 1.0)


def test_structure_can_request_vertices_only():
    design = ConstrainedMixtureDesign(_octahedron_factors(), structure="vertices")
    matrix = design._coded_design_matrix

    assert len(matrix) == 6


def test_structure_accepts_face_centroid_alias():
    design = ConstrainedMixtureDesign(
        _octahedron_factors(),
        structure={"vertices", "edge_midpoints", "face_centroid"},
    )

    assert len(design._coded_design_matrix) == 26


def test_three_component_mixture_is_supported():
    factors = {
        "A": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        "B": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        "C": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
    }

    design = ConstrainedMixtureDesign(factors)
    matrix = design._coded_design_matrix

    assert len(matrix) == 7
    assert np.allclose(matrix[["A", "B", "C"]].sum(axis=1), 1.0)


def test_three_component_constrained_polygon_is_supported():
    factors = {
        "A": MixtureFactor(lower_bound=0.0, upper_bound=0.7),
        "B": MixtureFactor(lower_bound=0.0, upper_bound=0.7),
        "C": MixtureFactor(lower_bound=0.0, upper_bound=0.7),
    }

    design = ConstrainedMixtureDesign(factors)
    matrix = design._coded_design_matrix

    assert len(matrix) == 13
    assert np.allclose(matrix[["A", "B", "C"]].sum(axis=1), 1.0)


def test_replicates_repeat_boundary_points_before_center_points():
    design = ConstrainedMixtureDesign(_octahedron_factors(), replicates=1, center_points=1)
    matrix = design._coded_design_matrix

    assert len(matrix) == 53
    duplicated_boundary = matrix.iloc[:52]
    assert len(duplicated_boundary.drop_duplicates()) == 26


def test_rejects_unsupported_component_count():
    factors = {
        "A": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        "B": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
        "C": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
    }

    factors["D"] = MixtureFactor(lower_bound=0.0, upper_bound=1.0)
    factors["E"] = MixtureFactor(lower_bound=0.0, upper_bound=1.0)

    with pytest.raises(NotImplementedError, match="three- or four-component"):
        ConstrainedMixtureDesign(factors)


def test_rejects_empty_domain():
    factors = {
        "A": MixtureFactor(lower_bound=0.4, upper_bound=1.0),
        "B": MixtureFactor(lower_bound=0.4, upper_bound=1.0),
        "C": MixtureFactor(lower_bound=0.4, upper_bound=1.0),
        "D": MixtureFactor(lower_bound=0.0, upper_bound=1.0),
    }

    with pytest.raises(ValueError, match="Sum of lower_bounds"):
        ConstrainedMixtureDesign(factors)
