import numpy as np
import pandas as pd
import pytest

from doetools.utils.grid_builder import lhs_grid, rectangular_grid, mixture_grid


class DummyFactor:
    def __init__(self, lower_bound: float, upper_bound: float, factor_type: str = "mix"):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.type = factor_type


# ==============================================================================
#                          Test lhs_grid
# ==============================================================================

# Test lhs_grid basic shape and range
@pytest.mark.parametrize(("vars_", "n"), [
    (["A"], 5),
    (["A", "B"], 10),
    (["X", "Y", "Z"], 15),
])
def test_lhs_grid_basic_shape_and_range(vars_, n):
    lhs = lhs_grid(vars_, n_of_exp=n, random_state=123)
    assert list(lhs.columns) == vars_
    assert lhs.shape == (n, len(vars_))
    assert (lhs.values >= -1).all() and (lhs.values <= 1).all()

# Test lhs_grid reproducibility
@pytest.mark.parametrize(("vars_", "n"), [
    (["X", "Y"], 20)])
def test_lhs_grid_reproducibility(vars_, n):
    lhs1 = lhs_grid(vars_, n_of_exp=n, random_state=7)
    lhs2 = lhs_grid(vars_, n_of_exp=n, random_state=7)

    pd.testing.assert_frame_equal(lhs1, lhs2)
        
# Test lhs_grid stratification
@pytest.mark.parametrize(("vars_", "n"), [
    (["X", "Y"], 20)])
def test_lhs_grid_stratification(vars_, n):
    lhs1 = lhs_grid(vars_, n_of_exp=n, random_state=7)

    edges = np.linspace(-1, 1, n + 1)
    for col in lhs1.columns:
        bin_idx = np.digitize(lhs1[col], edges, right=True)
        unique, counts = np.unique(bin_idx, return_counts=True)
        # Each interval should be occupied exactly once per column
        assert len(unique) == n
        assert (counts == 1).all()


# ==============================================================================
#                          Test rectangular_grid
# ==============================================================================

# Test different designs and constant settings
@pytest.mark.parametrize(("design", "constants"), [
    (pd.DataFrame({"A": [0], "B": [0]}), {}),
    (pd.DataFrame({"A": [0], "B": [0], "C": [0]}), {"C" : 0.5}),
    (pd.DataFrame({"A": [0], "B": [0], "C": [0], "D": [0]}), {"C": 0.5, "D": 0.5}),])

def test_rectangular_grid_basic(design, constants):
    
    grid = rectangular_grid(
        design_matrix=design,
        x="A",
        y="B",
        constants=constants,
        x_min=-1,
        x_max=1,
        y_min=-1,
        y_max=1,
        resolution=10,
    )

    assert grid.shape == (100, design.shape[1])
    assert grid["A"].nunique() == 10
    assert grid["B"].nunique() == 10
    assert list(grid.columns) == list(design.columns)
    for const_var in constants.keys():
        assert grid[const_var].nunique() == 1 and np.isclose(grid[const_var].iloc[0], constants[const_var])
    assert np.isclose(grid["A"].min(), -1) and np.isclose(grid["A"].max(), 1)
    assert np.isclose(grid["B"].min(), -1) and np.isclose(grid["B"].max(), 1)

# Test different resolutions
@pytest.mark.parametrize("resolution", [[5], [20], [50]])
def test_rectangular_grid_different_resolution(resolution):
    design = pd.DataFrame({"X": [0], "Y": [0]})

    for res in resolution:
        grid = rectangular_grid(
            design_matrix=design,
            x="X",
            y="Y",
            constants={},
            x_min=-1,
            x_max=1,
            y_min=-1,
            y_max=1,
            resolution=res,
        )

        assert grid.shape == (res * res, design.shape[1])
        assert grid["X"].nunique() == res
        assert grid["Y"].nunique() == res

# Test different bounds
@pytest.mark.parametrize(("x_min", "x_max", "y_min", "y_max"), [
    (-2, 2, -2, 2),
    (0, 5, -3, 3),
    (-1, 1, 0, 4)])
def test_rectangular_grid_different_bounds(x_min, x_max, y_min, y_max):
    design = pd.DataFrame({"X": [0], "Y": [0]})

    grid = rectangular_grid(
        design_matrix=design,
        x="X",
        y="Y",
        constants={},
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=10,
    )

    assert np.isclose(grid["X"].min(), x_min) and np.isclose(grid["X"].max(), x_max)
    assert np.isclose(grid["Y"].min(), y_min) and np.isclose(grid["Y"].max(), y_max)
    

# ==============================================================================
#                          Test mixture_grid
# ==============================================================================

def _make_factors(bounds):
    """Helper to create dummy factor objects for mixture testing."""
    return {name: DummyFactor(lb, ub) for name, (lb, ub) in bounds.items()}


def test_mixture_grid_basic_simplex_and_bounds():
    """Test basic properties: simplex constraint and bounds satisfaction."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X3", "X1", "X2"])  # test column reordering

    grid = mixture_grid(
        design_matrix=design,
        factors=factors,
        mix_factors=["X1", "X2", "X3"],
        m=2,
    )

    # For m=2 and k=3, expect 6 lattice points (C(m+k-1, k-1))
    assert grid.shape == (6, 3)
    assert list(grid.columns) == ["X3", "X1", "X2"]
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert (grid >= 0).all().all() and (grid <= 1).all().all()


def test_mixture_grid_known_points_m2_unconstrained():
    """Test against analytically computed simplex-lattice points for m=2."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=2)
    
    # For {2,3} simplex-lattice, the theoretical points are:
    # Vertices: (1,0,0), (0,1,0), (0,0,1)
    # Edge midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    expected = np.array([
        [1.0, 0.0, 0.0],  # Pure X1
        [0.0, 1.0, 0.0],  # Pure X2
        [0.0, 0.0, 1.0],  # Pure X3
        [0.5, 0.5, 0.0],  # X1-X2 edge
        [0.5, 0.0, 0.5],  # X1-X3 edge
        [0.0, 0.5, 0.5],  # X2-X3 edge
    ])
    
    for expected_point in expected:
        matches = np.isclose(grid[["X1", "X2", "X3"]].values, expected_point, atol=1e-10).all(axis=1)
        assert matches.any(), f"Expected point {expected_point} not found in grid"


def test_mixture_grid_known_points_m3_unconstrained():
    """Test against analytically computed simplex-lattice points for m=3."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=3)
    
    # For {3,3} simplex-lattice: 10 points
    # Vertices: (1,0,0), (0,1,0), (0,0,1)
    # Edge points: (2/3,1/3,0), (1/3,2/3,0), (2/3,0,1/3), (1/3,0,2/3), (0,2/3,1/3), (0,1/3,2/3)
    # Center: (1/3,1/3,1/3)
    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2/3, 1/3, 0.0],
        [1/3, 2/3, 0.0],
        [2/3, 0.0, 1/3],
        [1/3, 0.0, 2/3],
        [0.0, 2/3, 1/3],
        [0.0, 1/3, 2/3],
        [1/3, 1/3, 1/3],
    ])
    
    assert len(grid) == 10, f"Expected 10 points for {{3,3}} simplex-lattice, got {len(grid)}"
    
    # Check each expected point exists in the grid
    for expected_point in expected:
        matches = np.isclose(grid[["X1", "X2", "X3"]].values, expected_point, atol=1e-10).all(axis=1)
        assert matches.any(), f"Expected point {expected_point} not found in grid"
    
def test_mixture_grid_known_points_m10_unconstrained():
    """Test against analytically computed simplex-lattice points for m=10."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=10)
    
    expected = np.array([[0. , 0. , 1. ],
       [0. , 0.1, 0.9],
       [0. , 0.2, 0.8],
       [0. , 0.3, 0.7],
       [0. , 0.4, 0.6],
       [0. , 0.5, 0.5],
       [0. , 0.6, 0.4],
       [0. , 0.7, 0.3],
       [0. , 0.8, 0.2],
       [0. , 0.9, 0.1],
       [0. , 1. , 0. ],
       [0.1, 0. , 0.9],
       [0.1, 0.1, 0.8],
       [0.1, 0.2, 0.7],
       [0.1, 0.3, 0.6],
       [0.1, 0.4, 0.5],
       [0.1, 0.5, 0.4],
       [0.1, 0.6, 0.3],
       [0.1, 0.7, 0.2],
       [0.1, 0.8, 0.1],
       [0.1, 0.9, 0. ],
       [0.2, 0. , 0.8],
       [0.2, 0.1, 0.7],
       [0.2, 0.2, 0.6],
       [0.2, 0.3, 0.5],
       [0.2, 0.4, 0.4],
       [0.2, 0.5, 0.3],
       [0.2, 0.6, 0.2],
       [0.2, 0.7, 0.1],
       [0.2, 0.8, 0. ],
       [0.3, 0. , 0.7],
       [0.3, 0.1, 0.6],
       [0.3, 0.2, 0.5],
       [0.3, 0.3, 0.4],
       [0.3, 0.4, 0.3],
       [0.3, 0.5, 0.2],
       [0.3, 0.6, 0.1],
       [0.3, 0.7, 0. ],
       [0.4, 0. , 0.6],
       [0.4, 0.1, 0.5],
       [0.4, 0.2, 0.4],
       [0.4, 0.3, 0.3],
       [0.4, 0.4, 0.2],
       [0.4, 0.5, 0.1],
       [0.4, 0.6, 0. ],
       [0.5, 0. , 0.5],
       [0.5, 0.1, 0.4],
       [0.5, 0.2, 0.3],
       [0.5, 0.3, 0.2],
       [0.5, 0.4, 0.1],
       [0.5, 0.5, 0. ],
       [0.6, 0. , 0.4],
       [0.6, 0.1, 0.3],
       [0.6, 0.2, 0.2],
       [0.6, 0.3, 0.1],
       [0.6, 0.4, 0. ],
       [0.7, 0. , 0.3],
       [0.7, 0.1, 0.2],
       [0.7, 0.2, 0.1],
       [0.7, 0.3, 0. ],
       [0.8, 0. , 0.2],
       [0.8, 0.1, 0.1],
       [0.8, 0.2, 0. ],
       [0.9, 0. , 0.1],
       [0.9, 0.1, 0. ],
       [1. , 0. , 0. ]])

    assert len(grid) == 66, f"Expected 66 points for {{3,10}} simplex-lattice, got {len(grid)}"
    
    # Check each expected point exists in the grid
    for expected_point in expected:
        matches = np.isclose(grid[["X1", "X2", "X3"]].values, expected_point, atol=1e-10).all(axis=1)
        assert matches.any(), f"Expected point {expected_point} not found in grid"

def test_mixture_grid_simple_constraints():
    """Test with simple lower bounds where feasible points are calculable."""
    # Each component must be at least 0.2, so max any can be is 0.6
    bounds = {"X1": (0.2, 0.6), "X2": (0.2, 0.6), "X3": (0.2, 0.6)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=2)
    
    # For m=2 with these symmetric constraints:
    # The vertices should be at the upper bound for one component,
    # lower bound + (1-UB-LB)/2 for the other two
    # E.g., (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)
    vertices = [
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ]
    
    # Verify all vertices are present
    for vertex in vertices:
        matches = np.isclose(grid[["X1", "X2", "X3"]].values, vertex, atol=1e-10).all(axis=1)
        assert matches.any(), f"Vertex {vertex} not found in grid"
    
    # All points should satisfy bounds
    assert (grid["X1"] >= 0.2 - 1e-10).all() and (grid["X1"] <= 0.6 + 1e-10).all()
    assert (grid["X2"] >= 0.2 - 1e-10).all() and (grid["X2"] <= 0.6 + 1e-10).all()
    assert (grid["X3"] >= 0.2 - 1e-10).all() and (grid["X3"] <= 0.6 + 1e-10).all()
    
    # All points should sum to 1
    assert np.allclose(grid.sum(axis=1), 1.0)


@pytest.mark.parametrize("m,expected_count", [
    (2, 6),   # C(2+3-1, 3-1) = C(4,2) = 6
    (3, 10),  # C(3+3-1, 3-1) = C(5,2) = 10
    (4, 15),  # C(4+3-1, 3-1) = C(6,2) = 15
    (5, 21),  # C(5+3-1, 3-1) = C(7,2) = 21
    (6, 28),  # C(6+3-1, 3-1) = C(8,2) = 28
])
def test_mixture_grid_point_count_unconstrained(m, expected_count):
    """Test that unconstrained designs have correct number of lattice points."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=m)
    
    assert len(grid) == expected_count, \
        f"For {{m={m}, k=3}} simplex-lattice, expected {expected_count} points, got {len(grid)}"


def test_mixture_grid_symmetry():
    """Test that symmetric bounds produce symmetric designs."""
    # All components have identical bounds - design should be symmetric
    bounds = {"X1": (0.1, 0.8), "X2": (0.1, 0.8), "X3": (0.1, 0.8)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=3)
    
    # Check that mean proportions are equal (symmetry)
    means = grid.mean()
    assert np.allclose(means["X1"], means["X2"], atol=1e-10)
    assert np.allclose(means["X2"], means["X3"], atol=1e-10)
    
    # Check that standard deviations are equal
    stds = grid.std()
    assert np.allclose(stds["X1"], stds["X2"], atol=1e-10)
    assert np.allclose(stds["X2"], stds["X3"], atol=1e-10)
    
    # Check that ranges are equal
    ranges = grid.max() - grid.min()
    assert np.allclose(ranges["X1"], ranges["X2"], atol=1e-10)
    assert np.allclose(ranges["X2"], ranges["X3"], atol=1e-10)


def test_mixture_grid_asymmetric_bounds():
    """Test grid generation with asymmetric component bounds."""
    # Different bounds for each component
    bounds = {"X1": (0.1, 0.5), "X2": (0.2, 0.7), "X3": (0.15, 0.6)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=3)
    
    # All points should satisfy individual bounds
    assert (grid["X1"] >= 0.1 - 1e-10).all() and (grid["X1"] <= 0.5 + 1e-10).all()
    assert (grid["X2"] >= 0.2 - 1e-10).all() and (grid["X2"] <= 0.7 + 1e-10).all()
    assert (grid["X3"] >= 0.15 - 1e-10).all() and (grid["X3"] <= 0.6 + 1e-10).all()
    
    # All points should sum to 1
    assert np.allclose(grid.sum(axis=1), 1.0)
    
    # Should have at least some points (feasibility check)
    assert len(grid) > 0


def test_mixture_grid_tight_constraints():
    """Test with tight constraints that significantly reduce feasible region."""
    # Very tight constraints: each component between 0.3 and 0.4
    # This gives very little feasible space
    bounds = {"X1": (0.3, 0.4), "X2": (0.3, 0.4), "X3": (0.3, 0.4)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(design, factors, ["X1", "X2", "X3"], m=5)
    
    # All points should satisfy bounds
    for col in ["X1", "X2", "X3"]:
        assert (grid[col] >= 0.3 - 1e-10).all()
        assert (grid[col] <= 0.4 + 1e-10).all()
    
    # All points should sum to 1
    assert np.allclose(grid.sum(axis=1), 1.0)
    
    # Should find at least the centroid-like points
    assert len(grid) > 0


def test_mixture_grid_column_reordering():
    """Test that output respects the column order of the design matrix."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    
    # Test different column orders
    design1 = pd.DataFrame(columns=["X1", "X2", "X3"])
    design2 = pd.DataFrame(columns=["X3", "X2", "X1"])
    design3 = pd.DataFrame(columns=["X2", "X3", "X1"])
    
    grid1 = mixture_grid(design1, factors, ["X1", "X2", "X3"], m=2)
    grid2 = mixture_grid(design2, factors, ["X1", "X2", "X3"], m=2)
    grid3 = mixture_grid(design3, factors, ["X1", "X2", "X3"], m=2)
    
    assert list(grid1.columns) == ["X1", "X2", "X3"]
    assert list(grid2.columns) == ["X3", "X2", "X1"]
    assert list(grid3.columns) == ["X2", "X3", "X1"]


def test_mixture_grid_with_constant_components():
    """Test mixture grid with some components held constant."""
    bounds = {
        "X1": (0.0, 1.0), 
        "X2": (0.0, 1.0), 
        "X3": (0.0, 1.0),
        "X4": (0.0, 1.0)
    }
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3", "X4"])
    
    expected = np.array([
        [0.8, 0.0, 0.0, 0.2],
        [0.0, 0.8, 0.0, 0.2],
        [0.0, 0.0, 0.8, 0.2],
        [0.4, 0.4, 0.0, 0.2],
        [0.4, 0.0, 0.4, 0.2],
        [0.0, 0.4, 0.4, 0.2],
    ])
    # Fix X4, vary X1, X2, X3
    grid = mixture_grid(
        design, 
        factors, 
        ["X1", "X2", "X3"], 
        m=2,
        constant_levels={"X4": 0.2}
    )
    
    # X4 should be constant
    assert (grid["X4"] == 0.2).all()
    
    # Free components should sum to 0.8 (1 - 0.2)
    free_sum = grid["X1"] + grid["X2"] + grid["X3"]
    assert np.allclose(free_sum, 0.8)
    
    # All components should sum to 1
    assert np.allclose(grid.sum(axis=1), 1.0)
    
    # Check each expected point exists in the grid
    for expected_point in expected:
        matches = np.isclose(grid[["X1", "X2", "X3", "X4"]].values, expected_point, atol=1e-10).all(axis=1)
        assert matches.any(), f"Expected point {expected_point} not found in grid"

def test_mixture_grid_with_constraints():
    
    """Test mixture grid with bounds."""
    bounds = {
        "X1": (0.2, 0.5), 
        "X2": (0.2, 0.8), 
        "X3": (0.0, 0.8),
    }
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])
    
    grid = mixture_grid(
        design, 
        factors, 
        ["X1", "X2", "X3"], 
        m=10
    )
    
    expected = np.array([[0.2 , 0.2 , 0.6 ],
       [0.2 , 0.26, 0.54],
       [0.2 , 0.32, 0.48],
       [0.2 , 0.38, 0.42],
       [0.2 , 0.44, 0.36],
       [0.2 , 0.5 , 0.3 ],
       [0.2 , 0.56, 0.24],
       [0.2 , 0.62, 0.18],
       [0.2 , 0.68, 0.12],
       [0.2 , 0.74, 0.06],
       [0.2 , 0.8 , 0.  ],
       [0.26, 0.2 , 0.54],
       [0.26, 0.26, 0.48],
       [0.26, 0.32, 0.42],
       [0.26, 0.38, 0.36],
       [0.26, 0.44, 0.3 ],
       [0.26, 0.5 , 0.24],
       [0.26, 0.56, 0.18],
       [0.26, 0.62, 0.12],
       [0.26, 0.68, 0.06],
       [0.26, 0.74, 0.  ],
       [0.32, 0.2 , 0.48],
       [0.32, 0.26, 0.42],
       [0.32, 0.32, 0.36],
       [0.32, 0.38, 0.3 ],
       [0.32, 0.44, 0.24],
       [0.32, 0.5 , 0.18],
       [0.32, 0.56, 0.12],
       [0.32, 0.62, 0.06],
       [0.32, 0.68, 0.  ],
       [0.38, 0.2 , 0.42],
       [0.38, 0.26, 0.36],
       [0.38, 0.32, 0.3 ],
       [0.38, 0.38, 0.24],
       [0.38, 0.44, 0.18],
       [0.38, 0.5 , 0.12],
       [0.38, 0.56, 0.06],
       [0.38, 0.62, 0.  ],
       [0.44, 0.2 , 0.36],
       [0.44, 0.26, 0.3 ],
       [0.44, 0.32, 0.24],
       [0.44, 0.38, 0.18],
       [0.44, 0.44, 0.12],
       [0.44, 0.5 , 0.06],
       [0.44, 0.56, 0.  ],
       [0.5 , 0.2 , 0.3 ],
       [0.5 , 0.26, 0.24],
       [0.5 , 0.32, 0.18],
       [0.5 , 0.38, 0.12],
       [0.5 , 0.44, 0.06],
       [0.5 , 0.5 , 0.  ]])
    
    # All components should sum to 1
    assert np.allclose(grid.sum(axis=1), 1.0)
    
    # Check each expected point exists in the grid
    for expected_point in expected:
        matches = np.isclose(grid[["X1", "X2", "X3"]].values, expected_point, atol=1e-10).all(axis=1)
        assert matches.any(), f"Expected point {expected_point} not found in grid"
    


# --- Error handling tests ---


def test_mixture_grid_error_free_components_not_three():
    """Test error when number of free components is not exactly 3."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])

    with pytest.raises(ValueError, match="Expected 3 free mixture components"):
        mixture_grid(
            design_matrix=design,
            factors=factors,
            mix_factors=["X1", "X2", "X3"],
            m=3,
            constant_levels={"X1": 0.2},
        )


def test_mixture_grid_error_fixed_components_sum_greater_than_one():
    
    """Test error when fixed components sum to more than 1."""
    bounds = {"X1": (0.0, 1.0), "X2": (0.0, 1.0), "X3": (0.0, 1.0), "X4": (0.0, 1.0), "X5": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3", "X4", "X5"])

    with pytest.raises(ValueError, match="Sum of fixed components exceeds 1"):
        mixture_grid(
            design_matrix=design,
            factors=factors,
            mix_factors=["X1", "X2", "X3", "X4", "X5"],
            m=2,
            constant_levels={"X4": 0.6, "X5": 0.5},
        )


def test_mixture_grid_error_lb_sum_greater_than_total():
    """Test error when sum of lower bounds exceeds the remaining total."""
    bounds = {"X1": (0.6, 1.0), "X2": (0.3, 1.0), "X3": (0.2, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])

    with pytest.raises(ValueError, match=r"Infeasible: sum\(lower_bounds of free comps\) > remaining total"):
        mixture_grid(design, factors=factors, mix_factors=["X1", "X2", "X3"], m=3)


def test_mixture_grid_error_ub_sum_less_than_total():
    """Test error when sum of upper bounds is less than the remaining total."""
    bounds = {"X1": (0.0, 0.2), "X2": (0.0, 0.2), "X3": (0.0, 0.2)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])

    with pytest.raises(ValueError, match=r"Infeasible: sum\(upper_bounds of free comps\) < remaining totale"):
        mixture_grid(design, factors=factors, mix_factors=["X1", "X2", "X3"], m=2)


def test_mixture_grid_error_lower_bound_greater_than_upper():
    bounds = {"X1": (0.0, 1.0), "X2": (0.5, 0.3), "X3": (0.0, 1.0)}
    factors = _make_factors(bounds)
    design = pd.DataFrame(columns=["X1", "X2", "X3"])

    with pytest.raises(ValueError, match="Infeasible: some lower_bound > upper_bound among free comps"):
        mixture_grid(design, factors=factors, mix_factors=["X1", "X2", "X3"], m=2)
