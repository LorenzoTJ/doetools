import numpy as np
import pandas as pd
import itertools
from typing import Optional, List, Dict

def lhs_grid(
    vars: List[str],
    n_of_exp: int,
    random_state: Optional[int] = None,
    ) -> pd.DataFrame:
    """
    Generate a Latin Hypercube Sampling (LHS) design in coded space.
    
    Creates a space-filling design where the experimental space is divided into
    equally probable intervals for each factor, ensuring one sample per interval.
    The design is generated in coded space [-1, 1] for all factors.
    
    Parameters
    ----------
    vars : List[str]
        List of factor names for the design columns.
    n_of_exp : int
        Number of experimental runs (samples) to generate.
    random_state : Optional[int], optional
        Random seed for reproducibility. If None, uses random initialization.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_of_exp, len(vars)) containing the LHS design
        with values in coded space [-1, 1].
    
    Notes
    -----
    The Latin Hypercube Sampling method divides each factor's range into n_of_exp
    equally probable intervals. Within each interval, a random point is sampled,
    and the intervals are randomly permuted across factors to ensure good space-
    filling properties.
    
    Examples
    --------
    >>> lhs = lhs_grid(["A", "B", "C"], n_of_exp=10, random_state=42)
    >>> lhs.shape
    (10, 3)
    >>> (lhs.min().min() >= -1) and (lhs.max().max() <= 1)
    True
    """

    rng = np.random.default_rng(random_state)

    k = len(vars)
    lhs = np.zeros((n_of_exp, k))

    # interval edges: [0, 1/n, 2/n, ..., 1]
    edges = np.linspace(0, 1, n_of_exp + 1)

    for j in range(k):
        # random permutation of interval indices
        perm = rng.permutation(n_of_exp)

        # random point within each interval: U(edges[i], edges[i+1])
        low  = edges[perm]
        high = edges[perm + 1]
        lhs[:, j] = rng.uniform(low, high)
        
    lhs = (lhs-0.5)*2  # Scale to [-1, 1]
           
    return pd.DataFrame(lhs, columns=vars)

def rectangular_grid(design_matrix : pd.DataFrame,
                     x: str,   
                     y: str,
                     constants: Dict[str, float],
                     x_min : float = -1.0,
                     x_max : float = 1.0,
                     y_min : float = -1.0,
                     y_max : float = 1.0,
                     resolution: int = 100,
                     ) -> pd.DataFrame:
    """
    Generate a rectangular grid for contour and surface plotting.
    
    Creates a regular grid over a 2D region defined by two factors, while holding
    all other factors constant. This grid is used for visualizing response surfaces,
    contour plots, and exploring the response behavior over a rectangular region
    in the factor space.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        Reference design matrix whose columns define the full factor space.
    x : str
        Name of the factor to vary along the x-axis.
    y : str
        Name of the factor to vary along the y-axis.
    constants : Dict[str, float]
        Dictionary mapping factor names to constant values for factors not
        being varied (i.e., all factors except x and y).
    x_min : float, optional
        Minimum value for the x-axis factor in coded space. Default is -1.0.
    x_max : float, optional
        Maximum value for the x-axis factor in coded space. Default is 1.0.
    y_min : float, optional
        Minimum value for the y-axis factor in coded space. Default is -1.0.
    y_max : float, optional
        Maximum value for the y-axis factor in coded space. Default is 1.0.
    resolution : int, optional
        Number of points along each axis, creating a resolution × resolution grid.
        Default is 100.
    
    Returns
    -------
    pd.DataFrame
        Grid DataFrame with shape (resolution², n_factors) containing all
        combinations of x and y values with constant values for other factors.
    
    Notes
    -----
    The grid is constructed using np.meshgrid and flattened to create a DataFrame
    where each row represents one point in the 2D grid. All factors not specified
    in constants or as x/y are initialized to zero.
    
    Examples
    --------
    >>> design = pd.DataFrame({"A": [0], "B": [0], "C": [0]})
    >>> grid = rectangular_grid(
    ...     design, x="A", y="B", constants={"C": 0.5},
    ...     x_min=-1, x_max=1, y_min=-1, y_max=1, resolution=50
    ... )
    >>> grid.shape
    (2500, 3)
    >>> grid["C"].unique()
    array([0.5])
    """

    cols = list(design_matrix.columns)
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X1, X2 = np.meshgrid(x_vals, y_vals)

    # Create a grid DataFrame with all zeros and the shape of the design matrix
    grid = pd.DataFrame(
        np.zeros((resolution * resolution, len(cols))),
        columns=cols,
    )
    grid[x] = X1.ravel()
    grid[y] = X2.ravel()
    for name, val in constants.items():
        grid[name] = val

    return grid
    
def mixture_grid(design_matrix : pd.DataFrame,
                 factors: Dict,                
                 mix_factors: List[str],        
                 m: int,
                 constant_levels: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Generate a constrained mixture design grid on a ternary simplex.
    
    Creates a {m, k} simplex-lattice design for exactly three free mixture components,
    respecting component bounds and constraint conditions. This function is specifically
    designed for ternary mixture systems and generates all feasible lattice points
    that satisfy the simplex constraint (sum = 1) and individual component bounds.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        Reference design matrix whose columns define the full factor space ordering.
    factors : Dict
        Dictionary mapping factor names to factor objects with attributes:
        - type : str ("mix" for mixture components)
        - lower_bound : float
        - upper_bound : float
    mix_factors : List[str]
        List of mixture component names in the design.
    m : int
        Lattice resolution parameter. Generates points with proportions that are
        multiples of 1/m. Higher values produce finer grids.
    constant_levels : Optional[Dict[str, float]], optional
        Dictionary of mixture components to hold constant at specified values.
        Remaining components must total exactly 3 free components. Default is None.
    
    Returns
    -------
    pd.DataFrame
        Mixture design grid with all feasible lattice points. Columns match the
        order of the input design_matrix. Each row satisfies:
        - Sum of mixture components = 1
        - All component bounds are satisfied
        - Lattice point constraint: proportions are multiples of 1/m
    
    Raises
    ------
    ValueError
        If the number of free mixture components is not exactly 3.
    ValueError
        If the sum of fixed components exceeds 1.
    ValueError
        If the sum of lower bounds for free components exceeds the remaining total.
    ValueError
        If the sum of upper bounds for free components is less than the remaining total.
    ValueError
        If any component has lower_bound > upper_bound.
    
    Notes
    -----
    The function uses a transformation to z-space where z_i = (x_i - L_i) / S with
    S = T - sum(L_i), where T is the remaining total after accounting for fixed
    components and L_i are lower bounds. This allows efficient enumeration and
    filtering of feasible lattice points.
    
    The algorithm generates all integer combinations (r_1, r_2, r_3) where
    sum(r_i) = m and 0 <= r_i <= m, then transforms them to mixture proportions
    and filters out infeasible points based on component bounds.
    
    Examples
    --------
    >>> # Define mixture factors with bounds
    >>> factors = {
    ...     "X1": Factor(type="mix", lower_bound=0.1, upper_bound=0.8),
    ...     "X2": Factor(type="mix", lower_bound=0.1, upper_bound=0.8),
    ...     "X3": Factor(type="mix", lower_bound=0.1, upper_bound=0.8)
    ... }
    >>> design = pd.DataFrame(columns=["X1", "X2", "X3"])
    >>> grid = mixture_grid(
    ...     design,
    ...     factors=factors,
    ...     mix_factors=["X1", "X2", "X3"],
    ...     m=5
    ... )
    >>> # Verify simplex constraint
    >>> np.allclose(grid.sum(axis=1), 1.0)
    True
    """
    
    constant_levels = constant_levels or {}
    mix_constants = {k : v for k, v in constant_levels.items() if factors[k].type == "mix"}
    free_factors = [f for f in mix_factors if f not in constant_levels]
    k = len(free_factors)

    if k != 3:
        raise ValueError(f"Expected 3 free mixture components, got {k}. Free: {free_factors}")

    # Remaining total for free components
    fixed_sum = sum(mix_constants.values())
    T = 1.0 - fixed_sum
    if T < -1e-12:
        raise ValueError("Sum of fixed components exceeds 1.")
    T = max(T, 0.0)

    # Bounds for free comoponents
    lb = [factors[f].lower_bound for f in free_factors]
    ub = [factors[f].upper_bound for f in free_factors]

    # Feasibility checks for free components
    if sum(lb) > T + 1e-12:
        raise ValueError("Infeasible: sum(lower_bounds of free comps) > remaining total.")
    if sum(ub) < T - 1e-12:
        raise ValueError("Infeasible: sum(upper_bounds of free comps) < remaining totale.")
    if any(low > up for low, up in zip(lb, ub)):
        raise ValueError("Infeasible: some lower_bound > upper_bound among free comps.")

    # Slack for free allocation
    S = T - sum(lb)
    S = max(S, 0.0)

    # Calculate z-space upper bounds to filter points
    if S > 0:
        z_cap = [(up - low) / S for low, up in zip(lb, ub)]
    else:
        z_cap = [0.0] * k 

    rows = []
    for r in itertools.product(range(m + 1), repeat=k):
        if sum(r) != m:
            continue
        z = [ri / m for ri in r]

        # z-space upper bound check
        if any(zi > cap + 1e-12 for zi, cap in zip(z, z_cap)):
            continue

        # Map to x on the simplex of total T
        x_free = [low + S * zi for low, zi in zip(lb, z)]

        # Numerical safety
        if any(xi > ui + 1e-12 for xi, ui in zip(x_free, ub)):
            continue

        # Build full vector in original column order
        full = []
        free_iter = iter(x_free)
        for name in mix_factors:
            full.append(constant_levels.get(name, next(free_iter)))

        rows.append(full)
    
    df = pd.DataFrame(rows, columns=mix_factors)

    for name, val in constant_levels.items():
        df[name] = val
    
    df = df.reindex(columns=design_matrix.columns)

    return df