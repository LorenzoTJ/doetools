"""
Multi-objective optimization using Pareto front analysis.

This module provides functionality for computing and visualizing Pareto-optimal
solutions in design of experiments (DoE). It identifies non-dominated points
that represent optimal trade-offs between competing response objectives.

The Pareto front consists of all solutions where no objective can be improved
without degrading at least one other objective. For minimization problems,
point A dominates point B if A ≤ B in all objectives and A < B in at least one.

Key Features
------------
- Grid-based candidate generation across design space
- Support for process, mixture, and mixed design types
- Constraint filtering for feasible region definition
- Dominance sorting algorithm for Pareto front identification
- Interactive 2D and 3D visualization with Plotly

Typical Workflow
----------------
1. Fit response surface models to experimental data
2. Define optimization goals (maximize/minimize) for each response
3. Generate candidate points grid with specified resolution
4. Apply optional constraint filters
5. Identify non-dominated solutions
6. Visualize trade-offs between objectives
7. Select optimal conditions for further experimentation

See Also
--------
ParetoMixin : Main class providing Pareto optimization methods.
"""

# Import Standard Libraries
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
from typing import Callable


# Define Pareto Mixin Class
class ParetoMixin:
    """Mixin class for multi-objective optimization via Pareto analysis.

    Provides methods to compute Pareto-optimal solutions by generating candidate
    points across the design space, predicting responses, and identifying
    non-dominated points using dominance sorting.

    Attributes
    ----------
    _non_dominated_points : pd.DataFrame or None
        DataFrame containing factor values and predicted responses for points
        on the Pareto front. None until compute_pareto_front() is called.

    Methods
    -------
    compute_pareto_front(factors, m=None, filters=None)
        Generate candidate grid and identify Pareto-optimal solutions.
    non_dominated_points
        Property to access the computed Pareto front DataFrame.
    plot_pareto_front(x=None, y=None, z=None)
        Create interactive 2D or 3D visualization of Pareto solutions.

    Notes
    -----
    The dominance criterion for minimization is:
        A dominates B ⟺ ∀j: Aⱼ ≤ Bⱼ ∧ ∃j: Aⱼ < Bⱼ

    For maximization objectives, responses are negated before applying
    the dominance criterion.

    """

    def __init__(self):
        """
        Initialize ParetoMixin with empty non-dominated points storage.

        Sets _non_dominated_points to None. This will be populated when
        compute_pareto_front() is called.
        """
        self._non_dominated_points: pd.DataFrame = None

    def compute_pareto_front(
        self,
        factors: dict,
        m: int = None,
        filters: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
    ) -> None:
        """
        Compute the Pareto front for multi-objective optimization.

        Generates a grid of candidate points across the design space, predicts
        responses using fitted models, and identifies non-dominated solutions
        that represent optimal trade-offs between competing objectives.

        Parameters
        ----------
        factors : dict
            Dictionary mapping factor names to factor objects. Each factor must
            have n_levels specified to control grid resolution.

            For continuous factors: n_levels determines grid fineness
            For categorical factors: uses all defined levels
            For mixture factors: uses simplex lattice with degree m

        m : int, optional
            Grid resolution for mixture factors (simplex lattice degree).
            Higher values create finer grids but increase computation.
            If None and mixture factors present, defaults to 5.

        filters : list[Callable[[pd.DataFrame], pd.Series]], optional
            List of constraint functions to define feasible region.
            Each function takes a DataFrame of candidate points and returns
            a boolean Series indicating which points satisfy the constraint.

        Returns
        -------
        None
            Results are stored in _non_dominated_points attribute, accessible
            via the non_dominated_points property.

        Raises
        ------
        ValueError
            If model matrix, MLR models, responses, or response conditions
            have not been set up prior to calling this method.
        ValueError
            If number or names of factors don't match the design.
        ValueError
            If mixture factor bounds are infeasible (sum constraints violated).

        Notes
        -----
        Algorithm steps:
            1. Validate prerequisites (model fitted, responses defined, etc.)
            2. Generate candidate point grid:
               - Process factors: full factorial grid of specified resolution
               - Mixture factors: simplex lattice points satisfying bounds
               - Mixed designs: cross-product of process and mixture grids
            3. Apply constraint filters to identify feasible region
            4. Predict responses for all feasible candidates
            5. Identify non-dominated points using dominance sorting:
               - Transform maximization to minimization (negate objectives)
               - Point A dominates B if A ≤ B in all objectives and A < B in at least one
               - Keep only points not dominated by any other point

        For mixture factors, the algorithm maps from constrained simplex space
        to unconstrained z-space to generate lattice points, then transforms
        back and filters by upper bounds.

        See Also
        --------
        non_dominated_points : Property to access computed Pareto front.
        plot_pareto_front : Visualize Pareto-optimal solutions.
        set_response_conditions : Define optimization goals before computing front.
        """
        # ----------------------------------------- INPUT VALIDATION -------------------------------------------------- #
        if self._model_matrix is None:
            raise ValueError(
                "Model matrix has not been built yet; please call `set_model_terms` first."
            )
        if self._mlr_wrapper is None:
            raise ValueError(
                "MLR models have not been computed yet; please call `mlr_model_computation` first."
            )
        if self._responses is None:
            raise ValueError(
                "Responses haven't been loaded yet, please call 'load_responses' or 'import_responses' first."
            )
        if not self._response_conditions:
            raise ValueError(
                "Response conditions haven't been set yet, please call 'set_response_conditions' first."
            )

        # Extract factors information
        factor_names = list(factors.keys())
        cont_factors = [f for f in factor_names if factors[f].type == "cont"]
        cat_factors = [f for f in factor_names if factors[f].type == "cat"]
        pro_factors = cont_factors + cat_factors
        mix_factors = [f for f in factor_names if factors[f].type == "mix"]

        # Validate factors match design
        if len(factors.keys()) != len(self._factors.keys()):
            raise ValueError(
                "Number of factors provided does not match the number of factors in the design."
            )
        for var in factors.keys():
            if var not in self._factors.keys():
                raise ValueError(f"Variable '{var}' not found in design matrix.")

        # Set default m for mixture factors if not provided
        if len(mix_factors) > 0 and m is None:
            m = 5

        # ------------------------------------- BUILD THE MATRIX OF CP --------------------------------------- #

        # Build the process design matrix of candidate points
        if len(pro_factors) > 0:
            lvls: dict[str, np.ndarray] = {}
            for f in pro_factors:
                factor = factors[f]

                if factor.type == "cont":
                    # Use n_levels from the factor object
                    lvls[f] = np.linspace(-1, 1, num=factor.n_levels)  # coded levels
                else:  # categorical
                    lvls[f] = factor.coded_levels

            grid = list(itertools.product(*(lvls[f] for f in pro_factors)))
            pro_cp_coded = pd.DataFrame(grid, columns=pro_factors)

        # Build the mixture design matrix of candidate points
        if len(mix_factors) > 0:
            lb = [factors[f].lower_bound for f in mix_factors]
            ub = [factors[f].upper_bound for f in mix_factors]
            k = len(mix_factors)

            S = 1.0 - sum(lb)
            if S < -1e-12:
                raise ValueError("Sum of lower_bounds must be < 1.")
            S = max(S, 0.0)

            # feasibility for upper bounds
            if sum(ub) < 1 - 1e-12:
                raise ValueError("Infeasible: sum(upper_bounds) must be >= 1.")
            if any(low > up for low, up in zip(lb, ub)):
                raise ValueError("Infeasible: some lower_bound > upper_bound.")

            # cap in z-space
            if S > 0:
                z_cap = [(up - low) / S for low, up in zip(lb, ub)]
            else:
                # S == 0 means x is fixed at lb
                z_cap = [0.0] * k

            rows = []
            for r in itertools.product(range(m + 1), repeat=k):
                if sum(r) != m:
                    continue
                z = [ri / m for ri in r]

                # early z-space upper-bound test
                if any(zi > cap + 1e-12 for zi, cap in zip(z, z_cap)):
                    continue

                # map to x
                x = [low + S * zi for low, zi in zip(lb, z)]

                # (optional) numerical safety
                if any(xi > ui + 1e-12 for xi, ui in zip(x, ub)):
                    continue

                rows.append(x)

            mix_cp_coded = pd.DataFrame(rows, columns=mix_factors)

        # Merge process and mixture candidate points
        if len(pro_factors) > 0 and len(mix_factors) > 0:
            cp_coded = pro_cp_coded.merge(mix_cp_coded, how="cross")

        elif len(pro_factors) > 0:
            cp_coded = pro_cp_coded.copy()

        else:
            cp_coded = mix_cp_coded.copy()

        # Matrix of candidate points in real values
        cp = self._decode_matrix(cp_coded)

        # Round continuous factors to specified decimals
        for f in cont_factors:
            cp[f] = cp[f].round(self._factors[f].decimals)

        # Apply filters (constraints)
        if filters:
            mask = pd.Series(True, index=cp.index)
            for f in filters:
                mask &= f(cp)
            cp = cp[mask].reset_index(drop=True)

        # Prepare and save the candidate points
        cp = cp[self._design_matrix.columns].drop_duplicates().reset_index(drop=True)
        cp_coded = self._code_matrix(cp)

        # ----------------------------------------- COMPUTE PARETO FRONT -------------------------------------------------- #
        # Define responses to maximize/minimize
        responses = self._response_list.copy()
        maximize = [self._response_conditions[resp]["maximize"] for resp in responses]

        # Add Model Terms
        model_matrix = self._build_model_matrix(cp_coded, self._model_spec)
        # Predict Responses for Each Point
        pred_responses = self._mlr_predict(model_matrix, responses)
        # Create the objective matrix
        obj_matrix = pred_responses[responses].copy()
        # If maximize = False -> change the sign of the response
        for i, maximize_flag in enumerate(maximize):
            if maximize_flag:
                obj_matrix.iloc[:, i] = -obj_matrix.iloc[:, i]
        obj_array = obj_matrix.to_numpy()

        # Check if a point is dominated
        # A dominates B if ∀j Aj =< Bj and ∃j : Aj < Bj
        # Where with j we are looping across the responses
        # If ∃ A that dominates B => B is a dominated point and it doesn't belong to non dominated-points
        # The function is_dominated loops through all the "other points" and check if they dominate B ("point")
        # If at least one dominates B, we drop B
        def is_dominated(point, others):
            return np.any(
                np.all(others <= point, axis=1) & np.any(others < point, axis=1)
            )

        pareto_indices = []
        for i, point in enumerate(obj_array):
            others = np.delete(obj_array, i, axis=0)
            if not is_dominated(point, others):
                pareto_indices.append(i)
        # Build the Pareto DataFrame
        pareto_df = pd.concat([cp, pred_responses], axis=1).iloc[pareto_indices].copy()
        # pareto_df.reset_index(drop=True, inplace=True)
        self._non_dominated_points = pareto_df

    @property
    def non_dominated_points(self):
        """
        Access the computed Pareto-optimal solutions.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing factor values and predicted responses for
            non-dominated points. Columns include all factors plus all response
            variables. Returns None if compute_pareto_front() has not been called.

        Notes
        -----
        The DataFrame preserves the original index from the candidate set,
        which can be useful for tracing back to specific experimental conditions.
        """
        return self._non_dominated_points

    def plot_pareto_front(self, x: str = None, y: str = None, z: str = None):
        """Visualize Pareto-optimal solutions in two or three dimensions.

        The two-dimensional plot overlays configured lower and upper response
        limits. Hover information preserves each candidate's original index.

        Args:
            x (str, optional): Response plotted on the x-axis. When both ``x``
                and ``y`` are omitted, the first two responses are used.
            y (str, optional): Response plotted on the y-axis. It must be
                provided together with ``x`` and be distinct from it.
            z (str, optional): Distinct response plotted on the z-axis. When
                omitted, a two-dimensional plot is returned.

        Returns:
            plotly.graph_objects.Figure: Interactive Pareto-front scatter plot.

        Raises:
            ValueError: If only one of ``x`` and ``y`` is specified, axes are
                repeated, or an axis is absent from the Pareto-front data.

        Notes:
            Call :meth:`compute_pareto_front` before plotting. At least two
            response columns are required when ``x`` and ``y`` are omitted.
        """
        # Prepare the matrix of non_dominated_points
        df = self._non_dominated_points.copy()
        df["_idx"] = df.index

        # Validate x and y
        if (x is None) ^ (y is None):
            raise ValueError("Please specify both x and y variables")
        if x is None and y is None:
            x, y = self._responses_list[0], self._responses_list[1]
        if x == y:
            raise ValueError("x and y variables cannot be the same")
        for axis in (x, y):
            if axis not in df.columns:
                raise ValueError(f"{axis!r} is not a column in the Pareto front data")
        # Validate Z
        if z is not None:
            if z in (x, y):
                raise ValueError("z must be distinct from x and y")
            if z not in df.columns:
                raise ValueError(f"{z!r} is not a column in the Pareto front data")

        # Create the scatter plot
        hover_cfg = {"_idx": True}
        labels = {"_idx": "Exp."}
        if z is None:
            fig = px.scatter(
                df,
                x=x,
                y=y,
                hover_data=hover_cfg,
                labels=labels,
                title="Pareto Optimal Solutions",
                height=800,
                width=800,
            )
            # Create the lines for the feasible region
            resp_cond = self._response_conditions.copy()
            if len(resp_cond) != 0:
                if resp_cond[x]["lower_limit"] is not None:
                    vl_limit = resp_cond[x]["lower_limit"]
                    fig.add_vline(x=vl_limit, line_dash="solid", line_color="red")
                if resp_cond[x]["upper_limit"] is not None:
                    vu_limit = resp_cond[x]["upper_limit"]
                    fig.add_vline(x=vu_limit, line_dash="solid", line_color="red")
                if resp_cond[y]["lower_limit"] is not None:
                    hl_limit = resp_cond[y]["lower_limit"]
                    fig.add_hline(y=hl_limit, line_dash="solid", line_color="red")
                if resp_cond[y]["upper_limit"] is not None:
                    hu_limit = resp_cond[y]["upper_limit"]
                    fig.add_hline(y=hu_limit, line_dash="solid", line_color="red")
        else:
            fig = px.scatter_3d(
                df,
                x=x,
                y=y,
                z=z,
                hover_data=hover_cfg,
                labels=labels,
                title="Pareto Optimal Solutions (3D)",
                height=800,
                width=800,
            )

        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        # Update the layout
        fig.update_layout(
            title=dict(x=0.5, xanchor="center"),
            title_font_size=26,
            xaxis_title=dict(text=x, font=dict(size=20)),
            yaxis_title=dict(text=y, font=dict(size=20)),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        if z is not None:
            fig.update_layout(
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z,
                    xaxis=dict(title_font_size=18),
                    yaxis=dict(title_font_size=18),
                    zaxis=dict(title_font_size=18),
                )
            )

        fig.update_traces(marker=dict(size=10))

        return fig
