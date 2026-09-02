"""D-optimal experimental designs."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...graphs import GraphsMixin
from ...graphs.design_plot_builder import DesignPlotOptions, build_design_plot
from ...utils import Design
from ...utils.model_spec import ModelTerms, compile_model_spec
from .candidate_set import DOptimalCandidateSetMixin
from .optimizer import (
    build_d_optimal_solutions_figure,
    optimize_d_optimal,
)


class DOptDesign(DOptimalCandidateSetMixin, Design, GraphsMixin):
    """Generate and select a D-optimal experimental design.

    Candidate points are generated during initialization. Process candidates
    come from ``process_strategy``; mixture candidates are explicitly selected
    with ``mixture_include``, ``mixture_grid``, or both. For mixed designs, the
    process and mixture candidate sets are combined by Cartesian product.

    Parameters:
        factors (dict[str, object]): Mapping of names to ``ContinuousFactor``,
            ``CategoricalFactor``, or ``MixtureFactor`` objects.
        process_strategy (str, optional): Candidate strategy for process factors.
            Supported values are ``"grid"``, ``"lhs"``, ``"ccc"``, ``"ccf"``,
            ``"cci"``, and ``"bb"``. Required when process factors are present
            and omitted for mixture-only designs. Defaults to ``None``.
        mixture_include (str | Collection[str] | None, optional): Geometric
            mixture candidates. Pass ``"all"`` or select from ``"vertices"``,
            ``"edge_midpoints"``, ``"face_centroids"``, and
            ``"global_centroid"``. Defaults to an empty collection.
        mixture_grid (Mapping[str, int] | None, optional): Optional bounded
            simplex-lattice configuration. The mapping must contain ``"degree"``
            and may contain ``"max_candidates"``. Defaults to ``None``.
        lhs_n_samples (int | None, optional): Number of Latin-hypercube points.
            Required only when ``process_strategy="lhs"``. Defaults to ``None``.
        filters (list[Callable] | None, optional): Functions evaluated on the
            actual-valued candidate frame. Each function must return one Boolean
            value per candidate row. Defaults to ``None``.

    Raises:
        ValueError: If ``factors`` is empty or candidate-generation options are
            missing, incompatible, or produce an empty candidate set.
        TypeError: If a candidate filter does not return a pandas Series with one
            value per candidate row.

    Notes:
        Mixture geometry is opt-in. A design containing mixture factors must
        request ``mixture_include``, ``mixture_grid``, or both.
    """

    def __init__(
        self,
        factors: dict[str, object],
        process_strategy: Literal["grid", "ccf", "cci", "ccc", "bb", "lhs"]
        | None = None,
        mixture_include: Literal[
            "all",
            "vertices",
            "edge_midpoints",
            "face_centroids",
            "global_centroid",
        ]
        | Collection[str]
        | None = (),
        mixture_grid: Mapping[str, int] | None = None,
        lhs_n_samples: int | None = None,
        filters: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
    ) -> None:
        super().__init__()
        if not factors:
            raise ValueError("factors must contain at least one factor")
        self._factors = factors
        self.set_domain_filters(filters)
        self._coded_cp, self._cp = self.build_cp_matrix(
            factors=factors,
            process_strategy=process_strategy,
            mixture_include=mixture_include,
            mixture_grid=mixture_grid,
            lhs_n_samples=lhs_n_samples,
            filters=filters,
        )
        self._cp_model_matrix: pd.DataFrame | None = None
        self._best_idx: dict[int, list[int]] = {}
        self._log_det: pd.DataFrame | None = None
        self._design_type = "D-Optimal"

    def build_cp_matrix(
        self,
        factors: dict[str, object],
        process_strategy: str | None = None,
        mixture_include: str | Collection[str] | None = (),
        mixture_grid: Mapping[str, int] | None = None,
        lhs_n_samples: int | None = None,
        filters: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build and return coded and actual D-optimal candidate matrices."""
        return self._generate_d_optimal_candidates(
            factors=factors,
            process_strategy=process_strategy,
            mixture_include=mixture_include,
            mixture_grid=mixture_grid,
            lhs_n_samples=lhs_n_samples,
            filters=filters,
        )

    def set_model_terms(self, terms: ModelTerms = ModelTerms) -> None:
        """Define the model optimized by the D-optimal search.

        Args:
            terms (ModelTerms): Process and mixture terms to include.

        Raises:
            ValueError: If the model contains more coefficients than candidate
                points.
        """
        process = [
            name
            for name, factor in self._factors.items()
            if factor.type in {"cont", "cat"}
        ]
        mixture = [
            name for name, factor in self._factors.items() if factor.type == "mix"
        ]
        self._model_spec = compile_model_spec(terms, process, mixture)
        if self._model_spec.model_terms > len(self._cp):
            raise ValueError(
                f"The number of model terms ({self._model_spec.model_terms}) "
                f"exceeds the number of candidate points ({len(self._cp)})."
            )
        self._cp_model_matrix = self._build_model_matrix(
            self._coded_cp, self._model_spec
        )

    def compute_d_optimal(
        self,
        n_min: int,
        n_max: int,
        step: int = 1,
        trials: int = 50,
        max_no_improve: int = 5,
        graph: bool = True,
        random_state: int | np.random.Generator | None = None,
    ) -> go.Figure:
        """Compute D-optimal subsets for the requested total run counts.

        Args:
            n_min (int): Smallest total run count to optimize.
            n_max (int): Largest total run count to optimize.
            step (int, optional): Increment between optimized run counts.
                Defaults to 1.
            trials (int, optional): Random exchange-search starts per run count.
                Defaults to 50.
            max_no_improve (int, optional): Maximum rejected improving-exchange
                proposals before a trial stops. Defaults to 5.
            graph (bool, optional): Retained for API compatibility. The returned
                figure is never displayed automatically. Defaults to ``True``.
            random_state (int | numpy.random.Generator | None, optional): Seed or
                generator used by the randomized starts. Defaults to ``None``.

        Returns:
            plotly.graph_objects.Figure: D-optimality curve for the computed run
            counts.

        Raises:
            ValueError: If model terms have not been set or the requested run
                range is invalid for the candidate set and model.
            RuntimeError: If a full-rank subset cannot be found.
        """
        if self._cp_model_matrix is None:
            raise ValueError("No model matrix defined; call 'set_model_terms' first")
        result = optimize_d_optimal(
            self._cp_model_matrix,
            n_min=n_min,
            n_max=n_max,
            step=step,
            trials=trials,
            max_no_improve=max_no_improve,
            random_state=random_state,
        )
        self._best_idx = result.best_indices
        self._log_det = result.log_det
        return build_d_optimal_solutions_figure(self._log_det)

    def plot_candidate_set(
        self,
        ax1: str,
        ax2: str,
        ax3: str | None = None,
        ax4: str | None = None,
        coded: bool = False,
        *,
        show_title: bool = True,
        show_summary: bool = True,
        show_legend: bool = True,
        show_grid: bool = True,
        marker_color: str = "#495057",
        show_hover: bool = True,
        show_run_labels: bool = False,
        show_replicate_count: bool = True,
        aggregate_projected_points: bool = True,
        axis_label_mode: str = "symbol_unit",
        height: int | None = None,
        domain: Literal["full", "allowed"] = "full",
    ) -> go.Figure:
        """Visualize the candidate set before selecting a design.

        Args:
            ax1 (str): First factor shown in the plot.
            ax2 (str): Second factor shown in the plot.
            ax3 (str, optional): Third plotted factor.
            ax4 (str, optional): Fourth mixture component for a tetrahedral
                plot.
            coded (bool): Whether to display coded rather than actual units.
            show_title (bool): Whether to display the generated title.
            show_summary (bool): Whether to display the candidate-set summary.
            show_legend (bool): Whether to display the legend.
            show_grid (bool): Whether to display grid lines.
            marker_color (str): Plotly-compatible marker color.
            show_hover (bool): Whether to display point information on hover.
            show_run_labels (bool): Whether to label candidate points.
            show_replicate_count (bool): Whether to include counts for repeated
                candidate settings.
            aggregate_projected_points (bool): Whether candidates with identical
                plotted coordinates are combined when other factors are omitted.
            axis_label_mode (str): Axis-label format. Supported values are
                ``"symbol_unit"``, ``"symbol"``, and ``"name"``.
            height (int, optional): Figure height in pixels.
            domain (str): Mixture domain to draw. ``"full"`` draws the full
                simplex or tetrahedron; ``"allowed"`` draws the domain covered
                by the candidate points.

        Returns:
            plotly.graph_objects.Figure: Candidate-set visualization.

        Notes:
            Two or three non-mixture factors produce Cartesian plots. Two,
            three, or four mixture components produce mixture-line, ternary, or
            tetrahedral plots.
        """
        options = DesignPlotOptions(
            coded=coded,
            marker_color=marker_color,
            show_title=show_title,
            show_summary=show_summary,
            show_legend=show_legend,
            show_grid=show_grid,
            show_hover=show_hover,
            show_run_labels=show_run_labels,
            show_replicate_count=show_replicate_count,
            aggregate_projected_points=aggregate_projected_points,
            axis_label_mode=axis_label_mode,
            height=height,
            domain=domain,
        )
        return build_design_plot(
            design_matrix=self._cp,
            coded_design_matrix=self._coded_cp,
            factors=self._factors,
            design_type="Candidate Set Design",
            axes=[axis for axis in (ax1, ax2, ax3, ax4) if axis is not None],
            options=options,
        )

    def select_design(self, n: int) -> None:
        """Select a previously computed design containing ``n`` total runs.

        Args:
            n (int): Total run count returned by ``compute_d_optimal``.

        Raises:
            KeyError: If no solution was computed for ``n``.
        """
        if n not in self._best_idx:
            raise KeyError(
                f"No D-optimal solution was computed for {n} runs"
            )
        indices = self._best_idx[n]
        self._coded_design_matrix = self._coded_cp.iloc[indices].reset_index(drop=True)
        self._design_matrix = self._cp.iloc[indices].reset_index(drop=True)
        self._model_matrix = self._build_model_matrix(
            self._coded_design_matrix, self._model_spec
        )
        self._number_of_center_points(self._coded_design_matrix)

    @property
    def log_det(self) -> pd.DataFrame | None:
        """Normalized log-determinant values indexed by total run count."""
        return self._log_det
