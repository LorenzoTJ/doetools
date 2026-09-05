"""D-optimal augmentation of existing experiments."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...graphs import GraphsMixin
from ...graphs.design_plot_builder import DesignPlotOptions, build_design_plot
from ...utils import (
    CategoricalFactor,
    ContinuousFactor,
    Design,
    MixtureFactor,
    ParetoMixin,
)
from ...utils.model_spec import ModelTerms, compile_model_spec
from .candidate_set import DOptimalCandidateSetMixin
from .optimizer import (
    build_d_optimal_solutions_figure,
    optimize_d_optimal,
)


class DOptAddDesign(DOptimalCandidateSetMixin, Design, GraphsMixin, ParetoMixin):
    """Augment actual-valued existing experiments by D-optimal selection.

    The supplied factor objects define the candidate domain and the common
    coding transformation for existing and candidate rows. Existing
    continuous or mixture values may lie outside the declared candidate bounds;
    ``existing_runs_policy`` determines whether they contribute to the search.

    Parameters:
        factors (Mapping[str, object]): Mapping of names to
            ``ContinuousFactor``, ``CategoricalFactor``, or ``MixtureFactor``
            objects. These factors define the candidate domain and coding.
        source (str | Path | pd.DataFrame): DataFrame or CSV/Excel file containing
            existing factor values in actual units.
        responses (list[str] | None, optional): Existing response columns to
            retain for later export. Defaults to ``None``.

    Raises:
        ValueError: If factors, required file columns, categorical levels, or
            existing mixture rows are invalid.
        TypeError: If factor objects or existing numeric values have unsupported
            types.

    Notes:
        Existing continuous and mixture values may lie outside their declared
        candidate bounds. ``existing_runs_policy`` in ``generate_cp`` controls
        whether such rows contribute to optimization and the selected design.
    """

    def __init__(
        self,
        factors: Mapping[str, object],
        source: str | Path | pd.DataFrame,
        responses: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._factors = self._validate_factors(factors)
        self._factor_names = list(self._factors)

        data = self.upload_file(source)
        missing = [name for name in self._factor_names if name not in data]
        if missing:
            raise ValueError(
                "Factor columns not found in the imported source: "
                + ", ".join(missing)
            )
        if responses is not None:
            missing_responses = [name for name in responses if name not in data]
            if missing_responses:
                raise ValueError(
                    "Response columns not found in the imported source: "
                    + ", ".join(missing_responses)
                )
        if self.EXP_IDX_COL in data:
            data = data.sort_values(self.EXP_IDX_COL).reset_index(drop=True)

        experiments = self._validate_historical_experiments(
            data[self._factor_names].copy()
        )
        self._exp_done = experiments.reset_index(drop=True)
        self._coded_exp_done = self._code_matrix(self._exp_done).reset_index(
            drop=True
        )
        self._responses = (
            data[responses].reset_index(drop=True) if responses is not None else None
        )
        self._response_list = list(responses) if responses is not None else None

        self._filt_exp_done = self._exp_done.copy()
        self._filt_coded_exp_done = self._coded_exp_done.copy()
        self._filt_responses = (
            self._responses.copy() if self._responses is not None else None
        )
        self._coded_cp: pd.DataFrame | None = None
        self._cp: pd.DataFrame | None = None
        self._best_idx: dict[int, list[int]] = {}
        self._log_det: pd.DataFrame | None = None
        self._existing_runs_policy = "within_candidate_domain"
        self._allow_existing_replicates = True
        self._design_type = "D-Optimal Addition"
        self._number_of_center_points(self._coded_exp_done)

    def _validate_factors(
        self, factors: Mapping[str, object]
    ) -> dict[str, object]:
        if not isinstance(factors, Mapping) or not factors:
            raise ValueError("factors must be a non-empty mapping")

        supported = (ContinuousFactor, CategoricalFactor, MixtureFactor)
        validated: dict[str, object] = {}
        for name, factor in factors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Factor names must be non-empty strings")
            if not isinstance(factor, supported):
                raise TypeError(
                    f"Factor {name!r} must be a ContinuousFactor, "
                    "CategoricalFactor, or MixtureFactor"
                )
            if (
                isinstance(factor, ContinuousFactor)
                and factor.lower_bound == factor.upper_bound
            ):
                raise ValueError(
                    f"Continuous factor {name!r} must have distinct bounds"
                )
            validated[name] = factor

        mixture = [
            factor for factor in validated.values() if factor.type == "mix"
        ]
        if mixture:
            if sum(f.lower_bound for f in mixture) > 1 + self.TOLERANCE:
                raise ValueError("Mixture lower bounds must sum to at most 1")
            if sum(f.upper_bound for f in mixture) < 1 - self.TOLERANCE:
                raise ValueError("Mixture upper bounds must sum to at least 1")
        return validated

    def _validate_historical_experiments(
        self, experiments: pd.DataFrame
    ) -> pd.DataFrame:
        for name, factor in self._factors.items():
            series = experiments[name]
            if series.isna().any():
                raise ValueError(f"Factor {name!r} contains missing values")
            if factor.type == "cat":
                unknown = [
                    value
                    for value in pd.unique(series)
                    if value not in factor.levels
                ]
                if unknown:
                    raise ValueError(
                        f"Factor {name!r} contains levels not declared by its "
                        f"CategoricalFactor: {unknown}"
                    )
                continue
            try:
                numeric = pd.to_numeric(series, errors="raise")
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Factor {name!r} must contain only numeric values"
                ) from exc
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError(f"Factor {name!r} must contain only finite values")
            experiments[name] = numeric

        mixture_names = [
            name for name, factor in self._factors.items() if factor.type == "mix"
        ]
        if mixture_names:
            mixture = experiments[mixture_names].to_numpy(dtype=float)
            if np.any(mixture < -self.TOLERANCE) or np.any(
                mixture > 1 + self.TOLERANCE
            ):
                raise ValueError(
                    "Existing mixture component values must lie in [0, 1]"
                )
            if not np.allclose(
                mixture.sum(axis=1), 1.0, atol=1e-10, rtol=0.0
            ):
                raise ValueError(
                    "Existing mixture component values must sum to 1 in every row"
                )
        return experiments

    def set_model_terms(self, terms: ModelTerms = ModelTerms) -> None:
        """Define the model for the combined existing/new design.

        Args:
            terms (ModelTerms): Process and mixture terms to include.

        Notes:
            Changing the model invalidates previously computed and selected
            augmentations.
        """
        process = [
            name
            for name, factor in self._factors.items()
            if factor.type in {"cont", "cat"}
        ]
        mixture = [
            name for name, factor in self._factors.items() if factor.type == "mix"
        ]
        model_spec = compile_model_spec(terms, process, mixture)
        self._model_spec = model_spec
        self._invalidate_computed_design()

    def generate_cp(
        self,
        process_strategy: Literal["grid", "ccf", "cci", "ccc", "bb", "lhs"]
        | None = None,
        mixture_include: str | Collection[str] | None = (),
        mixture_grid: Mapping[str, int] | None = None,
        lhs_n_samples: int | None = None,
        filters: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
        existing_runs_policy: Literal["within_candidate_domain", "all"] =
        "within_candidate_domain",
        allow_existing_replicates: bool = True,
    ) -> None:
        """Generate and retain additional-run candidates for later use.

        Declared bounds and filters affect existing rows only when
        ``existing_runs_policy="within_candidate_domain"``. Regenerating
        candidates invalidates previously computed and selected augmentations.

        Args:
            process_strategy (str | None, optional): Candidate strategy for process
                factors. Supported values are ``"grid"``, ``"lhs"``, ``"ccc"``,
                ``"ccf"``, ``"cci"``, and ``"bb"``. Defaults to ``None``.
            mixture_include (str | Collection[str] | None, optional): Geometric
                mixture candidates. Pass ``"all"`` or select from ``"vertices"``,
                ``"edge_midpoints"``, ``"face_centroids"``, and
                ``"global_centroid"``. Defaults to an empty collection.
            mixture_grid (Mapping[str, int] | None, optional): Optional bounded
                simplex-lattice configuration containing ``"degree"`` and,
                optionally, ``"max_candidates"``. Defaults to ``None``.
            lhs_n_samples (int | None, optional): Number of Latin-hypercube points.
                Required only for ``process_strategy="lhs"``. Defaults to ``None``.
            filters (list[Callable] | None, optional): Functions evaluated on
                actual-valued candidates. Each must return one Boolean value per
                row. Defaults to ``None``.
            existing_runs_policy (str, optional): Use
                ``"within_candidate_domain"`` to retain only existing rows that
                satisfy current bounds and filters, or ``"all"`` to retain every
                valid imported row. Defaults to ``"within_candidate_domain"``.
            allow_existing_replicates (bool, optional): Keep candidate settings
                already present in the imported history. Set to ``False`` to
                remove them. Defaults to ``True``.

        Raises:
            ValueError: If candidate options or the existing-run policy are
                invalid, or if no candidates remain.
            TypeError: If ``allow_existing_replicates`` is not Boolean or a filter
                does not return a valid pandas Series.
        """
        if existing_runs_policy not in {"within_candidate_domain", "all"}:
            raise ValueError(
                "existing_runs_policy must be 'within_candidate_domain' or 'all'"
            )
        if not isinstance(allow_existing_replicates, bool):
            raise TypeError("allow_existing_replicates must be a boolean")
        self.set_domain_filters(filters)
        self._coded_cp, self._cp = self._generate_d_optimal_candidates(
            factors=self._factors,
            process_strategy=process_strategy,
            mixture_include=mixture_include,
            mixture_grid=mixture_grid,
            lhs_n_samples=lhs_n_samples,
            filters=filters,
        )

        historical_mask = pd.Series(True, index=self._exp_done.index)
        if existing_runs_policy == "within_candidate_domain":
            for name, factor in self._factors.items():
                if factor.type in {"cont", "mix"}:
                    historical_mask &= self._exp_done[name].between(
                        factor.lower_bound, factor.upper_bound
                    )
            if filters:
                for domain_filter in filters:
                    result = domain_filter(self._exp_done)
                    if not isinstance(result, pd.Series) or len(result) != len(
                        self._exp_done
                    ):
                        raise TypeError(
                            "Each filter must return a pandas Series with one value per row"
                        )
                    historical_mask &= result.astype(bool)

        retained_indices = self._exp_done.index[historical_mask]
        self._filt_exp_done = self._exp_done.loc[retained_indices].reset_index(
            drop=True
        )
        self._filt_coded_exp_done = self._code_matrix(
            self._filt_exp_done
        ).reset_index(drop=True)
        self._filt_responses = (
            self._responses.loc[retained_indices].reset_index(drop=True)
            if self._responses is not None
            else None
        )

        if not allow_existing_replicates:
            self._coded_cp, self._cp = self._drop_historical_candidates(
                self._coded_cp, self._cp
            )
            if self._cp.empty:
                raise ValueError(
                    "No candidate points remain after removing existing experiments"
                )

        self._existing_runs_policy = existing_runs_policy
        self._allow_existing_replicates = allow_existing_replicates
        self._invalidate_computed_design()

    def _invalidate_computed_design(self) -> None:
        self._best_idx = {}
        self._log_det = None
        self._coded_design_matrix = None
        self._design_matrix = None
        self._model_matrix = None

    def _drop_historical_candidates(
        self,
        coded_candidates: pd.DataFrame,
        candidates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        candidate_keys = self._comparison_keys(candidates)
        historical_keys = set(self._comparison_keys(self._exp_done))
        keep = ~candidate_keys.isin(historical_keys)
        return (
            coded_candidates.loc[keep].reset_index(drop=True),
            candidates.loc[keep].reset_index(drop=True),
        )

    def _comparison_keys(self, matrix: pd.DataFrame) -> pd.Series:
        normalized = matrix[self._factor_names].copy()
        for name, factor in self._factors.items():
            if factor.type in {"cont", "mix"}:
                normalized[name] = pd.to_numeric(normalized[name]).round(12)
            else:
                normalized[name] = normalized[name].astype(str)
        return normalized.apply(tuple, axis=1)

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
        """Compute D-optimal subsets of additional candidate runs.

        Args:
            n_min (int): Smallest number of additional runs to optimize.
            n_max (int): Largest number of additional runs to optimize.
            step (int, optional): Increment between optimized additional-run
                counts. Defaults to 1.
            trials (int, optional): Random exchange-search starts per run count.
                Defaults to 50.
            max_no_improve (int, optional): Maximum rejected improving-exchange
                proposals before a trial stops. Defaults to 5.
            graph (bool, optional): Retained for API compatibility. The returned
                figure is never displayed automatically. Defaults to ``True``.
            random_state (int | numpy.random.Generator | None, optional): Seed or
                generator used by randomized starts. Defaults to ``None``.

        Returns:
            plotly.graph_objects.Figure: D-optimality curve whose x-axis shows the
            combined existing and additional run count.

        Raises:
            ValueError: If candidates or model terms have not been defined, or
                the requested range cannot support the model.
            RuntimeError: If a full-rank combined design cannot be found.
        """
        if self._cp is None:
            raise ValueError("No candidate points defined; call 'generate_cp' first")
        if self._model_spec is None:
            raise ValueError("No model defined; call 'set_model_terms' first")
        candidate_model = self._build_model_matrix(
            self._coded_cp, self._model_spec
        )
        historical_model = self._build_model_matrix(
            self._filt_coded_exp_done, self._model_spec
        )
        result = optimize_d_optimal(
            candidate_model,
            existing_points=historical_model,
            n_min=n_min,
            n_max=n_max,
            step=step,
            trials=trials,
            max_no_improve=max_no_improve,
            random_state=random_state,
        )
        self._best_idx = result.best_indices
        self._log_det = result.log_det
        return build_d_optimal_solutions_figure(
            self._log_det, existing_runs=len(self._filt_exp_done)
        )

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
        """Visualize the generated additional-run candidate set.

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

        Raises:
            ValueError: If candidates have not been generated.

        Notes:
            Two or three non-mixture factors produce Cartesian plots. Two,
            three, or four mixture components produce mixture-line, ternary, or
            tetrahedral plots.
        """
        if self._cp is None:
            raise ValueError("No candidate points defined; call 'generate_cp' first")
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
        """Select a solution containing ``n`` additional experiments.

        Args:
            n (int): Additional-run count returned by ``compute_d_optimal``.

        Raises:
            KeyError: If no augmentation was computed for ``n``.

        Notes:
            The resulting design matrix contains the retained existing rows
            followed by the selected additional rows.
        """
        if n not in self._best_idx:
            raise KeyError(
                f"No D-optimal augmentation was computed for {n} additional runs"
            )
        indices = self._best_idx[n]
        selected_coded = self._coded_cp.iloc[indices].reset_index(drop=True)
        selected = self._cp.iloc[indices].reset_index(drop=True)
        self._coded_design_matrix = pd.concat(
            [self._filt_coded_exp_done, selected_coded], ignore_index=True
        )
        self._design_matrix = pd.concat(
            [self._filt_exp_done, selected], ignore_index=True
        )
        self._model_matrix = self._build_model_matrix(
            self._coded_design_matrix, self._model_spec
        )
        self._number_of_center_points(self._coded_design_matrix)

    def export_experiments(
        self,
        responses: list[str] | None = None,
        randomize: bool = True,
        destination: str | Path = "design_matrix.xlsx",
        coded: bool = False,
    ) -> None:
        """Export retained existing and selected additional experiments.

        Args:
            responses (list[str] | None, optional): Response columns to include.
                Retained existing values are preserved when imported through
                the constructor; new-run values are initialized to zero. Defaults
                to ``None``.
            randomize (bool, optional): Randomize exported execution order.
                Defaults to ``True``.
            destination (str | Path, optional): Destination Excel path. Defaults to
                ``"design_matrix.xlsx"``.
            coded (bool, optional): Export coded rather than actual factor values.
                Defaults to ``False``.

        Raises:
            ValueError: If no augmentation has been selected.
        """
        if self._design_matrix is None:
            raise ValueError("No design selected; call 'select_design' first")
        matrix = (
            self._coded_design_matrix.copy()
            if coded
            else self._design_matrix.copy()
        )
        self._response_list = responses
        for response in responses or []:
            values = np.zeros(len(matrix))
            if (
                self._filt_responses is not None
                and response in self._filt_responses
            ):
                values[: len(self._filt_responses)] = self._filt_responses[
                    response
                ].to_numpy()
            matrix[response] = values
        matrix.insert(0, self.EXP_IDX_COL, matrix.index)
        if randomize:
            order = np.random.permutation(len(matrix))
            matrix.insert(0, self.EXP_ORDER_COL, order)
            matrix = matrix.sort_values(self.EXP_ORDER_COL)
        else:
            matrix.insert(0, self.EXP_ORDER_COL, matrix.index)
        matrix.to_excel(destination, index=False)

    @property
    def log_det(self) -> pd.DataFrame | None:
        """Normalized log-determinants indexed by additional-run count."""
        return self._log_det

    def build_design_matrix(self):
        return NotImplementedError("This design cannot be modified")
