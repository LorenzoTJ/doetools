"""Shared candidate-set generation for D-optimal designs."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Mapping

import pandas as pd

from ...utils.grid_builder import lhs_grid
from ..mixture.mixture_cp_generator import build_candidate_points
from ..process import BoxBehnkenDesign, CentralCompositeDesign


MIXTURE_CANDIDATE_CATEGORIES = (
    "vertices",
    "edge_midpoints",
    "face_centroids",
    "global_centroid",
)


class DOptimalCandidateSetMixin:
    """Generate process/mixture candidate sets for D-optimal selection."""

    @staticmethod
    def _normalize_mixture_include(
        mixture_include: str | Collection[str] | None,
    ) -> tuple[str, ...]:
        if mixture_include is None:
            return ()
        if isinstance(mixture_include, str):
            selected = (
                MIXTURE_CANDIDATE_CATEGORIES
                if mixture_include == "all"
                else (mixture_include,)
            )
        else:
            selected = tuple(mixture_include)

        unknown = sorted(set(selected) - set(MIXTURE_CANDIDATE_CATEGORIES))
        if unknown:
            raise ValueError(
                "Unsupported mixture candidate category: "
                + ", ".join(map(str, unknown))
                + ". Use 'all' or exact category names: "
                + ", ".join(MIXTURE_CANDIDATE_CATEGORIES)
            )
        return tuple(dict.fromkeys(selected))

    def _generate_d_optimal_candidates(
        self,
        *,
        factors: Mapping[str, object],
        process_strategy: str | None = None,
        mixture_include: str | Collection[str] | None = (),
        mixture_grid: Mapping[str, int] | None = None,
        lhs_n_samples: int | None = None,
        filters: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return coded and actual candidate matrices for every factor family."""
        factor_names = list(factors)
        continuous = [name for name in factor_names if factors[name].type == "cont"]
        categorical = [name for name in factor_names if factors[name].type == "cat"]
        process = continuous + categorical
        mixture = [name for name in factor_names if factors[name].type == "mix"]

        include = self._normalize_mixture_include(mixture_include)
        if lhs_n_samples is not None and process_strategy != "lhs":
            raise ValueError("lhs_n_samples can be specified only with 'lhs' strategy")
        if process and process_strategy is None:
            raise ValueError("process_strategy is required when process factors are present")
        if not process and process_strategy is not None:
            raise ValueError("process_strategy cannot be used without process factors")
        if mixture and not include and mixture_grid is None:
            raise ValueError(
                "Mixture factors require a non-empty mixture_include or mixture_grid"
            )
        if not mixture and (include or mixture_grid is not None):
            raise ValueError("Mixture candidates were requested without mixture factors")

        process_coded: pd.DataFrame | None = None
        if process_strategy == "lhs":
            if lhs_n_samples is None:
                raise ValueError("lhs_n_samples must be specified for 'lhs' strategy")
            if lhs_n_samples <= 0:
                raise ValueError("lhs_n_samples must be greater than zero")
            if not continuous:
                raise ValueError("'lhs' strategy requires at least one continuous factor")
            process_coded = lhs_grid(continuous, lhs_n_samples, random_state=42)
            process_coded = self._cross_categorical(
                process_coded, factors, categorical
            )
        elif process_strategy == "grid":
            levels = [factors[name].coded_levels for name in process]
            process_coded = pd.DataFrame(
                itertools.product(*levels), columns=process
            )
        elif process_strategy in {"ccc", "ccf", "cci"}:
            if not continuous:
                raise ValueError(
                    f"'{process_strategy}' strategy requires continuous factors"
                )
            ccd = CentralCompositeDesign(
                factors={name: factors[name] for name in continuous},
                design=process_strategy,
                center_points=1,
                replicates=0,
            )
            process_coded = self._cross_categorical(
                ccd._coded_design_matrix, factors, categorical
            )
        elif process_strategy == "bb":
            if not continuous:
                raise ValueError("'bb' strategy requires continuous factors")
            bb = BoxBehnkenDesign(
                factors={name: factors[name] for name in continuous},
                replicates=0,
                center_points=1,
            )
            process_coded = self._cross_categorical(
                bb._coded_design_matrix, factors, categorical
            )
        elif process_strategy is not None:
            raise ValueError(f"Unknown process design strategy: {process_strategy!r}")

        process_actual: pd.DataFrame | None = None
        if process_coded is not None:
            process_actual = self._decode_matrix(process_coded)
            for name in continuous:
                process_actual[name] = process_actual[name].round(
                    factors[name].decimals
                )

        mixture_actual: pd.DataFrame | None = None
        if mixture:
            mixture_actual = build_candidate_points(
                lower_bounds=[factors[name].lower_bound for name in mixture],
                upper_bounds=[factors[name].upper_bound for name in mixture],
                component_names=mixture,
                include=include,
                grid=mixture_grid,
            )[mixture].reset_index(drop=True)

        if process_actual is not None and mixture_actual is not None:
            candidates = process_actual.merge(mixture_actual, how="cross")
        elif process_actual is not None:
            candidates = process_actual.copy()
        elif mixture_actual is not None:
            candidates = mixture_actual.copy()
        else:
            raise ValueError("No candidate points requested")

        if filters:
            mask = pd.Series(True, index=candidates.index)
            for domain_filter in filters:
                result = domain_filter(candidates)
                if not isinstance(result, pd.Series) or len(result) != len(candidates):
                    raise TypeError(
                        "Each filter must return a pandas Series with one value per candidate"
                    )
                mask &= result.astype(bool)
            candidates = candidates.loc[mask]

        candidates = candidates[factor_names].drop_duplicates().reset_index(drop=True)
        if candidates.empty:
            raise ValueError("Candidate-point generation produced an empty set")
        coded = self._code_matrix(candidates)[factor_names].reset_index(drop=True)
        return coded, candidates

    @staticmethod
    def _cross_categorical(
        process_coded: pd.DataFrame,
        factors: Mapping[str, object],
        categorical: list[str],
    ) -> pd.DataFrame:
        if not categorical:
            return process_coded.reset_index(drop=True)
        category_points = pd.DataFrame(
            itertools.product(
                *(factors[name].coded_levels for name in categorical)
            ),
            columns=categorical,
        )
        return process_coded.merge(category_points, how="cross")
