"""Internal Pareto-front analysis support for experimental designs."""

from __future__ import annotations

import itertools
from collections.abc import Collection, Mapping, Sequence
from math import prod
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


_MIXTURE_CANDIDATE_CATEGORIES = (
    "vertices",
    "edge_midpoints",
    "face_centroids",
    "global_centroid",
)


class _ParetoAnalysisMixin:
    """Provide deterministic, model-based Pareto-front computation."""

    _PARETO_ID_COLUMN = "Candidate Id"
    _PARETO_FLAG_COLUMN = "_is_pareto"

    def _initialize_pareto_analysis(self) -> None:
        self._pareto_candidates: pd.DataFrame | None = None
        self._pareto_objectives: tuple[str, ...] | None = None

    def _invalidate_pareto_results(self) -> None:
        self._pareto_candidates = None
        self._pareto_objectives = None

    @staticmethod
    def _normalize_mixture_include(
        mixture_include: str | Collection[str] | None,
    ) -> tuple[str, ...]:
        if mixture_include is None:
            return ()
        if isinstance(mixture_include, str):
            selected = (
                _MIXTURE_CANDIDATE_CATEGORIES
                if mixture_include == "all"
                else (mixture_include,)
            )
        else:
            selected = tuple(mixture_include)

        unknown = sorted(set(selected) - set(_MIXTURE_CANDIDATE_CATEGORIES))
        if unknown:
            raise ValueError(
                "Unsupported mixture candidate category: "
                + ", ".join(map(str, unknown))
                + ". Use 'all' or exact category names: "
                + ", ".join(_MIXTURE_CANDIDATE_CATEGORIES)
            )
        return tuple(dict.fromkeys(selected))

    def _validate_pareto_objectives(
        self, responses: Collection[str] | None
    ) -> tuple[str, ...]:
        if self._model_matrix is None:
            raise ValueError("No model matrix defined; call set_model_terms first.")
        if self._mlr_wrapper is None:
            raise ValueError("No MLR model computed; call compute_mlr_model first.")
        if self._responses is None or not self._response_list:
            raise ValueError("No responses defined; call import_responses first.")

        if responses is None:
            objectives = tuple(self._response_list)
        else:
            if isinstance(responses, str):
                raise TypeError("responses must be a collection of response names, not a string")
            objectives = tuple(responses)

        if len(objectives) < 2:
            raise ValueError("Pareto analysis requires at least two objective responses.")
        if not all(isinstance(name, str) for name in objectives):
            raise TypeError("responses must contain only response-name strings.")
        if len(set(objectives)) != len(objectives):
            raise ValueError("responses must contain unique response names.")

        available = set(self._response_list)
        unknown = [name for name in objectives if name not in available]
        if unknown:
            raise ValueError(
                f"Unknown response(s): {unknown}. Available responses: {self._response_list}"
            )

        missing_models = [name for name in objectives if name not in self._mlr_wrapper.results]
        if missing_models:
            raise ValueError(f"No fitted model for response(s): {missing_models}")

        missing_directions = [
            name
            for name in objectives
            if name not in self._response_conditions
            or "maximize" not in self._response_conditions[name]
        ]
        if missing_directions:
            raise ValueError(
                "Set response conditions before Pareto analysis for: "
                + ", ".join(missing_directions)
            )
        invalid_directions = [
            name
            for name in objectives
            if not isinstance(self._response_conditions[name]["maximize"], bool)
        ]
        if invalid_directions:
            raise TypeError(
                "The maximize condition must be boolean for: "
                + ", ".join(invalid_directions)
            )
        return objectives

    @staticmethod
    def _coerce_level_sequence(name: str, values: Sequence[Any]) -> list[Any]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"factor_levels[{name!r}] must be a sequence of levels")
        try:
            levels = list(values)
        except TypeError as exc:
            raise TypeError(
                f"factor_levels[{name!r}] must be a sequence of levels"
            ) from exc
        if not levels:
            raise ValueError(f"factor_levels[{name!r}] cannot be empty")
        return levels

    def _validate_continuous_levels(
        self, name: str, values: Sequence[Any]
    ) -> list[float]:
        factor = self._factors[name]
        raw_levels = self._coerce_level_sequence(name, values)
        levels: list[float] = []
        precision_tolerance = max(self.TOLERANCE, 10.0 ** (-factor.decimals) * 1e-9)
        for value in raw_levels:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(
                    f"Continuous factor {name!r} levels must be numeric and not boolean"
                )
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Continuous factor {name!r} levels must be finite")
            rounded = float(np.round(numeric, factor.decimals))
            if not np.isclose(numeric, rounded, rtol=0.0, atol=precision_tolerance):
                raise ValueError(
                    f"Continuous factor {name!r} level {value!r} exceeds "
                    f"the configured precision of {factor.decimals} decimal places"
                )
            if (
                rounded < factor.lower_bound - self.TOLERANCE
                or rounded > factor.upper_bound + self.TOLERANCE
            ):
                raise ValueError(
                    f"Continuous factor {name!r} level {value!r} is outside "
                    f"[{factor.lower_bound}, {factor.upper_bound}]"
                )
            levels.append(rounded)
        if len(set(levels)) != len(levels):
            raise ValueError(f"Continuous factor {name!r} levels must be unique")
        return levels

    def _validate_categorical_levels(
        self, name: str, values: Sequence[Any]
    ) -> list[str]:
        factor = self._factors[name]
        levels = self._coerce_level_sequence(name, values)
        invalid_types = [value for value in levels if not isinstance(value, str)]
        if invalid_types:
            raise TypeError(f"Categorical factor {name!r} levels must be strings")
        unknown = [value for value in levels if value not in factor.levels]
        if unknown:
            raise ValueError(
                f"Unknown levels for categorical factor {name!r}: {unknown}. "
                f"Declared levels: {factor.levels}"
            )
        if len(set(levels)) != len(levels):
            raise ValueError(f"Categorical factor {name!r} levels must be unique")
        return levels

    def _process_level_mapping(
        self,
        factor_levels: Mapping[str, Sequence[float | str]] | None,
    ) -> tuple[list[str], list[list[Any]]]:
        if factor_levels is None:
            overrides: Mapping[str, Sequence[float | str]] = {}
        elif not isinstance(factor_levels, Mapping):
            raise TypeError("factor_levels must be a mapping or None")
        else:
            overrides = factor_levels

        unknown = [name for name in overrides if name not in self._factors]
        if unknown:
            raise ValueError(f"Unknown factor(s) in factor_levels: {unknown}")
        mixture_overrides = [name for name in overrides if self._factors[name].type == "mix"]
        if mixture_overrides:
            raise ValueError(
                "Mixture factors must be configured with mixture_include or "
                f"mixture_grid, not factor_levels: {mixture_overrides}"
            )

        process_names = [
            name for name, factor in self._factors.items() if factor.type in {"cont", "cat"}
        ]
        levels_by_factor: list[list[Any]] = []
        for name in process_names:
            factor = self._factors[name]
            if name not in overrides:
                levels_by_factor.append(list(factor.levels))
            elif factor.type == "cont":
                levels_by_factor.append(
                    self._validate_continuous_levels(name, overrides[name])
                )
            else:
                levels_by_factor.append(
                    self._validate_categorical_levels(name, overrides[name])
                )
        return process_names, levels_by_factor

    def _build_pareto_candidates(
        self,
        *,
        factor_levels: Mapping[str, Sequence[float | str]] | None,
        mixture_include: str | Collection[str] | None,
        mixture_grid: Mapping[str, int] | None,
        max_candidates: int,
    ) -> pd.DataFrame:
        if (
            not isinstance(max_candidates, int)
            or isinstance(max_candidates, bool)
            or max_candidates < 1
        ):
            raise ValueError("max_candidates must be a positive integer")

        process_names, levels_by_factor = self._process_level_mapping(factor_levels)
        process_count = prod(len(levels) for levels in levels_by_factor)
        if process_names and process_count > max_candidates:
            raise ValueError(
                f"Requested process grid contains {process_count} candidates, "
                f"which exceeds max_candidates={max_candidates}."
            )

        process_candidates: pd.DataFrame | None = None
        if process_names:
            process_candidates = pd.DataFrame(
                itertools.product(*levels_by_factor), columns=process_names
            )

        mixture_names = [
            name for name, factor in self._factors.items() if factor.type == "mix"
        ]
        include = self._normalize_mixture_include(mixture_include)
        if mixture_grid is not None and not isinstance(mixture_grid, Mapping):
            raise TypeError("mixture_grid must be a mapping or None")
        if not mixture_names and (include or mixture_grid is not None):
            raise ValueError("Mixture candidates were requested without mixture factors")

        mixture_candidates: pd.DataFrame | None = None
        if mixture_names:
            from ..design.mixture.mixture_cp_generator import build_candidate_points

            process_multiplier = (
                len(process_candidates) if process_candidates is not None else 1
            )
            mixture_cap = max_candidates // process_multiplier
            grid = dict(mixture_grid) if mixture_grid is not None else None
            if not include and grid is None:
                grid = {"degree": 5, "max_candidates": mixture_cap}
            elif grid is not None:
                configured_cap = grid.get("max_candidates", mixture_cap)
                if isinstance(configured_cap, int) and not isinstance(
                    configured_cap, bool
                ):
                    grid["max_candidates"] = min(configured_cap, mixture_cap)
            mixture_candidates = build_candidate_points(
                lower_bounds=[self._factors[name].lower_bound for name in mixture_names],
                upper_bounds=[self._factors[name].upper_bound for name in mixture_names],
                component_names=mixture_names,
                include=include,
                grid=grid,
            )[mixture_names].reset_index(drop=True)

        process_size = len(process_candidates) if process_candidates is not None else 1
        mixture_size = len(mixture_candidates) if mixture_candidates is not None else 1
        requested_size = process_size * mixture_size
        if requested_size > max_candidates:
            raise ValueError(
                f"Requested candidate grid contains {requested_size} candidates, "
                f"which exceeds max_candidates={max_candidates}."
            )

        if process_candidates is not None and mixture_candidates is not None:
            candidates = process_candidates.merge(mixture_candidates, how="cross")
        elif process_candidates is not None:
            candidates = process_candidates.copy()
        elif mixture_candidates is not None:
            candidates = mixture_candidates.copy()
        else:
            raise ValueError("No candidate factors are available")

        factor_names = list(self._factors)
        candidates = candidates[factor_names].drop_duplicates().reset_index(drop=True)
        if self._domain_filters:
            mask = np.ones(len(candidates), dtype=bool)
            for domain_filter in self._domain_filters:
                result = domain_filter(candidates.copy())
                if not isinstance(result, pd.Series) or len(result) != len(candidates):
                    raise TypeError(
                        "Each domain filter must return a pandas Series with one "
                        "boolean value per candidate"
                    )
                if not is_bool_dtype(result.dtype) or result.isna().any():
                    raise TypeError(
                        "Each domain filter must return a boolean pandas Series "
                        "without missing values"
                    )
                mask &= result.to_numpy(dtype=bool)
            candidates = candidates.loc[mask].reset_index(drop=True)

        if candidates.empty:
            raise ValueError("Candidate generation produced no domain-valid points")
        candidates.insert(
            0,
            self._PARETO_ID_COLUMN,
            [f"P{index}" for index in range(1, len(candidates) + 1)],
        )
        return candidates

    @staticmethod
    def _pareto_mask(objectives: np.ndarray) -> np.ndarray:
        """Return an exact non-dominated mask for minimization objectives."""
        front: list[int] = []
        for index, point in enumerate(objectives):
            if front:
                current = objectives[front]
                dominated = np.all(current <= point, axis=1) & np.any(
                    current < point, axis=1
                )
                if dominated.any():
                    continue
                dominates = np.all(point <= current, axis=1) & np.any(
                    point < current, axis=1
                )
                if dominates.any():
                    front = [
                        existing
                        for existing, remove in zip(front, dominates)
                        if not remove
                    ]
            front.append(index)

        mask = np.zeros(len(objectives), dtype=bool)
        mask[front] = True
        return mask

    def compute_pareto_front(
        self,
        responses: Collection[str] | None = None,
        *,
        factor_levels: Mapping[str, Sequence[float | str]] | None = None,
        mixture_include: str | Collection[str] | None = (),
        mixture_grid: Mapping[str, int] | None = None,
        max_candidates: int = 100_000,
    ) -> None:
        """Compute the Pareto front over every domain-valid candidate.

        Response lower and upper limits are reference guides for plots; they do
        not restrict Pareto candidates. Factor bounds, mixture constraints, and
        filters set with ``set_domain_filters`` define the candidate domain.
        """
        self._invalidate_pareto_results()
        objectives = self._validate_pareto_objectives(responses)
        candidates = self._build_pareto_candidates(
            factor_levels=factor_levels,
            mixture_include=mixture_include,
            mixture_grid=mixture_grid,
            max_candidates=max_candidates,
        )

        factor_names = list(self._factors)
        coded = self._code_matrix(candidates[factor_names])
        predictions = self._predict(coded, list(objectives)).reset_index(drop=True)

        finite = np.isfinite(predictions.to_numpy(dtype=float))
        if not finite.all():
            details = []
            for column_index, response in enumerate(objectives):
                invalid_rows = np.flatnonzero(~finite[:, column_index])
                if invalid_rows.size:
                    ids = candidates.loc[
                        invalid_rows[:10], self._PARETO_ID_COLUMN
                    ].tolist()
                    suffix = (
                        ""
                        if invalid_rows.size <= 10
                        else f" (+{invalid_rows.size - 10} more)"
                    )
                    details.append(f"{response}: {ids}{suffix}")
            raise ValueError(
                "Pareto objective predictions must be finite; invalid candidates by "
                "response: " + "; ".join(details)
            )

        objective_values = predictions.loc[:, list(objectives)].to_numpy(dtype=float)
        transformed = objective_values.copy()
        for column_index, response in enumerate(objectives):
            if self._response_conditions[response]["maximize"]:
                transformed[:, column_index] *= -1.0

        pareto_mask = self._pareto_mask(transformed)
        evaluated = pd.concat([candidates.reset_index(drop=True), predictions], axis=1)
        evaluated[self._PARETO_FLAG_COLUMN] = pareto_mask
        self._pareto_candidates = evaluated
        self._pareto_objectives = objectives

    def get_pareto_front(self) -> pd.DataFrame:
        """Return a defensive copy of the computed non-dominated candidates."""
        if self._pareto_candidates is None:
            raise ValueError("No Pareto front computed; call compute_pareto_front first.")
        front = self._pareto_candidates.loc[
            self._pareto_candidates[self._PARETO_FLAG_COLUMN]
        ].drop(columns=self._PARETO_FLAG_COLUMN)
        return front.reset_index(drop=True).copy(deep=True)
