"""Shared validation for external factor settings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class _ExternalPointValidationMixin:
    """Validate and convert external points without exposing public methods."""

    @staticmethod
    def _numeric_external_column(
        series: pd.Series,
        name: str,
        *,
        context: str,
    ) -> pd.Series:
        try:
            numeric = pd.to_numeric(series, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} column {name!r} must contain numeric values only."
            ) from exc
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(
                f"{context} column {name!r} contains missing or non-finite values."
            )
        return numeric

    def _continuous_external_column(
        self,
        series: pd.Series,
        name: str,
        *,
        coded: bool,
        context: str,
        row_label: str,
    ) -> tuple[pd.Series, pd.Series]:
        factor = self._factors[name]
        numeric = self._numeric_external_column(series, name, context=context)
        lower = float(factor.lower_bound)
        upper = float(factor.upper_bound)
        span = upper - lower
        if np.isclose(span, 0.0):
            raise ValueError(
                f"Continuous factor {name!r} has identical lower and upper bounds."
            )
        center = (lower + upper) / 2.0
        decimals = int(getattr(factor, "decimals", 12))

        actual = center + numeric * span / 2.0 if coded else numeric
        actual = actual.round(decimals)

        design_values = pd.to_numeric(
            self._design_matrix[name], errors="coerce"
        ).to_numpy(dtype=float)
        allowed_lower = min(lower, float(np.nanmin(design_values)))
        allowed_upper = max(upper, float(np.nanmax(design_values)))
        tolerance = max(
            float(getattr(self, "TOLERANCE", 1e-12)),
            10.0 ** (-decimals) * 0.5,
        )
        outside = (actual < allowed_lower - tolerance) | (
            actual > allowed_upper + tolerance
        )
        if outside.any():
            rows = list(np.flatnonzero(outside.to_numpy()) + 1)
            raise ValueError(
                f"{context} factor {name!r} is outside the allowed domain "
                f"[{allowed_lower}, {allowed_upper}] at {row_label} {rows}."
            )

        coded_values = ((actual - center) * 2.0 / span).astype(float)
        return actual.astype(float), coded_values

    def _categorical_external_column(
        self,
        series: pd.Series,
        name: str,
        *,
        coded: bool,
        context: str,
        row_label: str,
    ) -> tuple[pd.Series, pd.Series]:
        factor = self._factors[name]
        levels = list(factor.levels)
        codes = np.asarray(factor.coded_levels, dtype=float)

        if coded:
            numeric = self._numeric_external_column(series, name, context=context)
            actual_values: list[Any] = []
            coded_values: list[float] = []
            for row, value in enumerate(numeric.to_numpy(dtype=float), start=1):
                matches = np.flatnonzero(
                    np.isclose(value, codes, atol=1e-10, rtol=0.0)
                )
                if len(matches) != 1:
                    raise ValueError(
                        f"Unknown coded level {value!r} for categorical factor "
                        f"{name!r} at {row_label[:-1]} {row}."
                    )
                position = int(matches[0])
                actual_values.append(levels[position])
                coded_values.append(float(codes[position]))
            return (
                pd.Series(actual_values, index=series.index, name=name),
                pd.Series(coded_values, index=series.index, name=name),
            )

        actual_values = []
        coded_values = []
        for row, value in enumerate(series, start=1):
            matches = [index for index, level in enumerate(levels) if value == level]
            if len(matches) != 1:
                raise ValueError(
                    f"Unknown level {value!r} for categorical factor {name!r} "
                    f"at {row_label[:-1]} {row}. Available levels: {levels}."
                )
            position = matches[0]
            actual_values.append(levels[position])
            coded_values.append(float(codes[position]))
        return (
            pd.Series(actual_values, index=series.index, name=name),
            pd.Series(coded_values, index=series.index, name=name),
        )

    def _mixture_external_column(
        self,
        series: pd.Series,
        name: str,
        *,
        context: str,
        row_label: str,
    ) -> tuple[pd.Series, pd.Series]:
        factor = self._factors[name]
        decimals = int(getattr(factor, "decimals", 12))
        values = self._numeric_external_column(
            series, name, context=context
        ).round(decimals)
        tolerance = max(
            float(getattr(self, "TOLERANCE", 1e-12)),
            10.0 ** (-decimals) * 0.5,
        )
        outside = (values < float(factor.lower_bound) - tolerance) | (
            values > float(factor.upper_bound) + tolerance
        )
        if outside.any():
            rows = list(np.flatnonzero(outside.to_numpy()) + 1)
            raise ValueError(
                f"{context} mixture factor {name!r} is outside its bounds "
                f"at {row_label} {rows}."
            )
        return values.astype(float), values.astype(float).copy()

    def _prepare_external_points(
        self,
        frame: pd.DataFrame,
        *,
        coded: bool,
        context: str,
        row_label: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        actual = pd.DataFrame(index=frame.index)
        coded_points = pd.DataFrame(index=frame.index)
        mixture_names: list[str] = []

        for name, factor in self._factors.items():
            if factor.type == "cont":
                actual[name], coded_points[name] = self._continuous_external_column(
                    frame[name],
                    name,
                    coded=coded,
                    context=context,
                    row_label=row_label,
                )
            elif factor.type == "cat":
                actual[name], coded_points[name] = self._categorical_external_column(
                    frame[name],
                    name,
                    coded=coded,
                    context=context,
                    row_label=row_label,
                )
            elif factor.type == "mix":
                mixture_names.append(name)
                actual[name], coded_points[name] = self._mixture_external_column(
                    frame[name],
                    name,
                    context=context,
                    row_label=row_label,
                )
            else:
                raise ValueError(f"Unsupported factor type {factor.type!r} for {name!r}.")

        if mixture_names:
            decimals = [
                int(getattr(self._factors[name], "decimals", 12))
                for name in mixture_names
            ]
            tolerance = max(
                float(getattr(self, "TOLERANCE", 1e-12)),
                sum(10.0 ** (-value) * 0.5 for value in decimals),
            )
            totals = actual[mixture_names].sum(axis=1).to_numpy(dtype=float)
            invalid = ~np.isclose(totals, 1.0, atol=tolerance, rtol=0.0)
            if invalid.any():
                rows = list(np.flatnonzero(invalid) + 1)
                raise ValueError(
                    f"{context} mixture components must sum to 1 at every "
                    f"{row_label[:-1]}; invalid {row_label}: {rows}."
                )

        self._validate_external_domain_filters(
            actual,
            context=context,
            row_label=row_label,
        )
        return actual, coded_points

    def _validate_external_domain_filters(
        self,
        actual: pd.DataFrame,
        *,
        context: str,
        row_label: str,
    ) -> None:
        mask = np.ones(len(actual), dtype=bool)
        for domain_filter in list(getattr(self, "_domain_filters", []) or []):
            result = domain_filter(actual.copy(deep=True))
            if np.isscalar(result):
                filter_mask = np.full(len(actual), bool(result), dtype=bool)
            else:
                if not hasattr(result, "__len__") or len(result) != len(actual):
                    raise ValueError(
                        f"Each domain filter must return one value per "
                        f"{context.lower()} {row_label[:-1]}."
                    )
                values = (
                    result.reindex(actual.index)
                    if isinstance(result, pd.Series)
                    else pd.Series(result, index=actual.index)
                )
                if values.ndim != 1:
                    raise ValueError(
                        "Each domain filter must return a one-dimensional mask."
                    )
                filter_mask = values.fillna(False).to_numpy(dtype=bool)
            mask &= filter_mask
        if not mask.all():
            rows = list(np.flatnonzero(~mask) + 1)
            raise ValueError(
                f"{context} {row_label} outside the configured domain: {rows}."
            )
