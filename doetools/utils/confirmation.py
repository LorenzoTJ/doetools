"""External confirmation-run validation for fitted OLS models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


class ConfirmationRunsMixin:
    """Load confirmation runs and evaluate PIMean prediction intervals."""

    def _initialize_confirmation_runs(self) -> None:
        self._confirmation_points: pd.DataFrame | None = None
        self._coded_confirmation_points: pd.DataFrame | None = None
        self._confirmation_responses: pd.DataFrame | None = None
        self._confirmation_metadata: pd.DataFrame | None = None
        self._confirmation_setting_ids: pd.Series | None = None

    def _require_confirmation_model(self):
        wrapper = getattr(self, "_mlr_wrapper", None)
        results = getattr(wrapper, "results", None)
        if not isinstance(results, dict) or not results:
            raise ValueError(
                "No MLR model computed; call compute_mlr_model first."
            )
        return results

    def _require_confirmation_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        actual = getattr(self, "_confirmation_points", None)
        coded = getattr(self, "_coded_confirmation_points", None)
        responses = getattr(self, "_confirmation_responses", None)
        setting_ids = getattr(self, "_confirmation_setting_ids", None)
        if not all(
            isinstance(value, expected)
            for value, expected in (
                (actual, pd.DataFrame),
                (coded, pd.DataFrame),
                (responses, pd.DataFrame),
                (setting_ids, pd.Series),
            )
        ):
            raise ValueError(
                "No confirmation runs available; call load_confirmation_runs first."
            )
        return actual, coded, responses, setting_ids

    @staticmethod
    def _numeric_column(series: pd.Series, name: str) -> pd.Series:
        try:
            numeric = pd.to_numeric(series, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Confirmation column {name!r} must contain numeric values only."
            ) from exc
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(
                f"Confirmation column {name!r} contains missing or non-finite values."
            )
        return numeric

    def _continuous_confirmation_column(
        self,
        series: pd.Series,
        name: str,
        *,
        coded: bool,
    ) -> tuple[pd.Series, pd.Series]:
        factor = self._factors[name]
        numeric = self._numeric_column(series, name)
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
        tolerance = max(float(getattr(self, "TOLERANCE", 1e-12)), 10.0 ** (-decimals) * 0.5)
        outside = (actual < allowed_lower - tolerance) | (
            actual > allowed_upper + tolerance
        )
        if outside.any():
            runs = list(np.flatnonzero(outside.to_numpy()) + 1)
            raise ValueError(
                f"Confirmation factor {name!r} is outside the allowed domain "
                f"[{allowed_lower}, {allowed_upper}] at runs {runs}."
            )

        coded_values = ((actual - center) * 2.0 / span).astype(float)
        return actual.astype(float), coded_values

    def _categorical_confirmation_column(
        self,
        series: pd.Series,
        name: str,
        *,
        coded: bool,
    ) -> tuple[pd.Series, pd.Series]:
        factor = self._factors[name]
        levels = list(factor.levels)
        codes = np.asarray(factor.coded_levels, dtype=float)

        if coded:
            numeric = self._numeric_column(series, name)
            actual_values: list[Any] = []
            coded_values: list[float] = []
            for run, value in enumerate(numeric.to_numpy(dtype=float), start=1):
                matches = np.flatnonzero(
                    np.isclose(value, codes, atol=1e-10, rtol=0.0)
                )
                if len(matches) != 1:
                    raise ValueError(
                        f"Unknown coded level {value!r} for categorical factor "
                        f"{name!r} at run {run}."
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
        for run, value in enumerate(series, start=1):
            matches = [index for index, level in enumerate(levels) if value == level]
            if len(matches) != 1:
                raise ValueError(
                    f"Unknown level {value!r} for categorical factor {name!r} "
                    f"at run {run}. Available levels: {levels}."
                )
            position = matches[0]
            actual_values.append(levels[position])
            coded_values.append(float(codes[position]))
        return (
            pd.Series(actual_values, index=series.index, name=name),
            pd.Series(coded_values, index=series.index, name=name),
        )

    def _mixture_confirmation_column(
        self,
        series: pd.Series,
        name: str,
    ) -> tuple[pd.Series, pd.Series]:
        factor = self._factors[name]
        decimals = int(getattr(factor, "decimals", 12))
        values = self._numeric_column(series, name).round(decimals)
        tolerance = max(float(getattr(self, "TOLERANCE", 1e-12)), 10.0 ** (-decimals) * 0.5)
        outside = (values < float(factor.lower_bound) - tolerance) | (
            values > float(factor.upper_bound) + tolerance
        )
        if outside.any():
            runs = list(np.flatnonzero(outside.to_numpy()) + 1)
            raise ValueError(
                f"Confirmation mixture factor {name!r} is outside its bounds "
                f"at runs {runs}."
            )
        return values.astype(float), values.astype(float).copy()

    def _prepare_confirmation_points(
        self,
        frame: pd.DataFrame,
        *,
        coded: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        actual = pd.DataFrame(index=frame.index)
        coded_points = pd.DataFrame(index=frame.index)
        mixture_names: list[str] = []

        for name, factor in self._factors.items():
            if factor.type == "cont":
                actual[name], coded_points[name] = self._continuous_confirmation_column(
                    frame[name], name, coded=coded
                )
            elif factor.type == "cat":
                actual[name], coded_points[name] = self._categorical_confirmation_column(
                    frame[name], name, coded=coded
                )
            elif factor.type == "mix":
                mixture_names.append(name)
                actual[name], coded_points[name] = self._mixture_confirmation_column(
                    frame[name], name
                )
            else:
                raise ValueError(f"Unsupported factor type {factor.type!r} for {name!r}.")

        if mixture_names:
            decimals = [int(getattr(self._factors[name], "decimals", 12)) for name in mixture_names]
            tolerance = max(
                float(getattr(self, "TOLERANCE", 1e-12)),
                sum(10.0 ** (-value) * 0.5 for value in decimals),
            )
            totals = actual[mixture_names].sum(axis=1).to_numpy(dtype=float)
            invalid = ~np.isclose(totals, 1.0, atol=tolerance, rtol=0.0)
            if invalid.any():
                runs = list(np.flatnonzero(invalid) + 1)
                raise ValueError(
                    "Confirmation mixture components must sum to 1 at every run; "
                    f"invalid runs: {runs}."
                )

        self._validate_confirmation_domain_filters(actual)
        return actual.reset_index(drop=True), coded_points.reset_index(drop=True)

    def _validate_confirmation_domain_filters(self, actual: pd.DataFrame) -> None:
        mask = np.ones(len(actual), dtype=bool)
        for domain_filter in list(getattr(self, "_domain_filters", []) or []):
            result = domain_filter(actual.copy(deep=True))
            if np.isscalar(result):
                filter_mask = np.full(len(actual), bool(result), dtype=bool)
            else:
                if not hasattr(result, "__len__") or len(result) != len(actual):
                    raise ValueError(
                        "Each domain filter must return one value per confirmation run."
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
            runs = list(np.flatnonzero(~mask) + 1)
            raise ValueError(
                f"Confirmation runs outside the configured domain: {runs}."
            )

    def load_confirmation_runs(
        self,
        file_path: str | Path,
        coded: bool = False,
    ) -> None:
        """Load and validate external confirmation runs for a fitted model.

        Args:
            file_path (str | Path): Excel or CSV file containing every factor and
                fitted-response column.
            coded (bool): Whether process and categorical factor values are coded.
                Mixture components are unaffected. Defaults to ``False``.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            TypeError: If ``coded`` is not Boolean.
            ValueError: If no MLR model has been fitted or the file contains
                missing, duplicate, nonnumeric, or out-of-domain data.

        Notes:
            Each row is one confirmation run. Identical factor settings are treated
            as replicates.
        """
        if not isinstance(coded, bool):
            raise TypeError("coded must be a boolean.")
        fitted_results = self._require_confirmation_model()
        frame = self.upload_file(file_path)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("The confirmation-runs file must contain at least one row.")
        if frame.columns.has_duplicates:
            duplicates = list(frame.columns[frame.columns.duplicated()].unique())
            raise ValueError(f"Confirmation-runs file contains duplicate columns: {duplicates}.")

        factor_names = list(self._factors)
        response_names = list(fitted_results)
        missing_factors = [name for name in factor_names if name not in frame.columns]
        missing_responses = [name for name in response_names if name not in frame.columns]
        if missing_factors:
            raise ValueError(
                f"Factor columns not found in confirmation file: {missing_factors}."
            )
        if missing_responses:
            raise ValueError(
                f"Response columns not found in confirmation file: {missing_responses}."
            )

        # Build every object locally so a failed load cannot alter existing state.
        actual, coded_points = self._prepare_confirmation_points(
            frame[factor_names].copy(), coded=coded
        )
        responses = pd.DataFrame(index=frame.index)
        for name in response_names:
            responses[name] = self._numeric_column(frame[name], name)
        responses = responses.reset_index(drop=True)

        metadata_columns = [
            name for name in ("Exp. Order", "Exp. Idx") if name in frame.columns
        ]
        metadata = (
            frame[metadata_columns].reset_index(drop=True).copy()
            if metadata_columns
            else None
        )
        setting_codes, _ = pd.factorize(
            pd.MultiIndex.from_frame(actual[factor_names]), sort=False
        )
        setting_ids = pd.Series(setting_codes + 1, name="Setting", dtype=int)

        self._confirmation_points = actual
        self._coded_confirmation_points = coded_points
        self._confirmation_responses = responses
        self._confirmation_metadata = metadata
        self._confirmation_setting_ids = setting_ids

    def get_confirmation_points(self, coded: bool = False) -> pd.DataFrame:
        """Return confirmation factor settings, including replicated rows.

        Args:
            coded (bool): Whether to return coded factor settings. Defaults to
                ``False``.

        Returns:
            pd.DataFrame: A defensive copy of the confirmation factor settings.

        Raises:
            TypeError: If ``coded`` is not Boolean.
            ValueError: If confirmation runs have not been loaded.
        """
        if not isinstance(coded, bool):
            raise TypeError("coded must be a boolean.")
        actual, coded_points, _, _ = self._require_confirmation_data()
        points = coded_points if coded else actual
        return points.copy(deep=True)

    def get_confirmation_responses(
        self,
        response: str | None = None,
    ) -> pd.DataFrame:
        """Return all confirmation responses or one selected response.

        Args:
            response (str | None): Response to select. If ``None``, return every
                confirmation response. Defaults to ``None``.

        Returns:
            pd.DataFrame: A defensive copy of the selected response data.

        Raises:
            ValueError: If confirmation runs have not been loaded or ``response``
                is unavailable.
        """
        _, _, responses, _ = self._require_confirmation_data()
        if response is None:
            return responses.copy(deep=True)
        if response not in responses.columns:
            raise ValueError(
                f"Response {response!r} is not available in confirmation runs. "
                f"Available responses: {list(responses.columns)}."
            )
        return responses[[response]].copy(deep=True)

    def get_confirmation_results(
        self,
        response: str,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Return grouped PIMean confirmation results for one response.

        Args:
            response (str): Fitted response to evaluate.
            alpha (float): Pointwise significance level for the two-sided PIMean
                interval. Must lie strictly between 0 and 1. Defaults to ``0.05``.

        Returns:
            pd.DataFrame: One row per distinct confirmation setting, including
                factor values, replicate count, observed mean, prediction,
                residual, interval limits, and the ``Within PI`` result.

        Raises:
            TypeError: If ``alpha`` is not numeric.
            ValueError: If the response or confirmation data are unavailable,
                ``alpha`` is invalid, or the fitted model cannot provide a PIMean
                interval.

        Notes:
            Replicates are grouped by identical factor settings. Residuals use
            ``Observed - Predicted``. Intervals are pointwise and are not adjusted
            across multiple confirmation settings.
        """
        fitted_results = self._require_confirmation_model()
        actual, coded_points, responses, setting_ids = self._require_confirmation_data()
        if response not in fitted_results or response not in responses.columns:
            available = [name for name in fitted_results if name in responses.columns]
            raise ValueError(
                f"Response {response!r} is not available for confirmation. "
                f"Available responses: {available}."
            )
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.number)):
            raise TypeError("alpha must be a numeric value between 0 and 1.")
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        fitted = fitted_results[response]
        model = fitted.model
        df_resid = float(model.df_resid)
        ms_res = float(fitted.anova.get("MS_res", np.nan))
        if df_resid <= 0 or not np.isfinite(ms_res) or ms_res < 0:
            raise ValueError(
                "PIMean is unavailable because the fitted model has no valid "
                "residual degrees of freedom or residual mean square."
            )

        setting_values = setting_ids.to_numpy(dtype=int)
        unique_ids = pd.unique(setting_values)
        positions = [int(np.flatnonzero(setting_values == value)[0]) for value in unique_ids]
        counts = np.asarray(
            [np.count_nonzero(setting_values == value) for value in unique_ids],
            dtype=int,
        )
        unique_coded = coded_points.iloc[positions].reset_index(drop=True)
        model_points = self._build_model_matrix(unique_coded, self._model_spec)
        expected_columns = list(model.params.index)
        if list(model_points.columns) != expected_columns:
            try:
                model_points = model_points[expected_columns]
            except KeyError as exc:
                raise ValueError(
                    "Confirmation model matrix does not match the fitted model terms."
                ) from exc

        predictions = np.asarray(model.predict(model_points), dtype=float).ravel()
        covariance = np.asarray(fitted.dispersion_matrix, dtype=float)
        model_values = model_points.to_numpy(dtype=float)
        leverages = np.einsum(
            "ij,jk,ik->i", model_values, covariance, model_values
        )
        tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(leverages)))) * 100
        if (leverages < -tolerance).any():
            raise ValueError("Computed confirmation leverage contains negative values.")
        leverages = np.maximum(leverages, 0.0)

        observed = np.asarray(
            [
                responses.loc[setting_values == value, response].mean()
                for value in unique_ids
            ],
            dtype=float,
        )
        critical = float(t.ppf(1.0 - alpha / 2.0, df_resid))
        half_width = critical * np.sqrt(
            ms_res * (1.0 / counts.astype(float) + leverages)
        )
        lower = predictions - half_width
        upper = predictions + half_width

        result = pd.DataFrame({"Setting": unique_ids.astype(int)})
        unique_actual = actual.iloc[positions].reset_index(drop=True)
        for name in self._factors:
            result[name] = unique_actual[name]
        result["n"] = counts
        result["Observed"] = observed
        result["Predicted"] = predictions
        result["Residual"] = observed - predictions
        result["PI Lower"] = lower
        result["PI Upper"] = upper
        result["Within PI"] = (observed >= lower) & (observed <= upper)
        return result.copy(deep=True)

    def clear_confirmation_runs(self) -> None:
        """Remove confirmation data without changing the fitted OLS model.

        Notes:
            This clears confirmation points, responses, metadata, and setting IDs.
            It does not change the original design or fitted model.
        """
        self._initialize_confirmation_runs()
