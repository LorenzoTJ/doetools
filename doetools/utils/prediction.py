"""Prediction workflow for externally supplied factor settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .external_points import _ExternalPointValidationMixin
from .intervals import Interval, VarianceSource, interval_half_width, validate_interval_options


class PredictionPointsMixin(_ExternalPointValidationMixin):
    """Load prediction points and evaluate fitted response models."""

    def _initialize_prediction_points(self) -> None:
        self._prediction_points: pd.DataFrame | None = None
        self._coded_prediction_points: pd.DataFrame | None = None

    def _require_prediction_points(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        actual = getattr(self, "_prediction_points", None)
        coded = getattr(self, "_coded_prediction_points", None)
        if not isinstance(actual, pd.DataFrame) or not isinstance(coded, pd.DataFrame):
            raise ValueError(
                "No prediction points available; call load_prediction_points first."
            )
        return actual, coded

    def _require_prediction_model(self):
        wrapper = getattr(self, "_mlr_wrapper", None)
        results = getattr(wrapper, "results", None)
        if not isinstance(results, dict) or not results:
            raise ValueError("No MLR model computed; call compute_mlr_model first.")
        return results

    def load_prediction_points(
        self,
        source: str | Path | pd.DataFrame,
        coded: bool = False,
    ) -> None:
        """Load and validate factor settings at which to make predictions.

        Args:
            source (str | Path | pd.DataFrame): DataFrame or CSV/XLSX file
                containing every factor column. Additional columns are ignored.
            coded (bool): Whether process and categorical factor values are coded.
                Mixture components are unaffected. Defaults to ``False``.

        Raises:
            FileNotFoundError: If a supplied path does not exist.
            TypeError: If ``coded`` is not Boolean.
            ValueError: If the file is empty, has duplicate or missing factor
                columns, or contains invalid factor settings.

        Notes:
            Row order, index, and duplicate factor settings are preserved. A
            failed load leaves any previously loaded points unchanged.
        """
        if not isinstance(coded, bool):
            raise TypeError("coded must be a boolean.")
        frame = self.upload_file(source)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("The prediction-points source must contain at least one row.")
        if frame.columns.has_duplicates:
            duplicates = list(frame.columns[frame.columns.duplicated()].unique())
            raise ValueError(
                f"Prediction-points source contains duplicate columns: {duplicates}."
            )

        factor_names = list(self._factors)
        missing_factors = [name for name in factor_names if name not in frame.columns]
        if missing_factors:
            raise ValueError(
                f"Factor columns not found in prediction source: {missing_factors}."
            )

        actual, coded_points = self._prepare_external_points(
            frame[factor_names].copy(),
            coded=coded,
            context="Prediction",
            row_label="points",
        )
        self._prediction_points = actual.copy(deep=True)
        self._coded_prediction_points = coded_points.copy(deep=True)

    def get_prediction_points(self, coded: bool = False) -> pd.DataFrame:
        """Return loaded prediction settings in actual or coded units.

        Args:
            coded (bool): Whether to return coded factor settings. Defaults to
                ``False``.

        Returns:
            pd.DataFrame: A defensive copy of the prediction factor settings.

        Raises:
            TypeError: If ``coded`` is not Boolean.
            ValueError: If prediction points have not been loaded.
        """
        if not isinstance(coded, bool):
            raise TypeError("coded must be a boolean.")
        actual, coded_points = self._require_prediction_points()
        points = coded_points if coded else actual
        return points.copy(deep=True)

    def get_prediction_results(
        self,
        response: str,
        alpha: float = 0.05,
        coded: bool = False,
        *,
        interval: Interval = "confidence",
        variance_source: VarianceSource = "residuals",
    ) -> pd.DataFrame:
        """Return predictions and pointwise confidence or prediction intervals.

        Args:
            response (str): Fitted response to predict.
            alpha (float): Pointwise significance level for the two-sided
                confidence interval. Must lie strictly between 0 and 1. Defaults
                to ``0.05``.
            coded (bool): Whether factor columns in the result use coded values.
                Defaults to ``False``.
            interval (str): "confidence" for the expected mean (default), or
                "prediction" for one new independent observation.
            variance_source (str): "residuals" (default) or "pure_error".
                Pure error requires valid replicated observations.

        Returns:
            pd.DataFrame: Factor settings followed by ``Predicted``, ``CI Lower``,
                and ``CI Upper`` columns for confidence intervals, or ``PI Lower``
                and ``PI Upper`` for prediction intervals, one row per loaded point.

        Raises:
            TypeError: If ``alpha`` is not numeric or ``coded`` is not Boolean.
            ValueError: If points or a fitted response model are unavailable, or if
                ``alpha`` is outside the open interval ``(0, 1)``.

        Notes:
            Half-widths are t * sqrt(MS * h) for the mean and
            t * sqrt(MS * (1 + h)) for one new observation. The selected
            variance and its degrees of freedom are used for both terms.
            These are two-sided pointwise intervals under the OLS assumptions
            of independent homoscedastic normal errors and an adequate model.
            Results are calculated from the current fitted model on every call.
            If residual degrees of freedom or the residual mean square are
            unavailable, point predictions are still returned and both selected
            interval columns contain ``NaN``. Unavailable pure error raises
            an explicit error instead of falling back to residual variance.
        """
        if not isinstance(coded, bool):
            raise TypeError("coded must be a boolean.")
        alpha = validate_interval_options(interval, variance_source, alpha)

        actual, coded_points = self._require_prediction_points()
        fitted_results = self._require_prediction_model()
        if response not in fitted_results:
            raise ValueError(
                f"Response {response!r} is not available for prediction. "
                f"Available responses: {list(fitted_results)}."
            )

        fitted = fitted_results[response]
        model = fitted.model

        model_points = self._build_model_matrix(coded_points, self._model_spec)
        expected_columns = list(model.params.index)
        if list(model_points.columns) != expected_columns:
            try:
                model_points = model_points[expected_columns]
            except KeyError as exc:
                raise ValueError(
                    "Prediction model matrix does not match the fitted model terms."
                ) from exc

        predictions = np.asarray(model.predict(model_points), dtype=float).ravel()
        settings = coded_points if coded else actual
        result = settings.copy(deep=True)
        result["Predicted"] = predictions

        half_width = interval_half_width(
            fitted, model_points, interval=interval,
            variance_source=variance_source, alpha=alpha,
        )
        prefix = "CI" if interval == "confidence" else "PI"
        result[f"{prefix} Lower"] = predictions - half_width
        result[f"{prefix} Upper"] = predictions + half_width
        return result.copy(deep=True)

    def clear_prediction_points(self) -> None:
        """Remove loaded prediction points without changing the fitted model."""
        self._initialize_prediction_points()
