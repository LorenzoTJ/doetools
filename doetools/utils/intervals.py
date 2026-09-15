"""Pointwise OLS intervals shared by response plots and external predictions."""

from numbers import Real
from typing import Literal

import numpy as np
from scipy.stats import t

Interval = Literal["confidence", "prediction"]
VarianceSource = Literal["residuals", "pure_error"]


def validate_interval_options(interval, variance_source, alpha, *, allow_none=False):
    """Validate the public interval options and return a numeric alpha."""
    if not (allow_none and interval is None) and interval not in ("confidence", "prediction"):
        raise ValueError("interval must be 'confidence' or 'prediction'" +
                         (", or None." if allow_none else "."))
    if variance_source not in ("residuals", "pure_error"):
        raise ValueError("variance_source must be 'residuals' or 'pure_error'.")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a numeric value between 0 and 1.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    return alpha


def interval_half_width(fitted, model_points, *, interval="confidence",
                        variance_source="residuals", alpha=0.05,
                        require_estimable=False):
    """Return t * sqrt(MS * (h + delta)), using the selected variance's df.

    delta is zero for a mean confidence interval and one for a new independent
    observation. Both terms use the same homoscedastic error variance estimate.
    Residual intervals that cannot be estimated return NaNs unless required by
    a plot. A requested but unavailable pure-error estimate always raises.
    """
    alpha = validate_interval_options(interval, variance_source, alpha)
    source = "res" if variance_source == "residuals" else "pe"
    variance = float(fitted.anova.get(f"MS_{source}", np.nan))
    dof = float(fitted.anova.get(f"df_{source}", np.nan))
    if not (np.isfinite(dof) and dof > 0 and np.isfinite(variance) and variance >= 0):
        if variance_source == "pure_error":
            raise ValueError("Pure-error intervals require valid replicates and positive pure-error degrees of freedom.")
        if require_estimable:
            raise ValueError("Residual intervals require positive residual degrees of freedom and a finite residual mean square.")
        return np.full(len(model_points), np.nan)

    expected_columns = list(fitted.model.params.index)
    try:
        values = model_points[expected_columns].to_numpy(dtype=float)
    except KeyError as exc:
        raise ValueError("Prediction model matrix does not match the fitted model terms.") from exc
    covariance = np.asarray(fitted.dispersion_matrix, dtype=float)
    leverages = np.einsum("ij,jk,ik->i", values, covariance, values)
    if not np.isfinite(leverages).all():
        raise ValueError("Computed prediction leverage must be finite.")
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(leverages), initial=0))) * 100
    if (leverages < -tolerance).any():
        raise ValueError("Computed prediction leverage contains negative values.")
    leverages = np.maximum(leverages, 0.0)
    return float(t.ppf(1 - alpha / 2, dof)) * np.sqrt(
        variance * (leverages + (1 if interval == "prediction" else 0))
    )
