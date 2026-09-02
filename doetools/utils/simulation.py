"""Helpers for creating reproducible synthetic DoE responses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .abstract_design import Design


def simulate_responses(
    design: "Design",
    response_names: Sequence[str] | str | None = None,
    *,
    coefficient_scale: float = 10.0,
    noise_std: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate and attach synthetic responses for an already specified model.

    A different set of coefficients is drawn for each response, then normally
    distributed experimental error is added to the model prediction.  The
    result is stored in ``design._responses`` and can immediately be passed to
    :meth:`~doetools.utils.abstract_design.Design.compute_mlr_model`.

    Parameters
    ----------
    design
        A design whose model terms have already been set with
        ``set_model_terms``.
    response_names
        Name, or names, of the simulated responses.  When omitted, existing
        response names on the design are reused; otherwise ``"Response"`` is
        used.
    coefficient_scale
        Maximum absolute size of randomly drawn non-intercept coefficients.
        The intercept, if present, is drawn between five and fifteen times
        this value so example responses are conveniently positive.
    noise_std
        Standard deviation of independent Gaussian experimental error.  A
        positive value creates residual variation for regression diagnostics.
    random_state
        Seed or NumPy random generator used to make examples reproducible.

    Returns
    -------
    pandas.DataFrame
        The generated response table.  The returned table is a copy of the
        data saved on the design.

    Raises
    ------
    ValueError
        If model terms have not been set or an argument is invalid.

    Examples
    --------
    >>> design.set_model_terms(ModelTerms(pro_int2=None, pro_quadratic=None))
    >>> simulate_responses(design, ["Yield", "Purity"], random_state=42)
    >>> design.compute_mlr_model()
    """
    model_matrix = getattr(design, "_model_matrix", None)
    if not isinstance(model_matrix, pd.DataFrame) or model_matrix.empty:
        raise ValueError(
            "No model matrix defined; call set_model_terms before simulating responses."
        )

    try:
        values = model_matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("The model matrix must contain numeric values only.") from exc
    if not np.isfinite(values).all():
        raise ValueError("The model matrix contains missing or non-finite values.")

    names = _normalize_response_names(response_names, getattr(design, "_response_list", None))
    coefficient_scale = _validate_nonnegative_finite(coefficient_scale, "coefficient_scale")
    noise_std = _validate_nonnegative_finite(noise_std, "noise_std")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    coefficients = rng.uniform(-coefficient_scale, coefficient_scale, size=(len(model_matrix.columns), len(names)))
    if "Int" in model_matrix.columns:
        intercept = model_matrix.columns.get_loc("Int")
        coefficients[intercept] = rng.uniform(
            5 * coefficient_scale, 15 * coefficient_scale, size=len(names)
        )

    response_values = values @ coefficients
    response_values += rng.normal(0.0, noise_std, size=response_values.shape)
    responses = pd.DataFrame(response_values, columns=names, index=model_matrix.index)

    design._response_list = list(names)
    design._responses = responses.copy(deep=True)
    # A previous fit no longer represents the generated observations.
    design._mlr_wrapper = None
    return responses.copy(deep=True)


def _normalize_response_names(
    response_names: Sequence[str] | str | None,
    existing_names: Sequence[str] | None,
) -> list[str]:
    """Return validated names, preserving previously configured response names."""
    if response_names is None:
        names = list(existing_names) if existing_names else ["Response"]
    elif isinstance(response_names, str):
        names = [response_names]
    else:
        names = list(response_names)

    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("response_names must contain one or more non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("response_names must be unique.")
    return names


def _validate_nonnegative_finite(value: float, name: str) -> float:
    """Validate numeric simulation-scale parameters."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a non-negative finite number.")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return value
