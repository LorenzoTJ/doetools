"""Numerically stable D-optimal exchange shared by optimal designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class DOptimalSearchResult:
    best_indices: dict[int, list[int]]
    log_det: pd.DataFrame


@dataclass(frozen=True)
class _PreconditionedProblem:
    candidates: np.ndarray
    existing: np.ndarray
    rank_tolerance: float
    raw_logdet_correction: float


def _matrix_rank(matrix: np.ndarray, *, tolerance: float) -> int:
    """Return matrix rank using one absolute SVD tolerance."""
    if matrix.shape[0] == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular_values > tolerance))


def _model_logdet(
    matrix: np.ndarray,
    *,
    parameters: int,
    rank_tolerance: float,
) -> float | None:
    """Compute log(det(X'X)) directly from the singular values of X."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if (
        singular_values.size < parameters
        or singular_values[parameters - 1] <= rank_tolerance
    ):
        return None
    return float(2.0 * np.log(singular_values[:parameters]).sum())


def _precondition_problem(
    candidates: np.ndarray,
    existing: np.ndarray,
) -> _PreconditionedProblem:
    """Whiten a full-rank model space without changing D-optimal rankings."""
    if not np.isfinite(candidates).all():
        raise ValueError("candidate_points must contain only finite values")
    if not np.isfinite(existing).all():
        raise ValueError("existing_points must contain only finite values")

    combined = np.vstack((existing, candidates))
    # Normalize each model column before the SVD.  Without this step, LAPACK
    # may choose slightly different singular vectors for matrices that differ
    # only by nonsingular column scaling.  Those round-off differences can
    # change the winner between exactly equivalent exchange proposals.
    column_magnitudes = np.max(np.abs(combined), axis=0)
    if np.any(column_magnitudes == 0.0):
        raise ValueError(
            "The combined existing design and candidate set cannot support a "
            "full-rank model"
        )
    pivot_rows = np.argmax(np.abs(combined), axis=0)
    pivot_values = combined[pivot_rows, np.arange(combined.shape[1])]
    column_scales = np.copysign(column_magnitudes, pivot_values)
    normalized = combined / column_scales
    _, singular_values, right_vectors = np.linalg.svd(
        normalized,
        full_matrices=False,
    )
    parameters = candidates.shape[1]
    if singular_values.size == 0:
        raise ValueError("The model matrix must contain at least one column")

    raw_rank_tolerance = (
        max(combined.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > raw_rank_tolerance))
    if rank < parameters:
        raise ValueError(
            "The combined existing design and candidate set cannot support a "
            "full-rank model"
        )

    transform = right_vectors.T / singular_values
    working_candidates = (candidates / column_scales) @ transform
    working_existing = (existing / column_scales) @ transform
    working_rank_tolerance = max(combined.shape) * np.finfo(float).eps
    return _PreconditionedProblem(
        candidates=working_candidates,
        existing=working_existing,
        rank_tolerance=float(working_rank_tolerance),
        raw_logdet_correction=float(
            2.0
            * (
                np.log(np.abs(column_scales)).sum()
                + np.log(singular_values).sum()
            )
        ),
    )


def _full_rank_random_start(
    candidates: np.ndarray,
    existing: np.ndarray,
    *,
    count: int,
    parameters: int,
    rng: np.random.Generator,
    rank_tolerance: float,
) -> np.ndarray:
    """Build a randomized subset whose combined model matrix has full rank."""
    permutation = rng.permutation(candidates.shape[0])
    selected: list[int] = []
    if existing.shape[0] == 0:
        row_basis = np.empty((0, parameters), dtype=float)
    else:
        _, singular_values, right_vectors = np.linalg.svd(
            existing,
            full_matrices=False,
        )
        existing_rank = int(
            np.count_nonzero(singular_values > rank_tolerance)
        )
        row_basis = right_vectors[:existing_rank].copy()

    for index in permutation:
        if row_basis.shape[0] >= parameters:
            break
        residual = candidates[index].copy()
        if row_basis.shape[0]:
            # Reorthogonalize to limit loss of independence in narrow spaces.
            residual -= row_basis.T @ (row_basis @ residual)
            residual -= row_basis.T @ (row_basis @ residual)
        residual_norm = np.linalg.norm(residual)
        if residual_norm > rank_tolerance:
            selected.append(int(index))
            row_basis = np.vstack((row_basis, residual / residual_norm))

    if row_basis.shape[0] < parameters:
        raise RuntimeError(
            "Could not construct a full-rank initial D-optimal subset"
        )

    selected_set = set(selected)
    for index in permutation:
        if len(selected) >= count:
            break
        if int(index) not in selected_set:
            selected.append(int(index))
            selected_set.add(int(index))

    if len(selected) != count:
        raise RuntimeError(
            f"Could not construct an initial D-optimal subset with {count} rows"
        )
    result = np.asarray(selected, dtype=int)
    combined = np.vstack((existing, candidates[result]))
    if _matrix_rank(combined, tolerance=rank_tolerance) < parameters:
        raise RuntimeError(
            "Could not construct a numerically full-rank D-optimal subset"
        )
    return result


def optimize_d_optimal(
    candidate_points: pd.DataFrame | np.ndarray,
    *,
    n_min: int,
    n_max: int,
    existing_points: pd.DataFrame | np.ndarray | None = None,
    step: int = 1,
    trials: int = 50,
    max_no_improve: int = 5,
    random_state: int | np.random.Generator | None = None,
) -> DOptimalSearchResult:
    """Select D-optimal candidate subsets, optionally augmenting fixed rows.

    Model columns are SVD-whitened internally. Existing rows contribute a
    fixed information block and are never included in the exchangeable index
    set. Reported scores are transformed back to the original model basis.
    """
    candidates = np.asarray(candidate_points, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("candidate_points must be a non-empty two-dimensional matrix")
    if existing_points is None:
        existing = np.empty((0, candidates.shape[1]), dtype=float)
    else:
        existing = np.asarray(existing_points, dtype=float)
        if existing.ndim != 2 or existing.shape[1] != candidates.shape[1]:
            raise ValueError(
                "existing_points must have the same number of columns as candidate_points"
            )

    rows, parameters = candidates.shape
    if parameters == 0:
        raise ValueError("candidate_points must contain at least one model column")
    if not isinstance(n_min, int) or not isinstance(n_max, int):
        raise TypeError("n_min and n_max must be integers")
    if n_min < 1:
        raise ValueError("n_min must be at least 1")
    if n_min > n_max:
        raise ValueError("n_min must be smaller than or equal to n_max")
    if n_max > rows:
        raise ValueError("n_max cannot exceed the number of candidate points")
    problem = _precondition_problem(candidates, existing)
    working_candidates = problem.candidates
    working_existing = problem.existing
    existing_rank = _matrix_rank(
        working_existing,
        tolerance=problem.rank_tolerance,
    )
    if existing_rank + n_min < parameters:
        raise ValueError(
            "The existing design rank plus the number of additional runs must "
            "be at least the number of model coefficients"
        )
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not isinstance(max_no_improve, int) or max_no_improve <= 0:
        raise ValueError("max_no_improve must be a positive integer")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    existing_information = working_existing.T @ working_existing
    best_indices: dict[int, list[int]] = {}
    scores: dict[int, dict[str, float]] = {}

    for count in range(n_min, n_max + 1, step):
        best_logdet = -np.inf
        best_selection: np.ndarray | None = None

        for _ in range(trials):
            try:
                selection = _full_rank_random_start(
                    working_candidates,
                    working_existing,
                    count=count,
                    parameters=parameters,
                    rng=rng,
                    rank_tolerance=problem.rank_tolerance,
                )
            except RuntimeError:
                continue
            selected_mask = np.zeros(rows, dtype=bool)
            selected_mask[selection] = True
            while True:
                selected = working_candidates[selection]
                combined_selected = np.vstack((working_existing, selected))
                current_logdet = _model_logdet(
                    combined_selected,
                    parameters=parameters,
                    rank_tolerance=problem.rank_tolerance,
                )
                if current_logdet is None:
                    break

                if current_logdet > best_logdet:
                    best_logdet = current_logdet
                    best_selection = selection.copy()

                information = existing_information + selected.T @ selected
                try:
                    cholesky = np.linalg.cholesky(information)
                except np.linalg.LinAlgError:
                    break

                remaining = np.flatnonzero(~selected_mask)
                if remaining.size == 0:
                    break
                outside = working_candidates[remaining]
                try:
                    selected_solved = np.linalg.solve(
                        cholesky.T,
                        np.linalg.solve(cholesky, selected.T),
                    ).T
                    outside_solved = np.linalg.solve(
                        cholesky.T,
                        np.linalg.solve(cholesky, outside.T),
                    ).T
                except np.linalg.LinAlgError:
                    break
                h_selected = np.einsum("ij,ij->i", selected, selected_solved)
                h_outside = np.einsum("ij,ij->i", outside, outside_solved)
                cross = selected @ outside_solved.T
                ratios = (
                    np.outer(1.0 - h_selected, 1.0 + h_outside)
                    + cross**2
                )
                proposal_order = np.argsort(ratios, axis=None)[::-1]
                accepted = False
                rejected = 0
                for flat_position in proposal_order:
                    drop_position, add_position = np.unravel_index(
                        flat_position,
                        ratios.shape,
                    )
                    ratio = ratios[drop_position, add_position]
                    if not np.isfinite(ratio) or ratio <= 1.0 + 1e-10:
                        break

                    proposed_selection = selection.copy()
                    proposed_selection[drop_position] = remaining[add_position]
                    proposed_selected = working_candidates[proposed_selection]
                    proposed_combined = np.vstack(
                        (working_existing, proposed_selected)
                    )
                    proposed_logdet = _model_logdet(
                        proposed_combined,
                        parameters=parameters,
                        rank_tolerance=problem.rank_tolerance,
                    )
                    if (
                        proposed_logdet is not None
                        and proposed_logdet > current_logdet + np.log1p(1e-10)
                    ):
                        dropped = selection[drop_position]
                        added = remaining[add_position]
                        selection = proposed_selection
                        selected_mask[dropped] = False
                        selected_mask[added] = True
                        accepted = True
                        break

                    rejected += 1
                    if rejected >= max_no_improve:
                        break

                if not accepted:
                    break

        if best_selection is None:
            raise RuntimeError(
                f"Could not find a full-rank D-optimal subset for n={count}"
            )

        ordered = np.sort(best_selection).tolist()
        total_runs = existing.shape[0] + count
        raw_logdet = best_logdet + problem.raw_logdet_correction
        normalized = (
            raw_logdet - parameters * np.log(total_runs)
        ) / np.log(10.0)
        best_indices[count] = ordered
        scores[count] = {"log_M": round(float(normalized), 4)}

    return DOptimalSearchResult(
        best_indices=best_indices,
        log_det=pd.DataFrame.from_dict(scores, orient="index"),
    )


def build_d_optimal_solutions_figure(
    log_det: pd.DataFrame,
    *,
    existing_runs: int = 0,
) -> go.Figure:
    """Build the shared D-optimality curve."""
    additional = np.asarray(log_det.index, dtype=int)
    total = additional + existing_runs
    figure = go.Figure()
    hover = (
        "<b>%{x} total experiments</b><br>"
        "%{customdata} additional experiments<br>"
        "log(det) = %{y:.2f}<extra></extra>"
        if existing_runs
        else "<b>%{x} experiments</b><br>log(det) = %{y:.2f}<extra></extra>"
    )
    figure.add_trace(
        go.Scatter(
            x=total,
            y=log_det["log_M"],
            customdata=additional,
            mode="lines+markers",
            name="D-optimal solution",
            line=dict(color="#003153", width=2),
            marker=dict(size=9, color="#1971c2"),
            hovertemplate=hover,
        )
    )
    figure.add_annotation(
        text="<b>D-Optimal Solutions</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.09,
        showarrow=False,
        font=dict(size=20, family="Arial", color="black"),
    )
    figure.update_xaxes(
        title_text="<b>Number of Exp.</b>",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="#e9ecef",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="<b>log of Normalized Determinant</b>",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="#e9ecef",
        zeroline=False,
    )
    figure.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=60),
        width=700,
        height=500,
        showlegend=False,
        meta={
            "plot_type": "d_optimal_solutions",
            "existing_runs": existing_runs,
        },
    )
    return figure
