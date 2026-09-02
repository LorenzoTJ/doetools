"""Focused tests for the shared D-optimal optimizer."""

import numpy as np
import pandas as pd
import pytest

from doetools import (
    CategoricalFactor,
    ContinuousFactor,
    DOptAddDesign,
    DOptDesign,
    MixtureFactor,
    ModelTerms,
)
from doetools.design.d_opt.optimizer import optimize_d_optimal


def test_each_random_start_is_full_rank():
    candidates = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    for seed in range(20):
        result = optimize_d_optimal(
            candidates,
            n_min=2,
            n_max=2,
            trials=1,
            random_state=seed,
        )
        selected = candidates[result.best_indices[2]]
        assert np.linalg.matrix_rank(selected) == 2


def test_random_start_uses_existing_design_rank():
    existing = np.array([[1.0, 0.0]])
    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = optimize_d_optimal(
        candidates,
        existing_points=existing,
        n_min=1,
        n_max=1,
        trials=1,
        random_state=3,
    )

    assert result.best_indices[1] == [1]


def test_infeasible_combined_rank_fails_before_search():
    candidates = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="cannot support a full-rank model"):
        optimize_d_optimal(
            candidates,
            n_min=2,
            n_max=2,
            trials=1,
            random_state=0,
        )


def test_preconditioning_is_invariant_to_nonsingular_column_scaling():
    levels = np.linspace(-1.0, 1.0, 9)
    candidates = np.column_stack((np.ones(9), levels, levels**2))
    scaling = np.diag([1e-6, 1e3, 1e3])

    original = optimize_d_optimal(
        candidates,
        n_min=3,
        n_max=5,
        trials=10,
        random_state=7,
    )
    scaled = optimize_d_optimal(
        candidates @ scaling,
        n_min=3,
        n_max=5,
        trials=10,
        random_state=7,
    )

    assert original.best_indices == scaled.best_indices
    pd.testing.assert_frame_equal(original.log_det, scaled.log_det)


def test_reported_score_uses_original_model_parameterization():
    levels = np.linspace(-1.0, 1.0, 9)
    candidates = np.column_stack((np.ones(9), levels, levels**2))
    result = optimize_d_optimal(
        candidates,
        n_min=4,
        n_max=4,
        trials=10,
        random_state=11,
    )

    selected = candidates[result.best_indices[4]]
    singular_values = np.linalg.svd(selected, compute_uv=False)
    raw_logdet = 2.0 * np.log(singular_values).sum()
    expected = (raw_logdet - selected.shape[1] * np.log(4)) / np.log(10.0)
    assert result.log_det.loc[4, "log_M"] == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_model_values_are_rejected(bad_value):
    candidates = np.array([[1.0, 0.0], [1.0, bad_value], [0.0, 1.0]])
    with pytest.raises(ValueError, match="only finite values"):
        optimize_d_optimal(candidates, n_min=2, n_max=2)


def test_constrained_pharmaceutical_mixture_is_numerically_estimable():
    factors = {
        "X1": MixtureFactor(0.098, 0.118),
        "X2": MixtureFactor(0.302, 0.322),
        "X3": MixtureFactor(0.14, 0.16),
        "X4": MixtureFactor(0.42, 0.44),
        "buffer": CategoricalFactor(levels=["A", "B"]),
    }
    design = DOptDesign(
        factors=factors,
        process_strategy="grid",
        mixture_include="all",
    )
    design.set_model_terms(
        ModelTerms(
            intercept=False,
            pro_main="all",
            pro_int2=None,
            pro_quadratic=None,
            mix_main="all",
            mix_int2="all",
            mix_int3="all",
        )
    )

    design.compute_d_optimal(
        n_min=15,
        n_max=25,
        trials=3,
        random_state=42,
    )

    assert list(design.log_det.index) == list(range(15, 26))
    for count, indices in design._best_idx.items():
        selected = design._cp_model_matrix.iloc[indices]
        assert len(indices) == count
        assert np.linalg.matrix_rank(selected) == selected.shape[1]


def test_augmentation_keeps_historical_experiments_immutable(tmp_path):
    source = tmp_path / "historical.csv"
    pd.DataFrame(
        {
            "X1": [0.0, 10.0, 0.0, 10.0],
            "X2": [0.0, 0.0, 10.0, 10.0],
            "Yield": [11.0, 12.0, 13.0, 14.0],
        }
    ).to_csv(source, index=False)
    design = DOptAddDesign(
        {
            "X1": ContinuousFactor(3, 0.0, 10.0),
            "X2": ContinuousFactor(3, 0.0, 10.0),
        },
        source,
        responses=["Yield"],
    )
    design.generate_cp(
        process_strategy="grid",
        existing_runs_policy="all",
        allow_existing_replicates=True,
    )
    design.set_model_terms(
        ModelTerms(
            intercept=True,
            pro_main="all",
            pro_int2=None,
            pro_int3=None,
            pro_quadratic=None,
        )
    )
    historical = design._exp_done.copy(deep=True)
    retained = design._filt_exp_done.copy(deep=True)
    retained_coded = design._filt_coded_exp_done.copy(deep=True)
    responses = design._responses.copy(deep=True)

    design.compute_d_optimal(2, 2, trials=5, random_state=19)

    pd.testing.assert_frame_equal(design._exp_done, historical)
    pd.testing.assert_frame_equal(design._filt_exp_done, retained)
    pd.testing.assert_frame_equal(design._filt_coded_exp_done, retained_coded)
    pd.testing.assert_frame_equal(design._responses, responses)
    assert all(0 <= index < len(design._cp) for index in design._best_idx[2])

    design.select_design(2)
    pd.testing.assert_frame_equal(
        design._design_matrix.iloc[: len(retained)].reset_index(drop=True),
        retained.reset_index(drop=True),
    )
    assert len(design._design_matrix) == len(retained) + 2
