"""Tests for the explicit D-optimal augmentation workflow."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from doetools import CategoricalFactor, ContinuousFactor, MixtureFactor
from doetools.design.d_opt import DOptAddDesign
from doetools.utils.model_spec import ModelTerms


@pytest.fixture
def process_file(tmp_path):
    path = tmp_path / "existing.csv"
    pd.DataFrame(
        {
            "X1": [0.0, 10.0, 0.0, 10.0, 5.0],
            "X2": [0.0, 0.0, 10.0, 10.0, 5.0],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def response_file(tmp_path):
    path = tmp_path / "existing_responses.csv"
    pd.DataFrame(
        {
            "Exp. Idx": [2, 0, 1],
            "X1": [10.0, 0.0, 5.0],
            "X2": [0.0, 0.0, 10.0],
            "Yield": [30.0, 10.0, 20.0],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def mixture_file(tmp_path):
    path = tmp_path / "existing_mixture.csv"
    pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.0, 0.5],
            "B": [0.0, 1.0, 0.0, 0.5],
            "C": [0.0, 0.0, 1.0, 0.0],
        }
    ).to_csv(path, index=False)
    return path


def process_factors(
    lower: float = 0.0,
    upper: float = 10.0,
    n_levels: int = 3,
) -> dict[str, ContinuousFactor]:
    return {
        "X1": ContinuousFactor(n_levels, lower, upper, decimals=1),
        "X2": ContinuousFactor(n_levels, lower, upper, decimals=1),
    }


def mixture_factors() -> dict[str, MixtureFactor]:
    return {
        "A": MixtureFactor(0.0, 1.0),
        "B": MixtureFactor(0.0, 1.0),
        "C": MixtureFactor(0.0, 1.0),
    }


def linear_terms():
    return ModelTerms(
        intercept=True,
        pro_main="all",
        pro_int2=None,
        pro_int3=None,
        pro_quadratic=None,
    )


class TestImport:
    def test_import_uses_declared_factor_metadata_and_common_coding(
        self, process_file
    ):
        factors = process_factors(lower=-10, upper=20, n_levels=4)
        design = DOptAddDesign(factors, process_file)

        assert design._design_type == "D-Optimal Addition"
        assert design._factors["X1"] is factors["X1"]
        assert len(design._exp_done) == 5

        assert design.generate_cp(process_strategy="grid") is None
        coded_candidates = design._coded_cp
        candidates = design._cp
        historical_index = design._exp_done.index[
            (design._exp_done["X1"] == 10)
            & (design._exp_done["X2"] == 0)
        ][0]
        candidate_index = candidates.index[
            (candidates["X1"] == 10) & (candidates["X2"] == 0)
        ][0]
        np.testing.assert_allclose(
            design._coded_exp_done.loc[historical_index, ["X1", "X2"]],
            coded_candidates.loc[candidate_index, ["X1", "X2"]],
        )

    def test_import_responses_is_sorted_by_experiment_index(self, response_file):
        design = DOptAddDesign(
            process_factors(),
            response_file,
            responses=["Yield"],
        )
        assert design._responses["Yield"].tolist() == [10.0, 20.0, 30.0]
        assert design._exp_done["X1"].tolist() == [0.0, 5.0, 10.0]

    def test_public_signature_removes_inferred_and_coded_inputs(self):
        constructor = inspect.signature(DOptAddDesign)
        generate_cp = inspect.signature(DOptAddDesign.generate_cp)

        assert list(constructor.parameters)[:2] == ["factors", "source"]
        assert "file_path" not in constructor.parameters
        assert "factor_names" not in constructor.parameters
        assert "factor_types" not in constructor.parameters
        assert "coded" not in constructor.parameters
        assert "factor_bounds" not in generate_cp.parameters

    def test_import_accepts_dataframe_defensively(self):
        source = pd.DataFrame(
            {
                "X1": [0.0, 10.0, 5.0],
                "X2": [0.0, 10.0, 5.0],
                "Yield": [10.0, 20.0, 15.0],
            }
        )

        design = DOptAddDesign(
            factors=process_factors(),
            source=source,
            responses=["Yield"],
        )
        source.loc[0, "X1"] = 999.0
        source.loc[0, "Yield"] = 999.0

        assert design._exp_done.loc[0, "X1"] == 0.0
        assert design._responses.loc[0, "Yield"] == 10.0

    def test_import_validates_factor_mapping_and_columns(self, process_file):
        with pytest.raises(ValueError, match="non-empty mapping"):
            DOptAddDesign({}, process_file)
        with pytest.raises(TypeError, match="ContinuousFactor"):
            DOptAddDesign({"X1": object()}, process_file)
        with pytest.raises(ValueError, match="not found"):
            DOptAddDesign(
                {"missing": ContinuousFactor(2, 0, 1)}, process_file
            )

    def test_import_rejects_unknown_categorical_levels(self, tmp_path):
        source = tmp_path / "categorical.csv"
        pd.DataFrame({"Material": ["A", "C"]}).to_csv(source, index=False)
        with pytest.raises(ValueError, match="not declared"):
            DOptAddDesign(
                {"Material": CategoricalFactor(["A", "B"])}, source
            )

    def test_import_rejects_invalid_mixture_rows(self, tmp_path):
        source = tmp_path / "invalid_mixture.csv"
        pd.DataFrame({"A": [0.2], "B": [0.3], "C": [0.4]}).to_csv(
            source, index=False
        )
        with pytest.raises(ValueError, match="sum to 1"):
            DOptAddDesign(mixture_factors(), source)

    def test_expand_design_was_removed(self, process_file):
        design = DOptAddDesign(process_factors(), process_file)
        assert not hasattr(design, "expand_design")


class TestCandidateGeneration:
    def test_grid_candidates_use_declared_levels(self, process_file):
        factors = process_factors(lower=-10, upper=20, n_levels=4)
        design = DOptAddDesign(factors, process_file)
        assert design.generate_cp(process_strategy="grid") is None

        assert list(design._coded_cp) == ["X1", "X2"]
        assert design._cp["X1"].drop_duplicates().tolist() == [-10, 0, 10, 20]
        assert design._factors["X1"] is factors["X1"]

    def test_lhs_uses_explicit_lhs_n_samples(self, process_file):
        design = DOptAddDesign(process_factors(), process_file)
        assert design.generate_cp(process_strategy="lhs", lhs_n_samples=12) is None
        assert len(design._cp) == 12
        with pytest.raises(ValueError, match="only with 'lhs'"):
            design.generate_cp(process_strategy="grid", lhs_n_samples=12)

    def test_mixture_requires_an_explicit_source(self, mixture_file):
        design = DOptAddDesign(mixture_factors(), mixture_file)
        with pytest.raises(ValueError, match="Mixture factors require"):
            design.generate_cp()

    def test_mixture_all_and_lattice_only(self, mixture_file):
        design = DOptAddDesign(mixture_factors(), mixture_file)
        assert design.generate_cp(mixture_include="all") is None
        geometric = design._cp.copy()
        assert np.allclose(geometric[["A", "B", "C"]].sum(axis=1), 1)
        assert design.generate_cp(
            mixture_include=(), mixture_grid={"degree": 3}
        ) is None
        lattice = design._cp
        assert np.allclose(lattice[["A", "B", "C"]].sum(axis=1), 1)

    def test_filters_apply_to_candidates(self, process_file):
        design = DOptAddDesign(process_factors(), process_file)
        assert design.generate_cp(
            process_strategy="grid",
            filters=[lambda frame: frame["X1"] + frame["X2"] <= 10],
        ) is None
        assert (design._cp["X1"] + design._cp["X2"] <= 10).all()

    def test_existing_run_policy_controls_rows_outside_declared_bounds(
        self, process_file
    ):
        factors = process_factors(lower=0, upper=5)
        within = DOptAddDesign(factors, process_file)
        within.generate_cp(process_strategy="grid")
        assert len(within._filt_exp_done) == 2

        all_runs = DOptAddDesign(factors, process_file)
        all_runs.generate_cp(
            process_strategy="grid", existing_runs_policy="all"
        )
        assert len(all_runs._filt_exp_done) == 5
        assert (all_runs._filt_coded_exp_done[["X1", "X2"]] > 1).any().any()

    def test_historical_replicates_are_allowed_by_default(self, process_file):
        design = DOptAddDesign(process_factors(), process_file)
        design.generate_cp(process_strategy="grid")
        assert ((design._cp["X1"] == 0) & (design._cp["X2"] == 0)).any()

        design.generate_cp(
            process_strategy="grid", allow_existing_replicates=False
        )
        historical = set(map(tuple, design._exp_done[["X1", "X2"]].to_numpy()))
        assert not any(
            tuple(row) in historical
            for row in design._cp[["X1", "X2"]].to_numpy()
        )

    def test_mixed_candidates_use_declared_categorical_and_mixture_factors(
        self, tmp_path
    ):
        source = tmp_path / "mixed.csv"
        pd.DataFrame(
            {
                "Temperature": [10.0, 20.0],
                "Material": ["A", "B"],
                "M1": [0.2, 0.8],
                "M2": [0.8, 0.2],
            }
        ).to_csv(source, index=False)
        factors = {
            "Temperature": ContinuousFactor(2, 10, 20),
            "Material": CategoricalFactor(["A", "B"]),
            "M1": MixtureFactor(0.2, 0.8),
            "M2": MixtureFactor(0.2, 0.8),
        }
        design = DOptAddDesign(factors, source)
        design.generate_cp(
            process_strategy="grid", mixture_include="all"
        )

        assert set(design._cp["Material"]) == {"A", "B"}
        assert np.allclose(design._cp[["M1", "M2"]].sum(axis=1), 1)


class TestComputeAndSelect:
    def test_explicit_workflow_and_plot(self, process_file):
        design = DOptAddDesign(process_factors(), process_file)
        design.generate_cp(process_strategy="grid")
        design.set_model_terms(linear_terms())
        figure = design.compute_d_optimal(
            n_min=1, n_max=2, trials=8, random_state=7
        )

        assert list(design.log_det.index) == [1, 2]
        assert figure.layout.meta["existing_runs"] == 5
        assert list(figure.data[0].x) == [6, 7]

        design.select_design(2)
        assert len(design._design_matrix) == 7
        assert len(design._model_matrix) == 7

    def test_compute_is_reproducible_with_seed(self, process_file):
        scores = []
        selections = []
        for _ in range(2):
            design = DOptAddDesign(process_factors(), process_file)
            design.generate_cp(process_strategy="grid")
            design.set_model_terms(linear_terms())
            design.compute_d_optimal(2, 2, trials=5, random_state=123)
            scores.append(design.log_det.copy())
            selections.append(design._best_idx[2])
        pd.testing.assert_frame_equal(scores[0], scores[1])
        assert selections[0] == selections[1]

    def test_plot_candidate_set(self, process_file):
        design = DOptAddDesign(process_factors(), process_file)
        design.generate_cp(process_strategy="grid")
        figure = design.plot_candidate_set("X1", "X2")
        assert figure.layout.meta["geometry"] == "process_2d"

    def test_candidate_or_model_changes_invalidate_selected_design(
        self, process_file
    ):
        design = DOptAddDesign(process_factors(), process_file)
        design.generate_cp(process_strategy="grid")
        design.set_model_terms(linear_terms())
        design.compute_d_optimal(1, 1, trials=5, random_state=4)
        design.select_design(1)

        design.generate_cp(process_strategy="grid")
        assert design.log_det is None
        assert design._design_matrix is None
        with pytest.raises(KeyError, match="was computed"):
            design.select_design(1)

        design.compute_d_optimal(1, 1, trials=5, random_state=4)
        design.select_design(1)
        design.set_model_terms(linear_terms())
        assert design.log_det is None
        assert design._design_matrix is None

    def test_export_preserves_retained_responses(self, response_file, tmp_path):
        design = DOptAddDesign(
            process_factors(), response_file, responses=["Yield"]
        )
        design.generate_cp(process_strategy="grid")
        design.set_model_terms(linear_terms())
        design.compute_d_optimal(1, 1, trials=5, random_state=4)
        design.select_design(1)
        output = tmp_path / "augmentation.xlsx"
        design.export_experiments(
            responses=["Yield"], randomize=False, destination=output
        )
        exported = pd.read_excel(output)
        assert exported["Yield"].iloc[:3].tolist() == [10, 20, 30]
        assert exported["Yield"].iloc[3] == 0
