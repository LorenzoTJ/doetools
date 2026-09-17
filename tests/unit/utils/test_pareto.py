"""Tests for the public Pareto-front workflow."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import doetools.utils as utils
from doetools import (
    CategoricalFactor,
    ContinuousFactor,
    DOptDesign,
    FractionalFactorialDesign,
    FullFactorialDesign,
    ImportDesign,
    MixtureFactor,
    ModelTerms,
    PlackettBurmanDesign,
    SimplexLatticeDesign,
)


def _fit_process_design(tmp_path):
    design = FullFactorialDesign(
        {
            "A": ContinuousFactor(
                n_levels=3, lower_bound=0, upper_bound=10, decimals=0
            ),
            "Mode": CategoricalFactor(levels=["Low", "High"]),
        }
    )
    design.set_model_terms(
        ModelTerms(pro_main="all", pro_int2="all", pro_quadratic=None)
    )
    destination = tmp_path / "pareto_process.xlsx"
    design.export_experiments(
        responses=["Yield", "Cost", "Purity"],
        randomize=False,
        destination=destination,
    )
    results = pd.read_excel(destination)
    high = (results["Mode"] == "High").astype(float)
    results["Yield"] = 2.0 * results["A"] + 4.0 * high
    results["Cost"] = results["A"] + 6.0 * high
    results["Purity"] = 20.0 - results["A"] + 3.0 * high
    design.import_responses(results)
    design.compute_mlr_model()
    design.set_response_conditions(
        lower_limits=[1000.0, None, None],
        upper_limits=[None, -1000.0, None],
        maximize=[True, False, True],
    )
    return design


def _fit_mixture_design(tmp_path):
    design = SimplexLatticeDesign(
        {
            "A": MixtureFactor(0.0, 1.0),
            "B": MixtureFactor(0.0, 1.0),
            "C": MixtureFactor(0.0, 1.0),
        },
        m=2,
    )
    design.set_model_terms(
        ModelTerms(
            intercept=False,
            pro_main=None,
            pro_int2=None,
            pro_quadratic=None,
            mix_main="all",
        )
    )
    destination = tmp_path / "pareto_mixture.xlsx"
    design.export_experiments(
        responses=["Strength", "Price"],
        randomize=False,
        destination=destination,
    )
    results = pd.read_excel(destination)
    results["Strength"] = 3 * results["A"] + results["B"] + 2 * results["C"]
    results["Price"] = results["A"] + 3 * results["B"] + 2 * results["C"]
    design.import_responses(results)
    design.compute_mlr_model()
    design.set_response_conditions(
        lower_limits=[None, None],
        upper_limits=[None, None],
        maximize=[True, False],
    )
    return design


def test_pareto_api_is_common_to_all_designs_and_mixin_is_not_public():
    for design_type in (
        FullFactorialDesign,
        FractionalFactorialDesign,
        PlackettBurmanDesign,
        SimplexLatticeDesign,
        ImportDesign,
        DOptDesign,
    ):
        assert hasattr(design_type, "compute_pareto_front")
        assert hasattr(design_type, "get_pareto_front")
        assert hasattr(design_type, "plot_pareto_front")
    assert not hasattr(utils, "ParetoMixin")


def test_getter_requires_computation_and_old_property_is_removed():
    design = FullFactorialDesign(
        {"A": ContinuousFactor(2, 0, 1), "B": ContinuousFactor(2, 0, 1)}
    )
    with pytest.raises(ValueError, match="No Pareto front computed"):
        design.get_pareto_front()
    assert not hasattr(design, "non_dominated_points")


def test_exact_continuous_and_categorical_level_overrides_preserve_order(tmp_path):
    design = _fit_process_design(tmp_path)
    design.compute_pareto_front(
        ["Yield", "Cost"],
        factor_levels={"A": [10, 0], "Mode": ["High"]},
    )

    candidates = design._pareto_candidates
    assert candidates["Candidate Id"].tolist() == ["P1", "P2"]
    assert candidates["A"].tolist() == [10.0, 0.0]
    assert candidates["Mode"].tolist() == ["High", "High"]


@pytest.mark.parametrize(
    ("factor_levels", "error", "message"),
    [
        ({"Unknown": [1]}, ValueError, "Unknown factor"),
        ({"A": []}, ValueError, "cannot be empty"),
        ({"A": [True]}, TypeError, "not boolean"),
        ({"A": [11]}, ValueError, "outside"),
        ({"A": [0, 0]}, ValueError, "unique"),
        ({"Mode": ["Other"]}, ValueError, "Unknown levels"),
        ({"Mode": [1]}, TypeError, "must be strings"),
    ],
)
def test_factor_level_validation(tmp_path, factor_levels, error, message):
    design = _fit_process_design(tmp_path)
    with pytest.raises(error, match=message):
        design.compute_pareto_front(factor_levels=factor_levels)


def test_domain_filters_exclude_candidates_but_response_limits_do_not(tmp_path):
    design = _fit_process_design(tmp_path)
    design.set_domain_filters([lambda frame: frame["A"] <= 5])
    design.compute_pareto_front(["Yield", "Cost"])

    candidates = design._pareto_candidates
    assert len(candidates) == 4
    assert candidates["A"].max() == 5
    assert (candidates["Yield"] < 1000).all()
    assert (candidates["Cost"] > -1000).all()


def test_invalid_domain_filter_result_is_rejected(tmp_path):
    design = _fit_process_design(tmp_path)
    design.set_domain_filters([lambda frame: np.ones(len(frame), dtype=bool)])
    with pytest.raises(TypeError, match="pandas Series"):
        design.compute_pareto_front()


def test_candidate_cap_is_checked_before_building_large_grid(tmp_path):
    design = _fit_process_design(tmp_path)
    with pytest.raises(ValueError, match="exceeds max_candidates=3"):
        design.compute_pareto_front(max_candidates=3)


def test_mixture_grid_uses_bounded_candidate_generator(tmp_path):
    design = _fit_mixture_design(tmp_path)
    design.compute_pareto_front(mixture_grid={"degree": 3})

    candidates = design._pareto_candidates
    assert len(candidates) == 10
    assert np.allclose(candidates[["A", "B", "C"]].sum(axis=1), 1.0)
    assert set(design.get_pareto_front().columns) == {
        "Candidate Id",
        "A",
        "B",
        "C",
        "Strength",
        "Price",
    }


def test_incremental_skyline_preserves_equal_objective_points():
    objectives = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 3.0],
        ]
    )
    mask = FullFactorialDesign._pareto_mask(objectives)
    assert mask.tolist() == [True, True, False, True]


def test_response_subset_and_mixed_directions(tmp_path):
    design = _fit_process_design(tmp_path)
    design.compute_pareto_front(["Yield", "Cost"])

    front = design.get_pareto_front()
    assert {"Yield", "Cost"}.issubset(front.columns)
    assert "Purity" not in front.columns
    assert tuple(design._pareto_objectives) == ("Yield", "Cost")


def test_getter_returns_a_defensive_copy(tmp_path):
    design = _fit_process_design(tmp_path)
    design.compute_pareto_front(["Yield", "Cost"])
    first = design.get_pareto_front()
    first.loc[0, "Yield"] = -999
    assert design.get_pareto_front().loc[0, "Yield"] != -999


def test_non_finite_predictions_are_reported_with_candidate_ids(tmp_path, monkeypatch):
    design = _fit_process_design(tmp_path)

    def invalid_predict(points, responses):
        result = pd.DataFrame(1.0, index=points.index, columns=responses)
        result.loc[result.index[0], responses[0]] = np.nan
        return result

    monkeypatch.setattr(design, "_predict", invalid_predict)
    with pytest.raises(ValueError, match=r"Yield: \['P1'\]"):
        design.compute_pareto_front(["Yield", "Cost"])


def test_state_changes_invalidate_cached_front(tmp_path):
    design = _fit_process_design(tmp_path)
    design.compute_pareto_front(["Yield", "Cost"])
    design.set_domain_filters([lambda frame: frame["A"] >= 0])
    with pytest.raises(ValueError, match="No Pareto front computed"):
        design.get_pareto_front()


def test_changing_only_response_plot_limits_keeps_cached_front(tmp_path):
    design = _fit_process_design(tmp_path)
    design.compute_pareto_front(["Yield", "Cost"])
    expected = design.get_pareto_front()

    design.set_response_conditions(
        lower_limits=[0.0, None, None],
        upper_limits=[None, 20.0, None],
        maximize=[True, False, True],
    )

    pd.testing.assert_frame_equal(design.get_pareto_front(), expected)


def test_plot_matches_library_style_and_can_hide_candidates(tmp_path):
    design = _fit_process_design(tmp_path)
    design.compute_pareto_front(["Yield", "Cost"])

    figure = design.plot_pareto_front()
    assert isinstance(figure, go.Figure)
    assert figure.layout.meta["plot_type"] == "pareto_front"
    assert figure.layout.paper_bgcolor == "white"
    assert figure.layout.plot_bgcolor == "white"
    assert figure.layout.font.family == "Arial"
    assert figure.layout.xaxis.mirror is True
    assert figure.layout.xaxis.gridcolor == "lightgray"
    assert {trace.name for trace in figure.data} == {
        "Domain-valid candidates",
        "Pareto front",
    }
    assert len(figure.layout.shapes) == 2

    front_only = design.plot_pareto_front(show_candidates=False)
    assert [trace.name for trace in front_only.data] == ["Pareto front"]


def test_three_dimensional_plot_uses_computed_objectives(tmp_path):
    design = _fit_process_design(tmp_path)
    design.compute_pareto_front()
    figure = design.plot_pareto_front(x="Yield", y="Cost", z="Purity")

    assert all(trace.type == "scatter3d" for trace in figure.data)
    assert figure.layout.scene.zaxis.title.text == "<b>Purity</b>"
    with pytest.raises(ValueError, match="computed objectives"):
        design.plot_pareto_front(x="Yield", y="Cost", z="Unknown")


def test_response_conditions_reject_false_limit_sentinel(tmp_path):
    design = _fit_process_design(tmp_path)
    with pytest.raises(TypeError, match="numeric or None"):
        design.set_response_conditions(
            lower_limits=[False, None, None],
            upper_limits=[None, None, None],
            maximize=[True, False, True],
        )
