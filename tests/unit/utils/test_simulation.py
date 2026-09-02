import numpy as np
import pandas as pd
import pytest

from doetools import FullFactorialDesign, ModelTerms, simulate_responses
from doetools.utils.factors import ContinuousFactor


def _configured_design(replicates: int = 1):
    design = FullFactorialDesign(
        {"A": ContinuousFactor(2, 0, 1), "B": ContinuousFactor(2, 0, 1)},
        replicates=replicates,
    )
    design.set_model_terms(ModelTerms(pro_int2=None, pro_quadratic=None))
    return design


def test_simulate_responses_saves_reproducible_data_and_supports_diagnostics():
    design = _configured_design()

    actual = simulate_responses(design, ["Yield", "Purity"], random_state=17)
    expected = simulate_responses(_configured_design(), ["Yield", "Purity"], random_state=17)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(design.get_responses(), actual)
    assert design._response_list == ["Yield", "Purity"]

    design.compute_mlr_model()
    for response in actual:
        result = design._mlr_wrapper.results[response]
        assert result.anova["df_res"] > 0
        assert not np.allclose(result.residuals, 0)


def test_simulate_responses_uses_configured_names_and_clears_stale_fit():
    design = _configured_design()
    design._response_list = ["Existing"]
    design.compute_mlr_model = lambda: None
    design._mlr_wrapper = object()

    result = simulate_responses(design, random_state=3)

    assert result.columns.tolist() == ["Existing"]
    assert design._mlr_wrapper is None


def test_simulate_responses_accepts_saturated_models():
    design = FullFactorialDesign(
        {"A": ContinuousFactor(2, 0, 1), "B": ContinuousFactor(2, 0, 1)}
    )
    design.set_model_terms(ModelTerms(pro_int2="all", pro_quadratic=None))

    result = simulate_responses(design, "Yield", random_state=1)

    assert result.columns.tolist() == ["Yield"]
    assert len(result) == len(design._model_matrix)


@pytest.mark.parametrize("names", [[], ["Yield", "Yield"], [""]])
def test_simulate_responses_validates_response_names(names):
    with pytest.raises(ValueError, match="response_names"):
        simulate_responses(_configured_design(), names)
