import pytest
import pandas as pd
from doetools.utils.model_spec import ModelSpec, ModelTerms, compile_model_spec
from doetools.utils.model_spec import _validate_all_or_list, _expand_main, _expand_int
from doetools.utils.model_spec import _validate_vars_exist, _flatten_tuples


# Test Small Helpers

def test_validate_all_or_list():
    # Test both all, list, str != all
    x1 = "all"
    out_x1 = _validate_all_or_list(x1, "x1")
    x2 = ["a", "b", "c"]
    out_x2 = _validate_all_or_list(x2, "x2")
    x3 = "error"
    
    assert out_x1 is None
    assert out_x2 is None
    with pytest.raises(ValueError):
        _validate_all_or_list(x3, "x3")


def test_validate_vars_exist():
    # Test valid variables
    allowed = {"A", "B", "C"}
    _validate_vars_exist(["A", "B"], allowed, "process")
    
    # Test invalid variable
    with pytest.raises(ValueError, match="Variable 'D' not defined in the process factors"):
        _validate_vars_exist(["A", "D"], allowed, "process")
    
    # Test mixture factors
    with pytest.raises(ValueError, match="Variable 'X' not defined in the mixture factors"):
        _validate_vars_exist(["X"], {"Y", "Z"}, "mixture")


def test_flatten_tuples():
    # Test flattening list of tuples
    tuples = [("A", "B"), ("C", "D"), ("E", "F")]
    result = _flatten_tuples(tuples)
    assert result == ["A", "B", "C", "D", "E", "F"]
    
    # Test empty list
    assert _flatten_tuples([]) == []
    
    # Test 3-tuples
    tuples_3 = [("A", "B", "C"), ("D", "E", "F")]
    result_3 = _flatten_tuples(tuples_3)
    assert result_3 == ["A", "B", "C", "D", "E", "F"]


def test_expand_main():
    factors = ["A", "B", "C"]
    
    # Test "all"
    assert _expand_main(factors, "all") == ["A", "B", "C"]
    
    # Test explicit list
    assert _expand_main(factors, ["A", "C"]) == ["A", "C"]
    
    # Test None
    assert _expand_main(factors, None) == []
    
    # Test empty list
    assert _expand_main(factors, []) == []


def test_expand_int():
    factors = ["A", "B", "C"]
    
    # Test "all" with r=2
    result = _expand_int(factors, 2, "all")
    assert result == [("A", "B"), ("A", "C"), ("B", "C")]
    
    # Test "all" with r=3
    result = _expand_int(factors, 3, "all")
    assert result == [("A", "B", "C")]
    
    # Test explicit list
    result = _expand_int(factors, 2, [("A", "B")])
    assert result == [("A", "B")]
    
    # Test None
    assert _expand_int(factors, 2, None) == []
    
    # Test empty factors
    assert _expand_int([], 2, "all") == []


# Test ModelSpec Class

def test_model_spec_initialization():
    # Test basic initialization
    spec = ModelSpec()
    assert spec.intercept is True
    assert spec.main is None
    assert spec.interaction2 is None
    assert spec.quadratic is None
    assert spec.interaction3 is None
    assert spec.model_terms == 1  # Only intercept


def test_model_spec_term_counting():
    # Test with all term types
    spec = ModelSpec(
        intercept=True,
        main=["A", "B", "C"],
        interaction2=[("A", "B"), ("B", "C")],
        quadratic=["A", "B"],
        interaction3=[("A", "B", "C")]
    )
    assert spec.model_terms == 1 + 3 + 2 + 2 + 1  # 9 terms
    
    # Test without intercept
    spec = ModelSpec(
        intercept=False,
        main=["A", "B"],
        interaction2=[("A", "B")]
    )
    assert spec.model_terms == 0 + 2 + 1  # 3 terms


def test_model_spec_to_vertical_df():
    spec = ModelSpec(
        intercept=True,
        main=["A", "B"],
        interaction2=[("A", "B")],
        quadratic=["A"],
        interaction3=None
    )
    df = spec.to_vertical_df()
    
    # Check it's a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Check rows
    assert "Intercept" in df.index
    assert "Main Effects" in df.index
    assert "2-Term Interactions" in df.index
    assert "Quadratic" in df.index
    assert "3-Term Interactions" in df.index
    assert "Total Model Terms" in df.index
    
    # Check values
    assert df.loc["Intercept", "Value"] is True
    assert df.loc["Main Effects", "Value"] == ["A", "B"]
    assert df.loc["Total Model Terms", "Value"] == 5


# Test ModelTerms Class

def test_model_terms_initialization():
    # Test default values
    terms = ModelTerms()
    assert terms.intercept is True
    assert terms.pro_main == "all"
    assert terms.pro_int2 == "all"
    assert terms.pro_quadratic == "all"
    assert terms.pro_int3 is None
    assert terms.scheffe_pol_order is None
    assert terms.mix_main is None
    assert terms.mix_int2 is None
    assert terms.mix_int3 is None


def test_model_terms_custom_values():
    # Test custom initialization
    terms = ModelTerms(
        intercept=False,
        pro_main=["A", "B"],
        pro_int2=None,
        scheffe_pol_order=2
    )
    assert terms.intercept is False
    assert terms.pro_main == ["A", "B"]
    assert terms.pro_int2 is None
    assert terms.scheffe_pol_order == 2


# Test compile_model_spec Function

def test_compile_model_spec_basic_process():
    # Test basic process model
    req = ModelTerms(
        intercept=True,
        pro_main="all",
        pro_int2="all",
        pro_quadratic=None,
        pro_int3=None
    )
    spec = compile_model_spec(req, pro_factors=["A", "B"], mix_factors=[])
    
    assert spec.intercept is True
    assert spec.main == ["A", "B"]
    assert spec.interaction2 == [("A", "B")]
    assert spec.quadratic == []
    assert spec.interaction3 == []


def test_compile_model_spec_custom_process_terms():
    # Test with custom process terms
    req = ModelTerms(
        intercept=True,
        pro_main=["A"],
        pro_int2=[("A", "B")],
        pro_quadratic=["B"],
        pro_int3=None
    )
    spec = compile_model_spec(req, pro_factors=["A", "B", "C"], mix_factors=[])
    
    assert spec.main == ["A"]
    assert spec.interaction2 == [("A", "B")]
    assert spec.quadratic == ["B"]
    assert spec.interaction3 == []


def test_compile_model_spec_mixture_basic():
    # Test basic mixture model
    req = ModelTerms(
        intercept=False,
        pro_main=None,
        pro_int2=None,
        pro_quadratic=None,
        mix_main="all",
        mix_int2="all"
    )
    spec = compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y", "Z"])
    
    assert spec.intercept is False
    assert spec.main == ["X", "Y", "Z"]
    assert spec.interaction2 == [("X", "Y"), ("X", "Z"), ("Y", "Z")]


def test_compile_model_spec_scheffe_order_1():
    # Test Scheffé polynomial order 1
    req = ModelTerms(
        intercept=False,
        pro_main=None,
        scheffe_pol_order=1
    )
    spec = compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y"])
    
    assert spec.main == ["X", "Y"]
    assert spec.interaction2 == []
    assert spec.interaction3 == []


def test_compile_model_spec_scheffe_order_2():
    # Test Scheffé polynomial order 2
    req = ModelTerms(
        intercept=False,
        pro_main=None,
        scheffe_pol_order=2
    )
    spec = compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y", "Z"])
    
    assert spec.main == ["X", "Y", "Z"]
    assert spec.interaction2 == [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    assert spec.interaction3 == []


def test_compile_model_spec_scheffe_order_3():
    # Test Scheffé polynomial order 3
    req = ModelTerms(
        intercept=False,
        pro_main=None,
        scheffe_pol_order=3
    )
    spec = compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y", "Z"])
    
    assert spec.main == ["X", "Y", "Z"]
    assert spec.interaction2 == [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    assert spec.interaction3 == [("X", "Y", "Z")]


def test_compile_model_spec_combined():
    # Test combined process and mixture
    req = ModelTerms(
        intercept=False,
        pro_main="all",
        pro_int2=None,
        pro_quadratic="all",
        mix_main="all",
        mix_int2="all"
    )
    spec = compile_model_spec(
        req,
        pro_factors=["A", "B"],
        mix_factors=["X", "Y"]
    )
    
    assert spec.main == ["A", "B", "X", "Y"]
    assert spec.interaction2 == [("X", "Y")]
    assert spec.quadratic == ["A", "B"]


# Test Error Conditions

def test_compile_model_spec_error_intercept_with_mixture():
    # Intercept cannot be used with mixture models
    req = ModelTerms(intercept=True, mix_main="all")
    with pytest.raises(ValueError, match="Intercept cannot be included in a mixture model"):
        compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y"])


def test_compile_model_spec_error_scheffe_without_mixture():
    # Scheffé order requires mixture factors
    req = ModelTerms(scheffe_pol_order=2)
    with pytest.raises(ValueError, match="Scheffé polynomial order can be set only for mixture designs"):
        compile_model_spec(req, pro_factors=["A", "B"], mix_factors=[])


def test_compile_model_spec_error_no_mixture_terms():
    # Mixture factors but no terms specified
    req = ModelTerms(
        intercept=False,
        pro_main=None,
        mix_main=None,
        mix_int2=None,
        mix_int3=None
    )
    with pytest.raises(ValueError, match="No mixture model terms specified"):
        compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y"])


def test_compile_model_spec_error_invalid_scheffe_order():
    # Invalid Scheffé polynomial order
    req = ModelTerms(intercept=False, scheffe_pol_order=4)
    with pytest.raises(ValueError, match="Scheffé polynomial order must be 1, 2, or 3"):
        compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y"])
    
    req = ModelTerms(intercept=False, scheffe_pol_order=0)
    with pytest.raises(ValueError, match="Scheffé polynomial order must be 1, 2, or 3"):
        compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y"])


def test_compile_model_spec_error_invalid_pro_var():
    # Invalid process variable
    req = ModelTerms(pro_main=["A", "D"])
    with pytest.raises(ValueError, match="Variable 'D' not defined in the process factors"):
        compile_model_spec(req, pro_factors=["A", "B", "C"], mix_factors=[])


def test_compile_model_spec_error_invalid_mix_var():
    # Invalid mixture variable
    req = ModelTerms(intercept=False, mix_main=["X", "W"])
    with pytest.raises(ValueError, match="Variable 'W' not defined in the mixture factors"):
        compile_model_spec(req, pro_factors=[], mix_factors=["X", "Y", "Z"])


def test_compile_model_spec_error_invalid_string_input():
    # Invalid string input (not "all")
    req = ModelTerms(pro_main="some")
    with pytest.raises(ValueError, match="Invalid input for pro_main"):
        compile_model_spec(req, pro_factors=["A", "B"], mix_factors=[])


def test_compile_model_spec_none_values():
    # Test with None values
    req = ModelTerms(
        intercept=True,
        pro_main=None,
        pro_int2=None,
        pro_quadratic=None,
        pro_int3=None
    )
    spec = compile_model_spec(req, pro_factors=["A", "B"], mix_factors=[])
    
    assert spec.intercept is True
    assert spec.main == []
    assert spec.interaction2 == []
    assert spec.quadratic == []
    assert spec.interaction3 == []
    assert spec.model_terms == 1  # Only intercept
    