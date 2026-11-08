"""Shared export behavior for ordinary and augmented designs."""

import pandas as pd
import pytest

from doetools import ContinuousFactor, DOptAddDesign, FullFactorialDesign, ModelTerms


@pytest.fixture(params=["factorial", "augmentation"])
def export_design(request):
    factors = {"A": ContinuousFactor(3, 0, 10)}
    if request.param == "factorial":
        return FullFactorialDesign(factors)
    design = DOptAddDesign(factors, pd.DataFrame({"A": [0.0, 10.0]}))
    design.generate_cp(process_strategy="grid")
    design.set_model_terms(ModelTerms(intercept=True, pro_main="all"))
    design.compute_d_optimal(1, 1, trials=5, random_state=4)
    design.select_design(1)
    return design


def test_export_defaults_to_csv(export_design, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    export_design.export_experiments(responses=["Yield"], randomize=False)

    frame = pd.read_csv("design_matrix.csv")
    assert list(frame.columns) == ["Exp. Order", "Exp. Idx", "A", "Yield"]
    assert frame["Yield"].tolist() == [0] * len(frame)
    assert not (tmp_path / "design_matrix.xlsx").exists()


@pytest.mark.parametrize("filename", ["experiments", "experiments.txt", "experiments.xls"])
def test_export_rejects_unsupported_destination(export_design, tmp_path, filename):
    path = tmp_path / filename
    with pytest.raises(ValueError, match=r"Supported formats: \.csv, \.xlsx"):
        export_design.export_experiments(destination=path)
    assert not path.exists()
