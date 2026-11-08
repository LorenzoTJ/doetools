"""Shared export behavior for ordinary and augmented designs."""

import csv

import pandas as pd
import numpy as np
import pytest
from openpyxl import load_workbook

from doetools import ContinuousFactor, DOptAddDesign, FullFactorialDesign, ImportDesign, ModelTerms


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
    assert list(frame.columns) == ["Run", "Idx", "A", "Yield"]
    assert frame["Yield"].isna().all()
    assert not (tmp_path / "design_matrix.xlsx").exists()


@pytest.mark.parametrize("filename", ["experiments", "experiments.txt", "experiments.xls"])
def test_export_rejects_unsupported_destination(export_design, tmp_path, filename):
    path = tmp_path / filename
    with pytest.raises(ValueError, match=r"Supported formats: \.csv, \.xlsx"):
        export_design.export_experiments(destination=path)
    assert not path.exists()


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
@pytest.mark.parametrize("randomize", [False, True])
def test_one_based_identifiers_round_trip(export_design, tmp_path, extension, randomize):
    path = tmp_path / f"experiments.{extension}"
    export_design.export_experiments(responses=["Yield"], destination=path, randomize=randomize)
    frame = export_design.upload_file(path)
    expected_ids = list(range(1, len(frame) + 1))
    assert frame["Run"].tolist() == expected_ids
    assert sorted(frame["Idx"]) == expected_ids
    frame["Yield"] = frame["Idx"] * 2.5
    # Row position must never determine response matching.
    frame = frame.iloc[::-1].reset_index(drop=True)
    if extension == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False)
    export_design.import_responses(path)
    assert export_design._responses["Yield"].tolist() == [idx * 2.5 for idx in expected_ids]
    assert export_design._experimental_metadata["Idx"].tolist() == expected_ids
    pd.testing.assert_frame_equal(
        export_design._experimental_metadata,
        frame.sort_values("Idx")[["Run", "Idx"]].reset_index(drop=True),
    )


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("coded", [False, True])
def test_unmeasured_responses_are_empty_cells(export_design, tmp_path, extension, randomize, coded):
    path = tmp_path / f"empty_responses.{extension}"
    export_design.export_experiments(
        responses=["Yield", "Purity"], destination=path, randomize=randomize, coded=coded
    )
    if extension == "csv":
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(export_design._design_matrix)
        assert all(row["Yield"] == row["Purity"] == "" for row in rows)
    else:
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = list(workbook.active.values)
            assert len(rows) == len(export_design._design_matrix) + 1
            assert rows[0][-2:] == ("Yield", "Purity")
            assert all(row[-2:] == (None, None) for row in rows[1:])
        finally:
            workbook.close()
    assert export_design.upload_file(path)[["Yield", "Purity"]].isna().all().all()


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_incomplete_round_trip_blocks_fit_until_completed(export_design, tmp_path, extension):
    if not isinstance(export_design, DOptAddDesign):
        export_design.set_model_terms(ModelTerms(intercept=True, pro_main="all"))
    path = tmp_path / f"incomplete.{extension}"
    export_design.export_experiments(responses=["Yield"], destination=path, randomize=True)
    frame = export_design.upload_file(path)
    frame["Yield"] = frame["Idx"].map({1: 0.0, 2: np.nan, 3: 4.1})
    # A file may be reordered independently of Run and Idx.
    frame = frame.iloc[::-1].reset_index(drop=True)

    def save_and_import():
        if extension == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_excel(path, index=False)
        export_design.import_responses(path)

    save_and_import()
    np.testing.assert_allclose(export_design._responses["Yield"], [0.0, np.nan, 4.1], equal_nan=True)
    assert export_design._experimental_metadata["Idx"].tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="Response contains missing or non-finite values"):
        export_design.compute_mlr_model()
    frame.loc[frame["Idx"] == 2, "Yield"] = 2.0
    save_and_import()
    export_design.compute_mlr_model()
    np.testing.assert_allclose(export_design._responses["Yield"], [0.0, 2.0, 4.1])
    assert np.isfinite(export_design.get_fitted_values().to_numpy()).all()


@pytest.mark.parametrize("column", ["Run", "Idx"])
@pytest.mark.parametrize("invalid", [[0, 1, 2], [1, 1, 3], [1, 2, 4], [1, 2.5, 3],
                                    [1, None, 3], [1, np.inf, 3], [True, False, True]])
def test_import_rejects_invalid_identifiers_without_changing_state(export_design, column, invalid):
    export_design._response_list = ["Yield"]
    previous = export_design._responses
    frame = pd.DataFrame({"Run": [3, 1, 2], "Idx": [1, 2, 3], "Yield": [10, 20, 30]})
    frame[column] = invalid
    with pytest.raises(ValueError, match=column):
        export_design.import_responses(frame)
    assert export_design._responses is previous


def test_import_requires_idx_but_not_run(export_design):
    export_design._response_list = ["Yield"]
    frame = pd.DataFrame({"Idx": [3, 1, 2], "Yield": [30, 10, 20]})
    export_design.import_responses(frame)
    assert export_design._responses["Yield"].tolist() == [10, 20, 30]
    with pytest.raises(ValueError, match="Idx"):
        export_design.import_responses(frame.drop(columns="Idx"))


def test_identifiers_use_positions_not_pandas_index_labels(export_design, tmp_path):
    export_design._design_matrix.index = [10, 30, 50]
    path = tmp_path / "nonstandard_index.csv"
    export_design.export_experiments(destination=path, randomize=True)
    frame = pd.read_csv(path)
    assert frame["Run"].tolist() == [1, 2, 3]
    assert sorted(frame["Idx"]) == [1, 2, 3]
    np.testing.assert_allclose(
        frame.sort_values("Idx")["A"], export_design._design_matrix["A"]
    )


@pytest.mark.parametrize("column", ["Run", "Idx"])
@pytest.mark.parametrize("conflict", ["factor", "response"])
def test_export_rejects_reserved_names(export_design, tmp_path, column, conflict):
    if conflict == "factor":
        export_design._factors[column] = ContinuousFactor(2, 0, 1)
    path = tmp_path / "reserved.csv"
    with pytest.raises(ValueError, match="reserved metadata"):
        export_design.export_experiments(responses=[column] if conflict == "response" else None,
                                         destination=path)
    assert not path.exists()


@pytest.mark.parametrize("legacy", ["Exp. Order", "Exp. Idx"])
@pytest.mark.parametrize("workflow", ["responses", "import_design", "augmentation"])
def test_experiment_imports_reject_legacy_headers(export_design, legacy, workflow):
    frame = pd.DataFrame({"A": [0, 5, 10], "Idx": [1, 2, 3],
                          "Yield": [10, 20, 30], legacy: [0, 1, 2]})
    export_design._response_list = ["Yield"]
    actions = {
        "responses": lambda: export_design.import_responses(frame),
        "import_design": lambda: ImportDesign(export_design._factors, frame),
        "augmentation": lambda: DOptAddDesign(export_design._factors, frame),
    }
    with pytest.raises(ValueError, match="Rename.*add 1"):
        actions[workflow]()
