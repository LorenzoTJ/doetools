"""Tests for importing externally generated experimental designs."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from doetools.design.generic import ImportDesign
from doetools.utils import CategoricalFactor, ContinuousFactor, MixtureFactor


@pytest.fixture
def actual_design_file(tmp_path):
    path = tmp_path / "design.csv"
    pd.DataFrame(
        {
            "Temperature": [20, 25, 40, 20],
            "Catalyst": ["B", "A", "C", "B"],
            "Response": [10.2, 11.0, 14.5, 10.8],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def process_factors():
    return {
        "Temperature": ContinuousFactor(
            n_levels=3,
            lower_bound=20,
            upper_bound=40,
            decimals=1,
        ),
        "Catalyst": CategoricalFactor(
            levels=["C", "A", "B"],
            reference_level="A",
        ),
    }


class TestImportDesign:
    def test_imports_factor_columns_in_mapping_order(
        self, actual_design_file, process_factors
    ):
        factors = OrderedDict(
            [
                ("Catalyst", process_factors["Catalyst"]),
                ("Temperature", process_factors["Temperature"]),
            ]
        )

        design = ImportDesign(factors=factors, file_path=actual_design_file)

        assert design._design_type == "Generic"
        assert list(design._design_matrix) == ["Catalyst", "Temperature"]
        assert "Response" not in design._design_matrix

    def test_non_equispaced_continuous_levels_are_preserved(
        self, actual_design_file, process_factors
    ):
        design = ImportDesign(factors=process_factors, file_path=actual_design_file)
        factor = design._factors["Temperature"]

        np.testing.assert_array_equal(factor.levels, [20, 25, 40])
        np.testing.assert_allclose(factor.coded_levels, [-1.0, -0.5, 1.0])
        assert factor.n_levels == 3
        np.testing.assert_allclose(
            design._coded_design_matrix["Temperature"],
            [-1.0, -0.5, 1.0, -1.0],
        )

    def test_categorical_order_and_reference_are_preserved(
        self, actual_design_file, process_factors
    ):
        design = ImportDesign(factors=process_factors, file_path=actual_design_file)
        factor = design._factors["Catalyst"]

        assert factor.levels == ["C", "A", "B"]
        assert factor.reference_level == "A"
        np.testing.assert_allclose(factor.coded_levels, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            design._coded_design_matrix["Catalyst"],
            [1.0, 0.0, -1.0, 1.0],
        )

    def test_supplied_factor_objects_are_not_mutated(
        self, actual_design_file, process_factors
    ):
        original = process_factors["Temperature"]
        np.testing.assert_array_equal(original.levels, [20, 30, 40])

        design = ImportDesign(factors=process_factors, file_path=actual_design_file)

        assert design._factors["Temperature"] is not original
        np.testing.assert_array_equal(original.levels, [20, 30, 40])
        np.testing.assert_array_equal(
            design._factors["Temperature"].levels, [20, 25, 40]
        )

    def test_decodes_a_coded_design(self, tmp_path):
        path = tmp_path / "coded.csv"
        pd.DataFrame(
            {
                "Temperature": [-1.0, -0.5, 1.0],
                "Catalyst": [-1.0, 0.0, 1.0],
            }
        ).to_csv(path, index=False)
        factors = {
            "Temperature": ContinuousFactor(3, 20, 40, decimals=1),
            "Catalyst": CategoricalFactor(["C", "A", "B"], reference_level="A"),
        }

        design = ImportDesign(factors=factors, file_path=path, coded=True)

        np.testing.assert_allclose(
            design._design_matrix["Temperature"], [20.0, 25.0, 40.0]
        )
        assert design._design_matrix["Catalyst"].tolist() == ["C", "A", "B"]
        np.testing.assert_allclose(
            design._factors["Temperature"].coded_levels, [-1.0, -0.5, 1.0]
        )

    def test_preserves_axial_codes_outside_unit_interval(self, tmp_path):
        path = tmp_path / "ccd.csv"
        pd.DataFrame({"Temperature": [15.86, 20.0, 30.0, 40.0, 44.14]}).to_csv(
            path, index=False
        )
        factors = {"Temperature": ContinuousFactor(3, 20, 40, decimals=2)}

        design = ImportDesign(factors=factors, file_path=path)

        np.testing.assert_allclose(
            design._factors["Temperature"].coded_levels,
            [-1.414, -1.0, 0.0, 1.0, 1.414],
        )

    def test_accepts_excel_files(self, tmp_path):
        path = tmp_path / "design.xlsx"
        pd.DataFrame({"Temperature": [20, 25, 40]}).to_excel(path, index=False)
        factors = {"Temperature": ContinuousFactor(3, 20, 40, decimals=1)}

        design = ImportDesign(factors=factors, file_path=path)

        assert len(design._design_matrix) == 3

    @pytest.mark.parametrize(
        ("factors", "exception", "message"),
        [
            ([], TypeError, "factors must be a mapping"),
            ({}, ValueError, "at least one factor"),
            ({"Temperature": object()}, TypeError, "must be a ContinuousFactor"),
            ({1: ContinuousFactor(2, 0, 1)}, TypeError, "non-empty strings"),
        ],
    )
    def test_validates_factor_mapping(
        self, actual_design_file, factors, exception, message
    ):
        with pytest.raises(exception, match=message):
            ImportDesign(factors=factors, file_path=actual_design_file)

    def test_rejects_missing_factor_columns(self, actual_design_file):
        factors = {"Pressure": ContinuousFactor(2, 1, 2)}

        with pytest.raises(ValueError, match="Factor columns not found"):
            ImportDesign(factors=factors, file_path=actual_design_file)

    def test_rejects_missing_factor_values(self, tmp_path):
        path = tmp_path / "missing.csv"
        pd.DataFrame({"Temperature": [20.0, np.nan, 40.0]}).to_csv(path, index=False)
        factors = {"Temperature": ContinuousFactor(3, 20, 40)}

        with pytest.raises(ValueError, match="contain missing values"):
            ImportDesign(factors=factors, file_path=path)

    def test_rejects_non_numeric_continuous_values(self, tmp_path):
        path = tmp_path / "invalid.csv"
        pd.DataFrame({"Temperature": [20, "hot", 40]}).to_csv(path, index=False)
        factors = {"Temperature": ContinuousFactor(3, 20, 40)}

        with pytest.raises(ValueError, match="must contain numeric actual values"):
            ImportDesign(factors=factors, file_path=path)

    def test_rejects_unknown_actual_category(self, tmp_path):
        path = tmp_path / "invalid_category.csv"
        pd.DataFrame({"Catalyst": ["A", "C"]}).to_csv(path, index=False)
        factors = {"Catalyst": CategoricalFactor(["A", "B"])}

        with pytest.raises(ValueError, match="Unknown levels"):
            ImportDesign(factors=factors, file_path=path)

    def test_rejects_unknown_coded_category(self, tmp_path):
        path = tmp_path / "invalid_code.csv"
        pd.DataFrame({"Catalyst": [-1.0, 0.25]}).to_csv(path, index=False)
        factors = {"Catalyst": CategoricalFactor(["A", "B"])}

        with pytest.raises(ValueError, match="Unknown coded levels"):
            ImportDesign(factors=factors, file_path=path, coded=True)

    def test_imports_mixture_factors(self, tmp_path):
        path = tmp_path / "mixture.csv"
        pd.DataFrame(
            {
                "A": [0.2, 0.4],
                "B": [0.3, 0.2],
                "C": [0.5, 0.4],
            }
        ).to_csv(path, index=False)
        factors = {
            name: MixtureFactor(lower_bound=0.0, upper_bound=1.0)
            for name in ["A", "B", "C"]
        }

        design = ImportDesign(factors=factors, file_path=path)

        assert design._coded_design_matrix.equals(design._design_matrix)
        np.testing.assert_allclose(design._factors["A"].levels, [0.2, 0.4])

    def test_rejects_invalid_mixture_sum(self, tmp_path):
        path = tmp_path / "invalid_mixture.csv"
        pd.DataFrame({"A": [0.2], "B": [0.3], "C": [0.4]}).to_csv(
            path, index=False
        )
        factors = {
            name: MixtureFactor(lower_bound=0.0, upper_bound=1.0)
            for name in ["A", "B", "C"]
        }

        with pytest.raises(ValueError, match="must sum to 1"):
            ImportDesign(factors=factors, file_path=path)

    def test_build_design_matrix_is_not_available(
        self, actual_design_file, process_factors
    ):
        design = ImportDesign(factors=process_factors, file_path=actual_design_file)

        result = design.build_design_matrix()

        assert isinstance(result, NotImplementedError)
