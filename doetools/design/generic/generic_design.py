"""Import an existing experimental design from a tabular file."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from ...graphs import GraphsMixin
from ...utils import (
    CategoricalFactor,
    ContinuousFactor,
    Design,
    FileUploaderMixin,
    MixtureFactor,
    ParetoMixin,
)


Factor = ContinuousFactor | CategoricalFactor | MixtureFactor


class ImportDesign(Design, GraphsMixin, FileUploaderMixin, ParetoMixin):
    """Import and validate an experimental design from a CSV or Excel file.

    Parameters
    ----------
    factors : mapping of str to factor
        Factor definitions keyed by the corresponding file-column names. The
        definitions provide the coding bounds for continuous factors and the
        declared order of categorical levels.
    file_path : str or pathlib.Path
        CSV or Excel file containing the experimental design.
    coded : bool, optional
        If True, factor columns contain coded values. If False, the columns
        contain values in experimental units. Default is False.

    Notes
    -----
    Continuous levels do not need to be equally spaced. After import, levels,
    coded_levels, and n_levels are synchronized with the distinct points
    observed in the file. The supplied lower and upper bounds continue to
    define the linear coding transformation, so axial points may legitimately
    have coded values outside [-1, 1].
    """

    def __init__(
        self,
        factors: Mapping[str, Factor],
        file_path: str | Path,
        *,
        coded: bool = False,
    ):
        super().__init__()

        self._validate_factor_mapping(factors)
        if not isinstance(coded, bool):
            raise TypeError("coded must be a boolean")

        frame = self.upload_file(file_path)
        factor_names = list(factors)
        missing_columns = [name for name in factor_names if name not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Factor columns not found in the file: {missing_columns}. "
                f"Available columns: {list(frame.columns)}"
            )

        self._factors = deepcopy(dict(factors))
        imported = frame.loc[:, factor_names].copy()
        self._validate_no_missing_values(imported)
        imported = self._prepare_imported_matrix(imported, coded=coded)

        if coded:
            self._coded_design_matrix = imported
            self._design_matrix = self._decode_matrix(imported)
        else:
            self._design_matrix = imported
            self._coded_design_matrix = self._code_matrix(imported)

        self._synchronize_observed_levels()
        self._validate_mixture_rows()
        self._design_type = "Generic"

    @staticmethod
    def _validate_factor_mapping(factors: Mapping[str, Factor]) -> None:
        if not isinstance(factors, Mapping):
            raise TypeError("factors must be a mapping from column names to factor objects")
        if not factors:
            raise ValueError("factors must contain at least one factor")

        supported = (ContinuousFactor, CategoricalFactor, MixtureFactor)
        for name, factor in factors.items():
            if not isinstance(name, str) or not name:
                raise TypeError("factor names must be non-empty strings")
            if not isinstance(factor, supported):
                raise TypeError(
                    f"Factor {name!r} must be a ContinuousFactor, "
                    "CategoricalFactor, or MixtureFactor"
                )

    @staticmethod
    def _validate_no_missing_values(matrix: pd.DataFrame) -> None:
        columns = matrix.columns[matrix.isna().any()].tolist()
        if columns:
            raise ValueError(f"Factor columns contain missing values: {columns}")

    def _prepare_imported_matrix(
        self, matrix: pd.DataFrame, *, coded: bool
    ) -> pd.DataFrame:
        prepared = matrix.copy()
        for name, factor in self._factors.items():
            if factor.type in {"cont", "mix"} or coded:
                try:
                    prepared[name] = pd.to_numeric(prepared[name], errors="raise")
                except (TypeError, ValueError) as exc:
                    coordinate_system = "coded" if coded else "actual"
                    raise ValueError(
                        f"Factor {name!r} must contain numeric {coordinate_system} values"
                    ) from exc

                values = prepared[name].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Factor {name!r} contains non-finite values")

            if factor.type == "cat":
                if coded:
                    prepared[name] = self._normalize_categorical_codes(
                        prepared[name], name, factor
                    )
                else:
                    unknown = prepared.loc[
                        ~prepared[name].isin(factor.levels), name
                    ].drop_duplicates().tolist()
                    if unknown:
                        raise ValueError(
                            f"Unknown levels for categorical factor {name!r}: {unknown}. "
                            f"Declared levels: {factor.levels}"
                        )

        return prepared

    @staticmethod
    def _normalize_categorical_codes(
        series: pd.Series,
        name: str,
        factor: CategoricalFactor,
    ) -> pd.Series:
        declared = np.asarray(factor.coded_levels, dtype=float)
        normalized: list[float] = []
        unknown: list[float] = []

        for value in series.to_numpy(dtype=float):
            matches = np.flatnonzero(np.isclose(value, declared, rtol=1e-9, atol=1e-12))
            if matches.size == 0:
                unknown.append(float(value))
            else:
                normalized.append(float(declared[matches[0]]))

        if unknown:
            unknown = list(dict.fromkeys(unknown))
            raise ValueError(
                f"Unknown coded levels for categorical factor {name!r}: {unknown}. "
                f"Declared codes: {declared.tolist()}"
            )

        return pd.Series(normalized, index=series.index, name=series.name)

    def _synchronize_observed_levels(self) -> None:
        for name, factor in self._factors.items():
            if factor.type == "cat":
                continue

            pairs = (
                pd.DataFrame(
                    {
                        "actual": self._design_matrix[name].to_numpy(),
                        "coded": self._coded_design_matrix[name].to_numpy(),
                    }
                )
                .drop_duplicates()
                .sort_values("actual")
            )
            factor.levels = pairs["actual"].to_numpy()

            if factor.type == "cont":
                factor.coded_levels = pairs["coded"].to_numpy(dtype=float)
                factor.n_levels = len(pairs)

    def _validate_mixture_rows(self) -> None:
        mixture_names = [
            name for name, factor in self._factors.items() if factor.type == "mix"
        ]
        if not mixture_names:
            return

        for name in mixture_names:
            factor = self._factors[name]
            values = self._design_matrix[name].to_numpy(dtype=float)
            outside = (values < factor.lower_bound) | (values > factor.upper_bound)
            if outside.any():
                invalid = np.unique(values[outside]).tolist()
                raise ValueError(
                    f"Mixture factor {name!r} contains values outside "
                    f"[{factor.lower_bound}, {factor.upper_bound}]: {invalid}"
                )

        totals = self._design_matrix[mixture_names].sum(axis=1).to_numpy(dtype=float)
        invalid_rows = np.flatnonzero(~np.isclose(totals, 1.0, rtol=1e-9, atol=1e-9))
        if invalid_rows.size:
            raise ValueError(
                "Mixture-factor values must sum to 1 in every row; invalid row "
                f"indices: {invalid_rows.tolist()}"
            )

    def build_design_matrix(self):
        """Return an error because an imported design cannot be regenerated."""
        return NotImplementedError("This design cannot be modified")
