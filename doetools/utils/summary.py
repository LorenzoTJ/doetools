"""Tabular summaries used by the public reporting API and PDF reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regression import RegressionAnalyzer
from .pdf_report import DesignReportPDF


class DesignSummaryMixin:
    """Provide validated, copy-safe summaries for a fitted design.

    The methods in this mixin are the public source of report data.  They never
    expose the design's internal DataFrames directly, so callers such as the PDF
    builder cannot accidentally modify the design state.
    """

    def _require_dataframe(self, attribute: str, description: str) -> pd.DataFrame:
        value = getattr(self, attribute, None)
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"No {description} available.")
        return value

    def _require_fitted_response(self, response: str):
        wrapper = getattr(self, "_mlr_wrapper", None)
        results = getattr(wrapper, "results", None)
        if not isinstance(results, dict):
            raise ValueError("No MLR model computed; call compute_mlr_model first.")
        if response not in results:
            raise ValueError(f"Response {response!r} is not available in the fitted model.")
        return results[response]

    @staticmethod
    def _factor_type(factor) -> str | None:
        return getattr(factor, "factor_type", getattr(factor, "type", None))

    def get_design_summary(self) -> pd.DataFrame:
        """Return high-level design counts, including mixture components."""
        coded = self._require_dataframe("_coded_design_matrix", "coded design matrix")
        factors_map = getattr(self, "_factors", None)
        if not isinstance(factors_map, dict):
            raise ValueError("No factors defined.")

        replicates = self._group_by_replicates()
        factors = list(factors_map.values())
        counts = {
            factor_type: sum(self._factor_type(factor) == factor_type for factor in factors)
            for factor_type in ("cont", "cat", "mix")
        }
        data = {
            "Design": [getattr(self, "_design_type", None)],
            "Runs": [coded.shape[0]],
            "Replicates": [sum(len(group) - 1 for group in replicates)],
            "Center Points": [self._number_of_center_points(coded)],
            "Continuous": [counts["cont"]],
            "Categorical": [counts["cat"]],
            "Mixture Components": [counts["mix"]],
        }
        result = pd.DataFrame(data)
        result.index = [""]
        return result.copy(deep=True)

    def get_factor_summary(self) -> pd.DataFrame:
        """Return type-aware factor information suitable for reports.

        Mixture components contain lower/upper bounds.  Their level-related
        columns are intentionally empty because those attributes are not part of
        the :class:`MixtureFactor` contract.
        """
        factors_map = getattr(self, "_factors", None)
        if not isinstance(factors_map, dict):
            raise ValueError("No factors defined.")

        rows = []
        for name, factor in factors_map.items():
            factor_type = self._factor_type(factor)
            levels = getattr(factor, "levels", None)
            coded_levels = getattr(factor, "coded_levels", None)
            if factor_type == "cont" and coded_levels is not None:
                coded_levels = np.round(coded_levels, decimals=getattr(factor, "decimals", 2))
            rows.append({
                "Factor": name,
                "Type": factor_type,
                "Levels Count": len(levels) if levels is not None else pd.NA,
                "Levels": levels if levels is not None else pd.NA,
                "Coded Levels": coded_levels if coded_levels is not None else pd.NA,
                "Reference Level": (
                    getattr(factor, "reference_level", pd.NA)
                    if factor_type == "cat"
                    else pd.NA
                ),
                "Lower Bound": getattr(factor, "lower_bound", pd.NA),
                "Upper Bound": getattr(factor, "upper_bound", pd.NA),
            })
        return pd.DataFrame(rows).copy(deep=True)

    def get_model_term_summary(self) -> pd.DataFrame:
        """Return the requested model specification, not matrix estimability."""
        spec = getattr(self, "_model_spec", None)
        if spec is None:
            raise ValueError("No model specification defined")
        int2 = [f"{a} : {b}" for a, b in (spec.interaction2 or [])]
        int3 = [f"{a} : {b} : {c}" for a, b, c in (spec.interaction3 or [])]
        quadratic = [f"{term}^2" for term in (spec.quadratic or [])]
        return pd.DataFrame({
            "Terms": ["Intercept", "Linear Terms", "2-Term Int.", "Quadratic Terms", "3-Term Int."],
            "Included": [spec.intercept, list(spec.main or []), int2, quadratic, int3],
        })

    def get_model_term_count(self) -> int:
        """Return the number of terms requested in the compiled specification."""
        spec = getattr(self, "_model_spec", None)
        if spec is None:
            raise ValueError("No model specification defined")
        return int(spec.model_terms)

    def get_response_condition_summary(self) -> pd.DataFrame:
        """Return response optimization limits and goals.

        A design without response conditions is not an error state, but this
        summary is unavailable until all response conditions are configured.
        """
        responses = self._require_dataframe("_responses", "responses")
        conditions = getattr(self, "_response_conditions", None)
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError("No response conditions defined.")
        missing = [name for name in responses.columns if name not in conditions]
        incomplete = [
            name for name in responses.columns
            if name in conditions and (
                not isinstance(conditions[name], dict)
                or not {"lower_limit", "upper_limit", "maximize"}.issubset(conditions[name])
            )
        ]
        if missing or incomplete:
            raise ValueError("Response conditions are incomplete for: " + ", ".join(missing + incomplete))

        return pd.DataFrame({
            "Response": list(responses.columns),
            "Lower Limit": [conditions[name]["lower_limit"] if conditions[name]["lower_limit"] is not None else "No"
                            for name in responses.columns],
            "Upper Limit": [conditions[name]["upper_limit"] if conditions[name]["upper_limit"] is not None else "No"
                            for name in responses.columns],
            "Goal": ["Maximize" if conditions[name]["maximize"] else "Minimize" for name in responses.columns],
        })

    def get_design_matrix(self) -> pd.DataFrame:
        """Return a defensive copy of the uncoded design matrix."""
        return self._require_dataframe("_design_matrix", "design matrix").copy(deep=True)

    def get_coded_design_matrix(self) -> pd.DataFrame:
        """Return a defensive copy of the coded design matrix."""
        return self._require_dataframe("_coded_design_matrix", "coded design matrix").copy(deep=True)

    def get_responses(self) -> pd.DataFrame:
        """Return a defensive copy of imported responses."""
        return self._require_dataframe("_responses", "responses").copy(deep=True)

    def get_predicted_responses(self) -> pd.DataFrame:
        """Return full-precision fitted predictions for every response."""
        responses = self._require_dataframe("_responses", "responses")
        names = list(responses.columns)
        predicted = pd.DataFrame(index=responses.index)
        for name in names:
            result = self._require_fitted_response(name)
            predicted[name] = pd.Series(result.y_hat, index=responses.index)
        return predicted.copy(deep=True)

    def get_leverages(self) -> pd.DataFrame:
        """Return leverage values for the configured model matrix."""
        self._require_dataframe("_model_matrix", "model matrix")
        coded = self._require_dataframe("_coded_design_matrix", "coded design matrix")
        return pd.DataFrame(self._compute_leverage(points=coded), columns=["Leverage"])

    def get_model_matrix(self) -> pd.DataFrame:
        """Return a defensive copy of the effective model matrix."""
        return self._require_dataframe("_model_matrix", "model matrix").copy(deep=True)

    def get_coefficient_summary(self, response: str) -> pd.DataFrame:
        """Return a defensive copy of a response coefficient table."""
        return self._require_fitted_response(response).coef.copy(deep=True)

    def get_anova_summary(self, response: str) -> pd.DataFrame:
        """Return ANOVA information, adding LOF rows only when available."""
        anova = self._require_fitted_response(response).anova
        required = ("SS_tot", "SS_reg", "SS_res", "df_tot", "df_reg", "df_res", "MS_tot", "MS_reg", "MS_res")
        missing = [key for key in required if key not in anova]
        if missing:
            raise ValueError("ANOVA summary is incomplete: " + ", ".join(missing))
        result = pd.DataFrame({
            "Source": ["Total", "Regression", "Residuals"],
            "SS": [anova["SS_tot"], anova["SS_reg"], anova["SS_res"]],
            "df": [anova["df_tot"], anova["df_reg"], anova["df_res"]],
            "MS": [anova["MS_tot"], anova["MS_reg"], anova["MS_res"]],
        })
        lof_keys = ("SS_pe", "SS_lof", "df_pe", "df_lof", "MS_pe", "MS_lof")
        if all(key in anova for key in lof_keys):
            result = pd.concat([result, pd.DataFrame({
                "Source": ["Pure Error", "Lack of Fit"],
                "SS": [anova["SS_pe"], anova["SS_lof"]],
                "df": [anova["df_pe"], anova["df_lof"]],
                "MS": [anova["MS_pe"], anova["MS_lof"]],
            })], ignore_index=True)
        return result.copy(deep=True)

    def get_vif(self) -> pd.DataFrame:
        """Return a defensive copy of model-matrix VIF diagnostics."""
        wrapper = getattr(self, "_mlr_wrapper", None)
        vif = getattr(wrapper, "vif", None)
        if not isinstance(vif, pd.DataFrame):
            raise ValueError("No MLR model computed; call compute_mlr_model first.")
        return vif.copy(deep=True)

    def get_dispersion_matrix(self, response: str, tol: float = 1e-8) -> pd.DataFrame:
        """Return a full-precision, copy-safe parameter dispersion matrix."""
        df = pd.DataFrame(self._require_fitted_response(response).dispersion_matrix).copy(deep=True)
        df = df.map(lambda value: 0 if np.isclose(value, 0, atol=tol) else value)
        df.index.name = None
        return df

    def get_replicate_summary(self, response: str) -> pd.DataFrame:
        """Return replicate statistics, or an empty table when not applicable."""
        replicates = self._require_fitted_response(response).replicates or {}
        columns = ["Run Index", "n*", "Mean", "Var", "Std Dev", "dof"]
        rows = [{
            "Run Index": index,
            "n*": values["n_replicates"],
            "Mean": values["mean"],
            "Var": values["var"],
            "Std Dev": values["std"],
            "dof": values["dof"],
        } for index, values in replicates.items()]
        return pd.DataFrame(rows, columns=columns)

    def get_metric_summary(self, response: str) -> pd.DataFrame:
        """Return a defensive one-row table of fitted model metrics."""
        metrics = self._require_fitted_response(response).metrics
        if not isinstance(metrics, dict):
            raise ValueError("Model metrics are unavailable.")
        return pd.DataFrame([metrics]).copy(deep=True)

    def get_model_f_test(self, response: str) -> pd.DataFrame:
        """Return the fitted-model F test for one response."""
        self._require_fitted_response(response)
        return RegressionAnalyzer.compute_model_f_test(response, self._mlr_wrapper).copy(deep=True)

    def get_lof_f_test(self, response: str) -> pd.DataFrame:
        """Return LOF F-test results, or an empty table when no LOF exists."""
        result = self._require_fitted_response(response)
        lof_keys = {"MS_lof", "MS_pe", "df_lof", "df_pe"}
        if not lof_keys.issubset(result.anova):
            return pd.DataFrame(columns=["F_value", "F_crit_95%", "F_crit_99%", "p_value"])
        if result.anova["df_lof"] <= 0 or result.anova["df_pe"] <= 0:
            return pd.DataFrame(columns=["F_value", "F_crit_95%", "F_crit_99%", "p_value"])
        return RegressionAnalyzer.compute_lof_f_test(response, self._mlr_wrapper).copy(deep=True)

    def get_model_summary_pdf(self, filename: str = "design_report.pdf") -> None:
        """Write a static PDF summary of the design and fitted response models.

        The report is assembled from the public summary getters. It includes
        design and factor information, model-matrix diagnostics, general
        diagnostics, and a separate analysis section for each fitted response.

        Args:
            filename (str): Destination PDF path. The filename must end in
                ``".pdf"`` and its parent directory must already exist.
                Defaults to ``"design_report.pdf"`` in the current directory.

        Returns:
            None: The report is written to ``filename``.

        Raises:
            ValueError: If the filename does not have a ``.pdf`` extension, the
                destination directory does not exist, or required design, model,
                response, or fitted-result data are unavailable.

        Notes:
            Optional analyses, such as lack-of-fit, replicate summaries, VIF,
            predictions, and response conditions, are replaced by explanatory
            notices when unavailable. The report is written to a temporary file
            in the destination directory and replaces an existing destination
            only after successful generation.
        """
        DesignReportPDF(self, title="Design and Model Summary Report").build(filename=filename)
