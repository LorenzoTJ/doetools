"""Multiple Linear Regression Analysis for Design of Experiments.

This module provides tools for fitting multiple linear regression models with
Design of Experiments (DoE) specific features including:
- ANOVA decomposition with sum of squares
- Lack-of-fit testing for replicated designs
- Leave-one-out cross-validation (LOOCV)
- Variance inflation factor (VIF) for collinearity detection
- F-tests for model significance and lack-of-fit

The implementation uses statsmodels for OLS fitting and provides comprehensive
diagnostics suitable for experimental design analysis.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class RegressionSummary:
    """
    Comprehensive summary of a single-response regression model.

    Contains all fitted model information, predictions, residuals, and
    diagnostic statistics for one response variable.

    Attributes
    ----------
    model : RegressionResultsWrapper
        Fitted statsmodels OLS regression model.
    coef : pd.DataFrame
        Coefficient table with columns: Variable, Coefficient, Std_Error,
        Conf_Int, Lower_CI, Upper_CI, p_value.
    dispersion_matrix : np.ndarray or pd.DataFrame
        Normalized covariance matrix of parameter estimates.
    y_hat : np.ndarray or pd.Series
        Predicted values from the fitted model.
    residuals : np.ndarray or pd.Series
        Residuals (y - y_hat).
    y_hat_cv : np.ndarray
        Leave-one-out cross-validation predictions.
    residuals_cv : np.ndarray or pd.Series
        Cross-validation residuals (y - y_hat_cv).
    metrics : dict[str, float]
        Model quality metrics with keys: R2, R2_adj, Q2, PRESS, RMSE,
        RMSE_CV.
    anova : dict[str, float]
        ANOVA table with keys: SS_tot, df_tot, MS_tot, SS_reg, df_reg, MS_reg,
        SS_res, df_res, MS_res. If replicates exist, also includes LOF keys.
    lof : dict[str, float], optional
        Lack-of-fit statistics with keys: SS_pe, df_pe, MS_pe, SS_lof,
        df_lof, MS_lof. Only present when replicate_groups are provided.
    replicates : dict, optional
        Summary statistics for each replicate group. Keys are tuples of
        observation indices, values are dicts with: n_replicates, mean,
        ss_pe, dof, var, std.

    Notes
    -----
    Q² is computed using leave-one-out cross-validation and may be negative.
    Its interpretation depends on the response, design, and intended use; no
    universal acceptance threshold is imposed here.

    Lack-of-fit testing requires replicated observations at identical factor
    settings to separate pure error from model inadequacy.
    """
    model: RegressionResultsWrapper
    coef: pd.DataFrame
    dispersion_matrix: np.ndarray

    y_hat: np.ndarray
    residuals: np.ndarray
    y_hat_cv: np.ndarray
    residuals_cv: np.ndarray

    metrics: dict[str, float]
    anova: dict[str, float]
    lof: Optional[dict[str, float]] = None
    replicates: Optional[dict] = None


@dataclass
class RegressionWrapper:
    """Container for multi-response regression results.

    Wraps regression results for multiple response variables along with
    shared diagnostics like variance inflation factors.

    Attributes
    ----------
    results : dict[str, RegressionSummary]
        Dictionary mapping response variable names to their RegressionSummary
        objects.
    vif : pd.DataFrame
        Variance inflation factors for predictor variables with columns:
        Variable, VIF. Intercept is excluded.

    Notes
    -----
    VIF values > 10 indicate problematic multicollinearity among predictors.
    Values > 5 suggest moderate collinearity that may warrant investigation.
    """
    results: dict[str, RegressionSummary]
    vif: pd.DataFrame


class RegressionAnalyzer:
    """Analyzer for multiple linear regression with DoE diagnostics.

    Provides comprehensive regression analysis including ANOVA, lack-of-fit
    testing, cross-validation, and variance inflation factors. Designed for
    Design of Experiments applications where model adequacy and prediction
    quality are critical.

    """

    def __init__(self):
        """Initialize a stateless regression analyzer."""

    @staticmethod
    def _validate_inputs(
        X: pd.DataFrame,
        y: pd.Series,
        replicate_groups: list[list[int]],
    ) -> tuple[pd.DataFrame, pd.Series, list[list[int]]]:
        """Validate and normalize a single-response regression input."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        if X.empty or X.shape[1] == 0:
            raise ValueError("The model matrix must contain runs and model terms.")
        if y.empty:
            raise ValueError("The response must contain observations.")
        if len(X) != len(y):
            raise ValueError("Model matrix and response must contain the same number of observations.")
        if not X.index.equals(y.index):
            raise ValueError("Model matrix and response indices must be aligned.")
        if len(X) < 2:
            raise ValueError("At least two observations are required for regression analysis.")
        if X.columns.has_duplicates:
            duplicates = list(X.columns[X.columns.duplicated()].unique())
            raise ValueError(f"Model matrix contains duplicate term names: {duplicates}.")

        try:
            numeric_X = X.apply(pd.to_numeric, errors="raise").astype(float)
            numeric_y = pd.to_numeric(y, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Model matrix and response must contain numeric values only.") from exc
        if not np.isfinite(numeric_X.to_numpy()).all():
            raise ValueError("Model matrix contains missing or non-finite values.")
        if not np.isfinite(numeric_y.to_numpy()).all():
            raise ValueError("Response contains missing or non-finite values.")

        if replicate_groups is None:
            groups: list[list[int]] = []
        elif not isinstance(replicate_groups, list):
            raise TypeError("replicate_groups must be a list of index lists.")
        else:
            groups = []
            used: set[int] = set()
            for group in replicate_groups:
                if not isinstance(group, (list, tuple)):
                    raise TypeError("Each replicate group must be a list of positional indices.")
                if len(group) < 2:
                    raise ValueError("Each replicate group must contain at least two observations.")
                normalized: list[int] = []
                for index in group:
                    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                        raise TypeError("Replicate indices must be integers.")
                    position = int(index)
                    if position < 0 or position >= len(X):
                        raise ValueError(f"Replicate index {position} is outside the response range.")
                    if position in used:
                        raise ValueError(f"Replicate index {position} occurs in more than one group.")
                    used.add(position)
                    normalized.append(position)
                reference_row = numeric_X.iloc[normalized[0]].to_numpy()
                grouped_rows = numeric_X.iloc[normalized].to_numpy()
                if not np.all(grouped_rows == reference_row):
                    raise ValueError(
                        "Replicate groups must contain runs with identical model-matrix rows."
                    )
                groups.append(normalized)
        return numeric_X, numeric_y, groups

    @staticmethod
    def _loocv_predictions(X: pd.DataFrame, y: pd.Series, full_rank: int) -> np.ndarray:
        """Return exact OLS leave-one-out predictions for the supplied matrix.

        A fold is deliberately left undefined when removing the observation
        reduces the rank of the effective model. In that case the omitted
        response is not estimable under the same parameter space as the full
        fit (the common saturated-model case).
        """
        predictions = np.full(len(y), np.nan, dtype=float)
        positions = np.arange(len(y))
        for omitted in positions:
            training = positions != omitted
            X_train = X.iloc[training]
            if np.linalg.matrix_rank(X_train.to_numpy()) < full_rank:
                continue
            fold = sm.OLS(y.iloc[training], X_train).fit()
            prediction = np.asarray(fold.predict(X.iloc[[omitted]]), dtype=float)
            if prediction.size == 1 and np.isfinite(prediction[0]):
                predictions[omitted] = prediction[0]
        return predictions

    def _coef_table(self, model: RegressionResultsWrapper) -> pd.DataFrame:
        """Extract coefficient table from fitted statsmodels OLS model.

        Parameters
        ----------
        model : RegressionResultsWrapper
            Fitted statsmodels OLS regression model.

        Returns
        -------
        pd.DataFrame
            Coefficient table with columns:
            - Variable: predictor variable name
            - Coefficient: estimated coefficient value
            - Std_Error: standard error of coefficient
            - Conf_Int: half-width of 95% confidence interval
            - Lower_CI: lower bound of 95% confidence interval
            - Upper_CI: upper bound of 95% confidence interval
            - p_value: p-value for coefficient significance test

        Notes
        -----
        Values retain full precision. Presentation layers such as the PDF are
        responsible for display rounding.
        """
        
        coefs = model.params
        if model.df_resid > 0:
            conf_int = model.conf_int()
            standard_errors = model.bse.values
            pvalues = model.pvalues.values
            lower_ci = conf_int[0].values
            upper_ci = conf_int[1].values
        else:
            size = len(coefs)
            standard_errors = np.full(size, np.nan)
            pvalues = np.full(size, np.nan)
            lower_ci = np.full(size, np.nan)
            upper_ci = np.full(size, np.nan)

        coef_df = pd.DataFrame({
            "Variable": coefs.index,
            "Coefficient": coefs.values,
            "Std_Error": standard_errors,
            "Conf_Int": (upper_ci - lower_ci) / 2,
            "Lower_CI": lower_ci,
            "Upper_CI": upper_ci,
            "p_value": pvalues,
        })
        return coef_df

    def _calc_vif(self, md_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate variance inflation factors for predictor variables.

        Parameters
        ----------
        md_matrix : pd.DataFrame
            Model matrix (design matrix) including intercept.

        Returns
        -------
        pd.DataFrame
            VIF table with columns Variable and VIF. Intercept is excluded.
            Values are rounded to 3 decimal places.

        Notes
        -----
        VIF measures how much the variance of a coefficient is inflated due to
        collinearity with other predictors. Common thresholds:
        - VIF < 5: Low collinearity
        - 5 ≤ VIF ≤ 10: Moderate collinearity
        - VIF > 10: High collinearity (problematic)

        VIF is calculated as 1/(1-R²) where R² is from regressing each
        predictor on all other predictors.
        """
        if not isinstance(md_matrix, pd.DataFrame) or md_matrix.empty:
            raise ValueError("A non-empty model matrix is required to calculate VIF.")
        values = md_matrix.to_numpy(dtype=float)
        rows = []
        for index, variable in enumerate(md_matrix.columns):
            if str(variable).strip().lower() in {"int", "intercept", "const"}:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    value = float(variance_inflation_factor(values, index))
            except (ValueError, TypeError, np.linalg.LinAlgError, ZeroDivisionError):
                value = np.nan
            rows.append({"Variable": variable, "VIF": value})
        return pd.DataFrame(rows, columns=["Variable", "VIF"])

    def _fit_single_response(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        replicate_groups: list[list[int]],
    ) -> RegressionSummary:
        """Fit regression model for a single response variable.

        Performs OLS regression with comprehensive diagnostics including ANOVA,
        cross-validation, and optional lack-of-fit analysis for replicated
        designs.

        Parameters
        ----------
        X : pd.DataFrame
            Model matrix (design matrix) with shape (n_samples, n_features).
            Should include intercept column if desired.
        y : pd.Series
            Response variable values with shape (n_samples,).
        replicate_groups : list[list[int]]
            List of replicate groups where each group is a list of observation
            indices that are replicates (identical factor settings). Empty list
            if no replicates exist.

        Returns
        -------
        RegressionSummary
            Complete regression results including model, coefficients,
            predictions, residuals, metrics, ANOVA, and optional LOF analysis.

        Notes
        -----
        Computed metrics:
        - R²: Coefficient of determination (1 - SS_res/SS_tot)
        - R²_adj: Adjusted R² accounting for number of predictors
        - Q²: Predictive R² from leave-one-out cross-validation
        - PRESS: Leave-one-out prediction error sum of squares
        - RMSE: Root mean squared error
        - RMSE_CV: RMSE from cross-validation

        Lack-of-fit decomposition (only if replicates exist):
        - SS_res = SS_lof + SS_pe
        - SS_lof: Sum of squares due to lack-of-fit (model inadequacy)
        - SS_pe: Pure error from replicated observations

        The F-test for lack-of-fit compares MS_lof to MS_pe. A non-significant
        result means that significant lack of fit was not detected; it does not
        prove model adequacy.

        """

        X, y, replicate_groups = self._validate_inputs(X, y, replicate_groups)
        n_sample = X.shape[0]

        # ---------------------------- OLS ---------------------------- #
        model = sm.OLS(y, X).fit()
        y_hat = model.fittedvalues.copy()
        res = model.resid.copy()
        full_rank = int(model.model.rank)

        # ------------------------ LOOCV ----------------------------- #
        y_hat_cv = self._loocv_predictions(X, y, full_rank)
        res_cv = y.to_numpy(dtype=float) - y_hat_cv

        disp_matrix = model.normalized_cov_params.copy()

        # -------------------------- ANOVA --------------------------- #
        ss_res = float(model.ssr)
        ss_reg = float(model.ess)
        has_constant = bool(model.model.k_constant)
        ss_tot = float(model.centered_tss if has_constant else model.uncentered_tss)
        df_res = int(model.df_resid)
        df_reg = int(model.df_model)
        df_tot = df_reg + df_res
        ms_res = float(model.mse_resid) if df_res > 0 and np.isfinite(model.mse_resid) else np.nan
        ms_reg = float(model.mse_model) if df_reg > 0 and np.isfinite(model.mse_model) else np.nan
        ms_tot = ss_tot / df_tot if df_tot > 0 else np.nan
        rmse = float(np.sqrt(ms_res)) if np.isfinite(ms_res) and ms_res >= 0 else np.nan

        # ------------------------ Metrics --------------------------- #
        tss_tolerance = np.finfo(float).eps * max(1.0, float(np.dot(y, y))) * n_sample
        tss_defined = np.isfinite(ss_tot) and ss_tot > tss_tolerance
        r2 = float(model.rsquared) if tss_defined and np.isfinite(model.rsquared) else np.nan
        r2_adj = (
            float(model.rsquared_adj)
            if tss_defined and df_res > 0 and np.isfinite(model.rsquared_adj)
            else np.nan
        )

        if np.isfinite(y_hat_cv).all():
            press = float(np.dot(res_cv, res_cv))
            rmse_cv = float(np.sqrt(press / n_sample))
            q2 = 1.0 - press / ss_tot if tss_defined else np.nan
        else:
            press = np.nan
            rmse_cv = np.nan
            q2 = np.nan

        metrics = {
            "R2": r2,
            "R2_adj": r2_adj,
            "Q2": q2,
            "PRESS": press,
            "RMSE": rmse,
            "RMSE_CV": rmse_cv,
        }

        anova = {
            "SS_tot": ss_tot,
            "df_tot": df_tot,
            "MS_tot": ms_tot,
            "SS_reg": ss_reg,
            "df_reg": df_reg,
            "MS_reg": ms_reg,
            "SS_res": ss_res,
            "df_res": df_res,
            "MS_res": ms_res,
        }

        # ------------------- Replicates & LOF ----------------------- #
        repl_summary: Optional[dict] = None
        lof: Optional[dict[str, float]] = None

        if replicate_groups:
            ss_pe = 0.0
            dof_pe = 0
            repl_summary = {}

            for group in replicate_groups:
                y_group = y.iloc[group]
                n_repl = len(group)
                mean_y = float(np.mean(y_group))
                ss_pe_g = float(np.sum((y_group - mean_y) ** 2))
                dof_pe_g = n_repl - 1
                var = ss_pe_g / dof_pe_g if dof_pe_g > 0 else 0.0
                std = float(np.sqrt(var))

                repl_summary[tuple(group)] = {
                    "n_replicates": n_repl,
                    "mean": mean_y,
                    "ss_pe": ss_pe_g,
                    "dof": dof_pe_g,
                    "var": var,
                    "std": std,
                }

                ss_pe += ss_pe_g
                dof_pe += dof_pe_g

            if dof_pe > 0:
                ms_pe = ss_pe / dof_pe
                ss_lof = ss_res - ss_pe
                lof_tolerance = 1e-10 * max(1.0, abs(ss_res), abs(ss_pe))
                if ss_lof < -lof_tolerance:
                    raise ValueError(
                        "Pure-error sum of squares exceeds residual sum of squares; "
                        "replicate groups are inconsistent with the fitted design."
                    )
                if abs(ss_lof) <= lof_tolerance:
                    ss_lof = 0.0
                df_lof = df_res - dof_pe
                ms_lof = ss_lof / df_lof if df_lof > 0 else np.nan
                lof = {
                    "SS_pe": ss_pe,
                    "df_pe": dof_pe,
                    "MS_pe": ms_pe,
                    "SS_lof": ss_lof,
                    "df_lof": df_lof,
                    "MS_lof": ms_lof,
                }
                anova.update(lof)

        coef_table = self._coef_table(model)

        return RegressionSummary(
            model=model,
            coef=coef_table,
            dispersion_matrix=disp_matrix,
            y_hat=y_hat,
            residuals=res,
            y_hat_cv=y_hat_cv,
            residuals_cv=res_cv,
            metrics=metrics,
            anova=anova,
            lof=lof,
            replicates=repl_summary,
        )

    def mlr_fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        replicate_groups: list[list[int]],
    ) -> RegressionWrapper:
        """Fit multiple linear regression models for multiple responses.

        Fits separate regression models for each response variable in Y using
        the same design matrix X. Computes variance inflation factors once
        since they depend only on X.

        Parameters
        ----------
        X : pd.DataFrame
            Model matrix (design matrix) with shape (n_samples, n_features).
            Should include intercept column if desired. Used for all responses.
        Y : pd.DataFrame
            Response variables with shape (n_samples, n_responses). Each column
            is a separate response variable.
        replicate_groups : list[list[int]]
            List of replicate groups where each group is a list of observation
            indices that are replicates. Applies to all responses. Empty list
            if no replicates exist.

        Returns
        -------
        RegressionWrapper
            Container with regression results for all responses and shared VIF
            diagnostics.

        Notes
        -----
        Each response is fitted independently. The design matrix X and
        replicate structure are shared across all responses, but fitted
        coefficients, predictions, and diagnostics are response-specific.

        VIF is computed once since it depends only on the predictor structure,
        not on response values.

        Examples
        --------
        >>> import pandas as pd
        >>> analyzer = RegressionAnalyzer()
        >>> X = pd.DataFrame({
        ...     'Int': [1, 1, 1, 1],
        ...     'Temp': [-1, 1, -1, 1],
        ...     'Press': [-1, -1, 1, 1]
        ... })
        >>> Y = pd.DataFrame({
        ...     'Yield': [45, 55, 50, 60],
        ...     'Purity': [85, 90, 88, 92]
        ... })
        >>> wrapper = analyzer.mlr_fit(X, Y, replicate_groups=[])
        >>> list(wrapper.results.keys())
        ['Yield', 'Purity']
        >>> wrapper.vif
           Variable  VIF
        1      Temp  1.0
        2     Press  1.0
        """

        if not isinstance(Y, pd.DataFrame):
            raise TypeError("Y must be a pandas DataFrame.")
        if Y.empty or Y.shape[1] == 0:
            raise ValueError("At least one response is required for regression analysis.")
        if Y.columns.has_duplicates:
            raise ValueError("Response names must be unique.")

        results: dict[str, RegressionSummary] = {}

        for col in Y.columns:
            results[col] = self._fit_single_response(X, Y[col], replicate_groups)

        vif = self._calc_vif(X)
        return RegressionWrapper(results=results, vif=vif)
    
    def compute_model_f_test(response: str, regression_wrapper: RegressionWrapper) -> pd.DataFrame:
        """Compute F-test for overall model significance.

        Uses the overall F-test supplied by statsmodels. With an explicit or
        implicit constant this compares the fitted model with a constant-only
        model. Without a constant it tests whether all coefficients are zero.

        Parameters
        ----------
        response : str
            Name of the response variable to test.
        regression_wrapper : RegressionWrapper
            Fitted regression results from mlr_fit.

        Returns
        -------
        pd.DataFrame
            F-test results with columns:
            - F_value: computed F-statistic (MS_reg / MS_res)
            - F_crit_95%: critical F-value at α=0.05
            - F_crit_99%: critical F-value at α=0.01
            - p_value: probability of observing F-value under null hypothesis

        Notes
        -----
        The F-statistic tests:
        H₀: β₁ = β₂ = ... = βₚ = 0 (model has no predictive power)
        H₁: At least one βᵢ ≠ 0

        Decision rules:
        - If F_value > F_crit_95%, reject H₀ at 95% confidence
        - If p_value < 0.05, reject H₀ at 95% confidence
        - Larger F-values indicate stronger evidence against H₀

        """
        
        result = regression_wrapper.results[response]
        model = result.model
        df_reg = int(model.df_model)
        df_res = int(model.df_resid)

        if df_reg > 0 and df_res > 0:
            raw_f = model.fvalue
            raw_p = model.f_pvalue
            f_value = float(raw_f) if raw_f is not None else np.nan
            p_value = float(raw_p) if raw_p is not None else np.nan
            f_95 = f.ppf(0.95, df_reg, df_res)
            f_99 = f.ppf(0.99, df_reg, df_res)
        else:
            f_value = np.nan
            f_95 = np.nan
            f_99 = np.nan
            p_value = np.nan

        data = {
            "F_value": [f_value],
            "F_crit_95%": [f_95],
            "F_crit_99%": [f_99],
            "p_value": [p_value]
        }

        return pd.DataFrame(data)
    
    def compute_lof_f_test(response: str, regression_wrapper: RegressionWrapper) -> pd.DataFrame:
        """Compute F-test for lack-of-fit in replicated designs.

        Tests whether the residual variation attributable to lack of fit exceeds
        the pure error estimated from replicated design points.

        Parameters
        ----------
        response : str
            Name of the response variable to test.
        regression_wrapper : RegressionWrapper
            Fitted regression results from mlr_fit. Must include replicate
            groups in the original fit.

        Returns
        -------
        pd.DataFrame
            LOF F-test results with columns:
            - F_value: computed F-statistic (MS_lof / MS_pe)
            - F_crit_95%: critical F-value at α=0.05
            - F_crit_99%: critical F-value at α=0.01
            - p_value: probability of observing F-value under null hypothesis

        Notes
        -----
        Requires replicated observations (multiple runs at identical factor
        settings). The residual sum of squares is decomposed:
        SS_res = SS_lof + SS_pe

        Where:
        - SS_lof: lack-of-fit (model inadequacy)
        - SS_pe: pure error (experimental variability)

        The F-test evaluates:
        H₀: Model is adequate (no lack-of-fit)
        H₁: Model is inadequate (significant lack-of-fit)

        Decision rules:
        - If F_value < F_crit_95% or p_value > 0.05, fail to reject H₀
          (no statistically significant lack of fit was detected)
        - If F_value > F_crit_95% or p_value < 0.05, reject H₀
          (significant lack-of-fit, consider higher-order terms)

        """
        
        ols_sum = regression_wrapper.results[response].anova

        ms_lof = ols_sum["MS_lof"]
        ms_pe = ols_sum["MS_pe"]
        df_lof = ols_sum["df_lof"]
        df_pe = ols_sum["df_pe"]

        if df_lof > 0 and df_pe > 0 and np.isfinite(ms_lof):
            if ms_pe > 0:
                f_value = ms_lof / ms_pe
                p_value = f.sf(f_value, df_lof, df_pe)
            elif ms_pe == 0 and ms_lof > 0:
                f_value = np.inf
                p_value = 0.0
            else:
                f_value = np.nan
                p_value = np.nan

            f_95 = f.ppf(0.95, df_lof, df_pe)
            f_99 = f.ppf(0.99, df_lof, df_pe)
        else:
            f_value = np.nan
            f_95 = np.nan
            f_99 = np.nan
            p_value = np.nan

        data = {
            "F_value": [f_value],
            "F_crit_95%": [f_95],
            "F_crit_99%": [f_99],
            "p_value": [p_value]
        }

        return pd.DataFrame(data)
