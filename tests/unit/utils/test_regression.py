"""
Comprehensive unit tests for the regression module.

Tests cover:
- Correctness against statsmodels reference implementation
- Mathematical properties and identities
- Edge cases and boundary conditions
- Lack-of-fit calculations with replicates
- F-test computations
- VIF calculations
- Cross-validation predictions
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from doetools.utils.regression import RegressionAnalyzer, RegressionWrapper, RegressionSummary


# ==============================================================================
#                          Test Fixtures
# ==============================================================================

@pytest.fixture
def simple_linear_data():
    """Simple linear regression data: y = 2 + 3*x + noise."""
    np.random.seed(42)
    X = pd.DataFrame({
        'Int': [1, 1, 1, 1, 1, 1, 1, 1],
        'x': [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    })
    # True relationship: y = 2 + 3*x, with noise
    y_true = 2 + 3 * X['x']
    noise = np.random.normal(0, 0.3, len(X))
    y = pd.Series(y_true + noise)
    return X, y


@pytest.fixture
def multiple_regression_data():
    """Multiple linear regression with interaction: y = 10 + 2*x1 + 3*x2 + 0.5*x1*x2 + noise."""
    np.random.seed(123)
    # Full factorial design at 2 levels
    X = pd.DataFrame({
        'Int': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'x1': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
        'x2': [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    })
    # True relationship with interaction
    y_true = 10 + 2*X['x1'] + 3*X['x2'] + 0.5*X['x1']*X['x2']
    noise = np.random.normal(0, 0.5, len(X))
    y = pd.Series(y_true + noise)
    return X, y


@pytest.fixture
def perfect_fit_data():
    """Data with perfect linear fit: y = 2*x (R² = 1, no noise for edge case testing)."""
    X = pd.DataFrame({
        'Int': [1, 1, 1, 1],
        'x': [1, 2, 3, 4]
    })
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    return X, y


@pytest.fixture
def data_with_replicates():
    """Data with replicated design points for LOF testing: y = 1 + 2*x + noise."""
    np.random.seed(456)
    X = pd.DataFrame({
        'Int': [1, 1, 1, 1, 1, 1, 1, 1],
        'x': [-1, -1, 0, 0, 1, 1, 0, 0]  # Replicates at -1, 0, and 1
    })
    # True linear relationship
    y_true = 1 + 2 * X['x']
    # Add realistic measurement noise
    noise = np.random.normal(0, 0.15, len(X))
    y = pd.Series(y_true + noise)
    replicate_groups = [[0, 1], [2, 3, 6, 7], [4, 5]]  # 4 replicates at center
    return X, y, replicate_groups


@pytest.fixture
def collinear_data():
    """Data with collinear predictors for VIF testing."""
    np.random.seed(789)
    n = 15
    X = pd.DataFrame({
        'Int': np.ones(n),
        'x1': np.linspace(-2, 2, n),
        'x2': 2 * np.linspace(-2, 2, n) + np.random.normal(0, 0.1, n),  # Nearly collinear with x1
        'x3': np.random.randn(n)  # Independent
    })
    # Response depends on all three with noise
    y_true = 10 + 2*X['x1'] + 3*X['x2'] + 1.5*X['x3']
    y = pd.Series(y_true + np.random.normal(0, 1.0, n))
    return X, y


@pytest.fixture
def analyzer():
    """Create a RegressionAnalyzer instance."""
    return RegressionAnalyzer()


@pytest.fixture
def quadratic_data():
    """Quadratic response surface: y = 50 - 2*x1 - 3*x2 + 0.5*x1² + 0.8*x2² - 1.2*x1*x2 + noise."""
    np.random.seed(111)
    # Central composite design points
    x_vals = np.array([-1, 1, -1, 1, -1.414, 1.414, 0, 0, 0, 0, 0])
    y_vals = np.array([-1, -1, 1, 1, 0, 0, -1.414, 1.414, 0, 0, 0])
    
    X = pd.DataFrame({
        'Int': np.ones(len(x_vals)),
        'x1': x_vals,
        'x2': y_vals,
        'x1^2': x_vals**2,
        'x2^2': y_vals**2,
        'x1*x2': x_vals * y_vals
    })
    
    # True quadratic relationship
    y_true = (50 - 2*X['x1'] - 3*X['x2'] + 0.5*X['x1^2'] + 
              0.8*X['x2^2'] - 1.2*X['x1*x2'])
    noise = np.random.normal(0, 0.8, len(X))
    y = pd.Series(y_true + noise)
    
    return X, y


@pytest.fixture
def curvature_data_with_replicates():
    """Data with curvature and center point replicates for LOF testing."""
    np.random.seed(222)
    # Factorial points + center point replicates
    X = pd.DataFrame({
        'Int': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'A': [-1, 1, -1, 1, 0, 0, 0, 0, 0],
        'B': [-1, -1, 1, 1, 0, 0, 0, 0, 0]
    })
    
    # True relationship has quadratic terms (curvature)
    # Linear model will have lack-of-fit
    y_true = 50 + 5*X['A'] + 3*X['B'] + 2*X['A']**2 + 1.5*X['B']**2
    noise = np.random.normal(0, 0.5, len(X))
    y = pd.Series(y_true + noise)
    
    replicate_groups = [[4, 5, 6, 7, 8]]  # 5 center point replicates
    return X, y, replicate_groups


# ==============================================================================
#                          Test Basic Functionality
# ==============================================================================

class TestRegressionBasics:
    """Test basic functionality and structure."""
    
    def test_analyzer_initialization(self, analyzer):
        """The analyzer is stateless; Statsmodels is fitted per response/fold."""
        assert isinstance(analyzer, RegressionAnalyzer)
        assert not hasattr(analyzer, "_lr")
        assert not hasattr(analyzer, "_loo")
    
    def test_fit_returns_regression_summary(self, analyzer, simple_linear_data):
        """Test that fit returns a RegressionSummary object with all attributes."""
        X, y = simple_linear_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        assert isinstance(result, RegressionSummary)
        assert result.model is not None
        assert isinstance(result.coef, pd.DataFrame)
        assert isinstance(result.dispersion_matrix, (np.ndarray, pd.DataFrame))
        assert isinstance(result.y_hat, (np.ndarray, pd.Series))
        assert isinstance(result.residuals, (np.ndarray, pd.Series))
        assert isinstance(result.y_hat_cv, np.ndarray)
        assert isinstance(result.residuals_cv, (np.ndarray, pd.Series))
        assert isinstance(result.metrics, dict)
        assert isinstance(result.anova, dict)
    
    def test_mlr_fit_multiple_responses(self, analyzer, multiple_regression_data):
        """Test fitting multiple responses simultaneously."""
        X, y = multiple_regression_data
        Y = pd.DataFrame({
            'response1': y,
            'response2': y * 1.5
        })
        
        result = analyzer.mlr_fit(X, Y, replicate_groups=[])
        
        assert isinstance(result, RegressionWrapper)
        assert len(result.results) == 2
        assert 'response1' in result.results
        assert 'response2' in result.results
        assert isinstance(result.vif, pd.DataFrame)
    
    def test_coef_table_structure(self, analyzer, simple_linear_data):
        """Test coefficient table has correct structure and columns."""
        X, y = simple_linear_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        coef = result.coef
        expected_cols = ['Variable', 'Coefficient', 'Std_Error', 'Conf_Int', 
                        'Lower_CI', 'Upper_CI', 'p_value']
        
        assert list(coef.columns) == expected_cols
        assert len(coef) == X.shape[1]  # One row per predictor


# ==============================================================================
#                    Test Correctness Against Reference
# ==============================================================================

class TestRegressionCorrectness:
    """Test calculations against statsmodels reference implementation."""
    
    def test_coefficients_match_statsmodels(self, analyzer, simple_linear_data):
        """Test that coefficients match statsmodels exactly."""
        X, y = simple_linear_data
        
        # Our implementation
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # Reference (statsmodels)
        model_sm = sm.OLS(y, X).fit()
        
        # Compare coefficients (should be identical)
        assert np.allclose(result.model.params.values, model_sm.params.values, atol=1e-10)
    
    def test_r_squared_matches_statsmodels(self, analyzer, multiple_regression_data):
        """Test that R² matches statsmodels."""
        X, y = multiple_regression_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        model_sm = sm.OLS(y, X).fit()
        
        assert np.isclose(result.metrics['R2'], model_sm.rsquared, atol=1e-10)
    
    def test_adjusted_r_squared_matches_statsmodels(self, analyzer, multiple_regression_data):
        """Test that adjusted R² matches statsmodels."""
        X, y = multiple_regression_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        model_sm = sm.OLS(y, X).fit()
        
        assert np.isclose(result.metrics['R2_adj'], model_sm.rsquared_adj, rtol=1e-4)
    
    def test_residuals_match_statsmodels(self, analyzer, simple_linear_data):
        """Test that residuals match statsmodels."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        model_sm = sm.OLS(y, X).fit()
        
        assert np.allclose(result.residuals, model_sm.resid.values, atol=1e-10)
    
    def test_predictions_match_statsmodels(self, analyzer, multiple_regression_data):
        """Test that predictions match statsmodels."""
        X, y = multiple_regression_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        model_sm = sm.OLS(y, X).fit()
        
        assert np.allclose(result.y_hat, model_sm.fittedvalues.values, atol=1e-10)
    
    def test_standard_errors_match_statsmodels(self, analyzer, simple_linear_data):
        """Test that standard errors match statsmodels."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        model_sm = sm.OLS(y, X).fit()
        
        assert np.allclose(result.coef['Std_Error'].values, 
                          model_sm.bse.values, atol=1e-4)
    
    def test_confidence_intervals_match_statsmodels(self, analyzer, simple_linear_data):
        """Test that confidence intervals match statsmodels."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        model_sm = sm.OLS(y, X).fit()
        
        ci_sm = model_sm.conf_int()
        
        assert np.allclose(result.coef['Lower_CI'].values, 
                          ci_sm[0].values, atol=1e-4)
        assert np.allclose(result.coef['Upper_CI'].values, 
                          ci_sm[1].values, atol=1e-4)
    
    def test_p_values_match_statsmodels(self, analyzer, simple_linear_data):
        """Test that p-values match statsmodels."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        model_sm = sm.OLS(y, X).fit()
        
        assert np.allclose(result.coef['p_value'].values, 
                          model_sm.pvalues.values, atol=1e-4)

    @pytest.mark.parametrize(
        "X",
        [
            pd.DataFrame({"Int": np.ones(8), "x": np.linspace(-1, 1, 8)}),
            pd.DataFrame({"x": np.linspace(-1, 1, 8)}),
            pd.DataFrame({
                "A": [1.0, 0.0, 0.5, 0.2, 0.8, 0.4, 0.7, 0.1],
                "B": [0.0, 1.0, 0.5, 0.8, 0.2, 0.6, 0.3, 0.9],
            }),
            pd.DataFrame({
                "A": [1.0, 0.0, 0.5, 0.2, 0.8, 0.4, 0.7, 0.1],
                "B": [0.0, 1.0, 0.5, 0.8, 0.2, 0.6, 0.3, 0.9],
                "Process": [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.5, -0.5],
            }),
        ],
        ids=["process-intercept", "process-no-intercept", "mixture", "mixed"],
    )
    def test_anova_and_metrics_use_statsmodels_source_of_truth(self, analyzer, X):
        y = pd.Series(np.linspace(2.0, 9.0, len(X)) + np.sin(np.arange(len(X))) * 0.2)

        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        reference = sm.OLS(y, X).fit()
        expected_tss = reference.centered_tss if reference.model.k_constant else reference.uncentered_tss

        assert np.isclose(result.anova["SS_res"], reference.ssr)
        assert np.isclose(result.anova["SS_reg"], reference.ess)
        assert np.isclose(result.anova["SS_tot"], expected_tss)
        assert result.anova["df_reg"] == int(reference.df_model)
        assert result.anova["df_res"] == int(reference.df_resid)
        assert result.anova["df_tot"] == int(reference.df_model + reference.df_resid)
        assert np.isclose(result.metrics["R2"], reference.rsquared)
        assert np.isclose(result.metrics["R2_adj"], reference.rsquared_adj)
        assert np.isclose(result.metrics["RMSE"], np.sqrt(reference.mse_resid))

    def test_rank_deficient_statistics_use_effective_rank(self, analyzer):
        x = np.linspace(-2.0, 2.0, 9)
        X = pd.DataFrame({"Int": 1.0, "x": x, "alias": 2.0 * x})
        y = pd.Series(3.0 + 1.5 * x + np.linspace(-0.1, 0.1, len(x)))

        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        reference = sm.OLS(y, X).fit()

        assert reference.model.rank == 2
        assert result.anova["df_reg"] == 1
        assert result.anova["df_res"] == len(X) - 2
        assert np.isclose(result.metrics["R2"], reference.rsquared)
        assert np.isclose(result.metrics["R2_adj"], reference.rsquared_adj)


# ==============================================================================
#                    Test Mathematical Properties
# ==============================================================================

class TestRegressionProperties:
    """Test mathematical properties that must hold."""
    
    def test_sum_of_squares_decomposition(self, analyzer, multiple_regression_data):
        """Test fundamental ANOVA identity: SS_tot = SS_reg + SS_res."""
        X, y = multiple_regression_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        anova = result.anova
        ss_total_computed = anova['SS_reg'] + anova['SS_res']
        
        assert np.isclose(anova['SS_tot'], ss_total_computed, atol=1e-8)
    
    def test_degrees_of_freedom_sum(self, analyzer, simple_linear_data):
        """Test that df_tot = df_reg + df_res."""
        X, y = simple_linear_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        anova = result.anova
        
        assert anova['df_tot'] == anova['df_reg'] + anova['df_res']
    
    def test_r2_bounds(self, analyzer, multiple_regression_data):
        """Test that R² is between 0 and 1 and model fits well with noise."""
        X, y = multiple_regression_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # R² should be bounded
        assert 0 <= result.metrics['R2'] <= 1
        assert 0 <= result.metrics['R2_adj'] <= 1
        
        # With moderate noise, should still fit well (R² > 0.8)
        assert result.metrics['R2'] > 0.8
    
    def test_residuals_sum_to_zero_with_intercept(self, analyzer, simple_linear_data):
        """Test that residuals sum to (nearly) zero when intercept is included."""
        X, y = simple_linear_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # Residuals should sum to zero (within numerical precision)
        assert np.abs(result.residuals.sum()) < 1e-10
    
    def test_mean_of_predictions_equals_mean_of_y(self, analyzer, multiple_regression_data):
        """Test that mean(y_hat) = mean(y) when intercept is included."""
        X, y = multiple_regression_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        assert np.isclose(result.y_hat.mean(), y.mean(), atol=1e-10)
    
    def test_ss_res_equals_sum_of_squared_residuals(self, analyzer, simple_linear_data):
        """Test that SS_res = sum(residuals²)."""
        X, y = simple_linear_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        ss_res_computed = np.sum(result.residuals ** 2)
        
        assert np.isclose(result.anova['SS_res'], ss_res_computed, atol=1e-10)
    
    def test_rmse_calculation(self, analyzer, multiple_regression_data):
        """Test that RMSE = sqrt(MS_res)."""
        X, y = multiple_regression_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        rmse_computed = np.sqrt(result.anova['MS_res'])
        
        assert np.isclose(result.metrics['RMSE'], rmse_computed, atol=1e-10)
    
    def test_adjusted_r2_less_than_or_equal_r2(self, analyzer, multiple_regression_data):
        """Test that adjusted R² ≤ R²."""
        X, y = multiple_regression_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        assert result.metrics['R2_adj'] <= result.metrics['R2'] + 1e-10


# ==============================================================================
#                          Test Edge Cases
# ==============================================================================

class TestRegressionEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_perfect_fit(self, analyzer, perfect_fit_data):
        """Test when model perfectly fits data (R²=1, residuals=0)."""
        X, y = perfect_fit_data
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        assert np.isclose(result.metrics['R2'], 1.0, atol=1e-10)
        assert np.allclose(result.residuals, 0.0, atol=1e-10)
        assert np.isclose(result.anova['SS_res'], 0.0, atol=1e-10)
    
    def test_no_relationship(self, analyzer):
        """Test when predictors have no relationship with response (R²≈0)."""
        # Generate random noise - predictor and response are independent
        np.random.seed(999)
        n = 25
        X = pd.DataFrame({
            'Int': np.ones(n),
            'x': np.random.randn(n)
        })
        y = pd.Series(np.random.randn(n))
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # R² should be very small (close to 0) - allowing some random correlation
        assert result.metrics['R2'] < 0.3
    
    def test_single_predictor(self, analyzer):
        """Test with only intercept (no predictors)."""
        X = pd.DataFrame({
            'Int': [1, 1, 1, 1]
        })
        y = pd.Series([2.0, 3.0, 4.0, 5.0])
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # With only intercept, y_hat should be constant (mean of y)
        assert np.allclose(result.y_hat, y.mean(), atol=1e-10)
        assert np.isclose(result.metrics['R2'], 0.0, atol=1e-10)
        assert result.anova['df_reg'] == 0
        assert np.isnan(result.anova['MS_reg'])
        assert np.isclose(result.metrics['R2_adj'], 0.0, atol=1e-10)

    def test_saturated_model_keeps_decomposition_but_undefined_error_ms(self, analyzer):
        """A saturated model has no residual DOF, but SS decomposition remains valid."""
        X = pd.DataFrame({
            'Int': [1, 1, 1],
            'x1': [0, 1, 0],
            'x2': [0, 0, 1],
        })
        y = pd.Series([2.0, 4.0, 7.0])

        result = analyzer._fit_single_response(X, y, replicate_groups=[])

        assert result.anova['df_res'] == 0
        assert np.isclose(
            result.anova['SS_tot'],
            result.anova['SS_reg'] + result.anova['SS_res'],
            atol=1e-10,
        )
        assert np.isclose(result.metrics['R2'], 1.0, atol=1e-10)
        assert np.isnan(result.anova['MS_res'])
        assert np.isnan(result.metrics['RMSE'])
        assert np.isnan(result.metrics['R2_adj'])

    def test_model_f_test_undefined_without_required_dof(self, analyzer):
        """Overall F-test is undefined when regression or residual DOF are absent."""
        X = pd.DataFrame({
            'Int': [1, 1, 1, 1]
        })
        y = pd.Series([2.0, 3.0, 4.0, 5.0])

        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        wrapper = RegressionWrapper(
            results={'response': result},
            vif=pd.DataFrame(),
        )
        f_test = RegressionAnalyzer.compute_model_f_test('response', wrapper)

        assert np.isnan(f_test['F_value'].values[0])
        assert np.isnan(f_test['F_crit_95%'].values[0])
        assert np.isnan(f_test['F_crit_99%'].values[0])
        assert np.isnan(f_test['p_value'].values[0])
    
    def test_more_predictors_than_observations(self, analyzer):
        """Test behavior with p > n (overdetermined system)."""
        X = pd.DataFrame({
            'Int': [1, 1, 1],
            'x1': [1, 2, 3],
            'x2': [2, 3, 4],
            'x3': [3, 4, 5],
            'x4': [4, 5, 6]
        })
        y = pd.Series([5, 10, 15])
        
        # Should not raise error but R² may be problematic
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # Basic checks
        assert result.model is not None
        assert len(result.y_hat) == len(y)
    
    def test_identical_observations(self, analyzer):
        """Test with all observations having same y value."""
        X = pd.DataFrame({
            'Int': [1, 1, 1, 1],
            'x': [1, 2, 3, 4]
        })
        y = pd.Series([5.0, 5.0, 5.0, 5.0])
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # SS_tot should be 0, R² undefined (set to nan)
        assert np.isclose(result.anova['SS_tot'], 0.0, atol=1e-10)
        assert np.isnan(result.metrics['R2'])

    @pytest.mark.parametrize(
        ("X", "y", "message"),
        [
            (pd.DataFrame({"x": [1.0, np.nan]}), pd.Series([1.0, 2.0]), "non-finite"),
            (pd.DataFrame({"x": [1.0, 2.0]}), pd.Series([1.0, np.inf]), "non-finite"),
            (pd.DataFrame({"x": [1.0, 2.0]}, index=[1, 2]), pd.Series([1.0, 2.0]), "aligned"),
        ],
    )
    def test_invalid_fit_inputs_fail_explicitly(self, analyzer, X, y, message):
        with pytest.raises(ValueError, match=message):
            analyzer._fit_single_response(X, y, replicate_groups=[])

    def test_duplicate_model_terms_are_rejected(self, analyzer):
        X = pd.DataFrame([[1.0, 2.0], [1.0, 3.0]], columns=["x", "x"])

        with pytest.raises(ValueError, match="duplicate term names"):
            analyzer._fit_single_response(X, pd.Series([1.0, 2.0]), replicate_groups=[])

    def test_invalid_replicate_groups_are_rejected(self, analyzer):
        X = pd.DataFrame({"Int": [1.0] * 4, "x": [0.0, 0.0, 1.0, 1.0]})
        y = pd.Series([1.0, 1.1, 2.0, 2.1])

        with pytest.raises(ValueError, match="more than one group"):
            analyzer._fit_single_response(X, y, replicate_groups=[[0, 1], [1, 0]])

    def test_replicate_groups_must_identify_identical_model_rows(self, analyzer):
        X = pd.DataFrame({"Int": [1.0] * 4, "x": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([1.0, 2.0, 1.1, 2.1])

        with pytest.raises(ValueError, match="identical model-matrix rows"):
            analyzer._fit_single_response(X, y, replicate_groups=[[0, 1]])


# ==============================================================================
#                        Test Lack-of-Fit Calculations
# ==============================================================================

class TestLackOfFit:
    """Test DOE-specific lack-of-fit calculations."""
    
    def test_lof_with_replicates(self, analyzer, data_with_replicates):
        """Test LOF calculation with known replicate groups."""
        X, y, replicate_groups = data_with_replicates
        
        result = analyzer._fit_single_response(X, y, replicate_groups)
        
        assert result.lof is not None
        assert 'SS_pe' in result.lof
        assert 'df_pe' in result.lof
        assert 'MS_pe' in result.lof
        assert 'SS_lof' in result.lof
        assert 'df_lof' in result.lof
        assert 'MS_lof' in result.lof
    
    def test_pure_error_degrees_of_freedom(self, analyzer, data_with_replicates):
        """Test that pure error DOF = sum(n_i - 1) for each replicate group."""
        X, y, replicate_groups = data_with_replicates
        
        result = analyzer._fit_single_response(X, y, replicate_groups)
        
        # 3 groups with 2 replicates each -> df_pe = 3*(2-1) = 3
        expected_df_pe = sum(len(group) - 1 for group in replicate_groups)
        
        assert result.lof['df_pe'] == expected_df_pe
    
    def test_lof_decomposition(self, analyzer, data_with_replicates):
        """Test that SS_res = SS_lof + SS_pe."""
        X, y, replicate_groups = data_with_replicates
        
        result = analyzer._fit_single_response(X, y, replicate_groups)
        
        ss_decomposed = result.lof['SS_lof'] + result.lof['SS_pe']
        
        assert np.isclose(result.anova['SS_res'], ss_decomposed, atol=1e-8)
    
    def test_lof_df_decomposition(self, analyzer, data_with_replicates):
        """Test that df_res = df_lof + df_pe."""
        X, y, replicate_groups = data_with_replicates
        
        result = analyzer._fit_single_response(X, y, replicate_groups)
        
        df_decomposed = result.lof['df_lof'] + result.lof['df_pe']
        
        assert result.anova['df_res'] == df_decomposed
    
    def test_replicate_summary_statistics(self, analyzer, data_with_replicates):
        """Test replicate group summary statistics."""
        X, y, replicate_groups = data_with_replicates
        
        result = analyzer._fit_single_response(X, y, replicate_groups)
        
        assert result.replicates is not None
        assert len(result.replicates) == len(replicate_groups)
        
        # Check first replicate group
        first_group_key = tuple(replicate_groups[0])
        first_repl = result.replicates[first_group_key]
        
        assert first_repl['n_replicates'] == 2
        assert np.isclose(first_repl['mean'], y.iloc[[0, 1]].mean())
    
    def test_pure_error_calculation(self, analyzer):
        """Test pure error with manually computable values."""
        np.random.seed(333)
        X = pd.DataFrame({
            'Int': [1, 1, 1, 1],
            'x': [1, 1, 2, 2]
        })
        y = pd.Series([10.0, 12.0, 20.0, 22.0])
        replicate_groups = [[0, 1], [2, 3]]
        
        result = analyzer._fit_single_response(X, y, replicate_groups)
        
        # Manual calculation:
        # Group 1: mean=11, SS=(10-11)² + (12-11)² = 2
        # Group 2: mean=21, SS=(20-21)² + (22-21)² = 2
        # Total SS_pe = 4
        expected_ss_pe = 4.0
        
        assert np.isclose(result.lof['SS_pe'], expected_ss_pe, atol=1e-10)
    
    def test_no_replicates_means_no_lof(self, analyzer, simple_linear_data):
        """Test that LOF is None when no replicates are provided."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        assert result.lof is None
        assert result.replicates is None

    def test_non_replicated_rows_are_rejected_before_lof_decomposition(self, analyzer):
        X = pd.DataFrame({"Int": [1.0] * 4, "x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="identical model-matrix rows"):
            analyzer._fit_single_response(X, y, replicate_groups=[[0, 3]])

    def test_lof_statistics_match_manual_decomposition_and_f_distribution(self, analyzer):
        """Validate pure error, LOF, and its F-test without reusing report values."""
        X = pd.DataFrame({
            "Int": [1.0] * 6,
            "x": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        })
        y = pd.Series([0.0, 2.0, 4.0, 6.0, 16.0, 18.0])
        groups = [[0, 1], [2, 3], [4, 5]]

        wrapper = analyzer.mlr_fit(X, pd.DataFrame({"response": y}), groups)
        result = wrapper.results["response"]
        lof_test = RegressionAnalyzer.compute_lof_f_test("response", wrapper)

        group_means = np.array([y.iloc[group].mean() for group in groups])
        pure_error = sum(float(((y.iloc[group] - mean) ** 2).sum()) for group, mean in zip(groups, group_means))
        reference = sm.OLS(y, X).fit()
        ss_lof = reference.ssr - pure_error
        df_pe = sum(len(group) - 1 for group in groups)
        df_lof = int(reference.df_resid) - df_pe
        f_value = (ss_lof / df_lof) / (pure_error / df_pe)

        assert np.isclose(result.lof["SS_pe"], pure_error)
        assert np.isclose(result.lof["SS_lof"], ss_lof)
        assert result.lof["df_pe"] == df_pe
        assert result.lof["df_lof"] == df_lof
        assert np.isclose(lof_test.loc[0, "F_value"], f_value)
        assert np.isclose(lof_test.loc[0, "p_value"], stats.f.sf(f_value, df_lof, df_pe))
        assert np.isclose(lof_test.loc[0, "F_crit_95%"], stats.f.ppf(0.95, df_lof, df_pe))

    def test_zero_lof_degrees_of_freedom_retains_pure_error_summary(self, analyzer):
        X = pd.DataFrame({"Int": [1.0] * 4, "x": [1.0, 1.0, 2.0, 2.0]})
        y = pd.Series([10.0, 12.0, 20.0, 22.0])

        result = analyzer._fit_single_response(X, y, replicate_groups=[[0, 1], [2, 3]])

        assert result.lof["df_lof"] == 0
        assert np.isnan(result.lof["MS_lof"])
        assert np.isclose(result.lof["SS_pe"], 4.0)


# ==============================================================================
#                          Test VIF Calculations
# ==============================================================================

class TestVIFCalculations:
    """Test variance inflation factor calculations."""
    
    def test_vif_calculation_basic(self, analyzer, simple_linear_data):
        """Test VIF calculation on simple data."""
        X, y = simple_linear_data
        
        vif = analyzer._calc_vif(X)
        
        assert isinstance(vif, pd.DataFrame)
        assert 'Variable' in vif.columns
        assert 'VIF' in vif.columns
    
    def test_vif_excludes_intercept(self, analyzer, multiple_regression_data):
        """Test that VIF calculation excludes intercept."""
        X, y = multiple_regression_data
        
        vif = analyzer._calc_vif(X)
        
        # Intercept should be excluded
        assert 'Int' not in vif['Variable'].values
        assert 'Intercept' not in vif['Variable'].values

    def test_vif_excludes_named_intercept_regardless_of_column_position(self, analyzer):
        X = pd.DataFrame({
            "x": [-1.0, 0.0, 1.0, 2.0],
            "Intercept": [1.0, 1.0, 1.0, 1.0],
        })

        vif = analyzer._calc_vif(X)

        assert list(vif["Variable"]) == ["x"]
    
    def test_vif_high_collinearity(self, analyzer, collinear_data):
        """Test VIF detects high collinearity."""
        X, y = collinear_data
        
        vif = analyzer._calc_vif(X)
        
        # x1 and x2 are nearly collinear, should have high VIF
        vif_x1 = vif.loc[vif['Variable'] == 'x1', 'VIF'].values[0]
        vif_x2 = vif.loc[vif['Variable'] == 'x2', 'VIF'].values[0]
        
        # VIF > 5 indicates problematic collinearity (relaxed from perfect collinearity)
        assert vif_x1 > 5
        assert vif_x2 > 5
        
        # x3 is independent, should have low VIF
        vif_x3 = vif.loc[vif['Variable'] == 'x3', 'VIF'].values[0]
        assert vif_x3 < 3
    
    def test_vif_no_collinearity(self, analyzer):
        """Test VIF is low when predictors are independent."""
        # Generate orthogonal predictors
        np.random.seed(42)
        X = pd.DataFrame({
            'Int': np.ones(20),
            'x1': np.random.randn(20),
            'x2': np.random.randn(20)
        })
        
        vif = analyzer._calc_vif(X)
        
        # Independent predictors should have VIF close to 1
        for vif_value in vif['VIF']:
            assert vif_value < 5  # Common threshold


# ==============================================================================
#                      Test Cross-Validation
# ==============================================================================

class TestCrossValidation:
    """Test LOOCV predictions."""
    
    def test_loocv_predictions_exist(self, analyzer, simple_linear_data):
        """Test that LOOCV predictions are computed."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        assert result.y_hat_cv is not None
        assert len(result.y_hat_cv) == len(y)
    
    def test_loocv_residuals_computed(self, analyzer, simple_linear_data):
        """Test that LOOCV residuals are computed correctly."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # Residuals should be y - y_hat_cv
        expected_residuals = y.values - result.y_hat_cv
        
        assert np.allclose(result.residuals_cv, expected_residuals, atol=1e-10)
    
    def test_q2_bounds(self, analyzer, multiple_regression_data):
        """Test that Q² is reasonable (can be negative for poor models)."""
        X, y = multiple_regression_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # Q² should typically be less than R²
        assert result.metrics['Q2'] <= result.metrics['R2'] + 0.01
    
    def test_rmse_cv_calculation(self, analyzer, simple_linear_data):
        """Test that RMSE_CV is computed correctly."""
        X, y = simple_linear_data
        
        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        
        # RMSE_CV = sqrt(SS_cv / n)
        ss_cv = np.sum(result.residuals_cv ** 2)
        expected_rmse_cv = np.sqrt(ss_cv / len(y))
        
        assert np.isclose(result.metrics['RMSE_CV'], expected_rmse_cv, atol=1e-10)

    @pytest.mark.parametrize(
        "X",
        [
            pd.DataFrame({"x": [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0]}),
            pd.DataFrame({
                "A": [1.0, 0.0, 0.5, 0.2, 0.8, 0.4],
                "B": [0.0, 1.0, 0.5, 0.8, 0.2, 0.6],
            }),
            pd.DataFrame({"A": [1.0, 0.0, 0.5, 0.2, 0.8, 0.4]}),
        ],
        ids=["process-no-intercept", "mixture-implicit-constant", "partial-mixture"],
    )
    def test_loocv_refits_the_identical_statsmodels_parameterization(self, analyzer, X):
        y = pd.Series([2.0, 2.8, 4.2, 6.1, 8.0, 10.2])
        expected = np.empty(len(y))
        for omitted in range(len(y)):
            keep = np.arange(len(y)) != omitted
            fold = sm.OLS(y.iloc[keep], X.iloc[keep]).fit()
            expected[omitted] = float(fold.predict(X.iloc[[omitted]]).iloc[0])

        result = analyzer._fit_single_response(X, y, replicate_groups=[])
        reference = sm.OLS(y, X).fit()
        tss = reference.centered_tss if reference.model.k_constant else reference.uncentered_tss
        press = float(np.sum((y.to_numpy() - expected) ** 2))

        assert np.allclose(result.y_hat_cv, expected)
        assert np.isclose(result.metrics["PRESS"], press)
        assert np.isclose(result.metrics["RMSE_CV"], np.sqrt(press / len(y)))
        assert np.isclose(result.metrics["Q2"], 1.0 - press / tss)
        assert "Q2_adj" not in result.metrics

    def test_saturated_model_has_no_interpretable_loocv_metrics(self, analyzer):
        X = pd.DataFrame({"Int": [1.0, 1.0, 1.0], "x1": [1.0, 0.0, 0.0], "x2": [0.0, 1.0, 0.0]})
        y = pd.Series([2.0, 4.0, 7.0])

        result = analyzer._fit_single_response(X, y, replicate_groups=[])

        assert np.isnan(result.y_hat_cv).all()
        assert np.isnan(result.metrics["PRESS"])
        assert np.isnan(result.metrics["RMSE_CV"])
        assert np.isnan(result.metrics["Q2"])

    def test_any_loocv_fold_that_loses_rank_makes_cv_metrics_unavailable(self, analyzer):
        X = pd.DataFrame({
            "Int": [1.0, 1.0, 1.0, 1.0],
            "rare_term": [1.0, 0.0, 0.0, 0.0],
        })
        y = pd.Series([3.0, 1.0, 1.2, 0.8])

        result = analyzer._fit_single_response(X, y, replicate_groups=[])

        assert np.isnan(result.y_hat_cv[0])
        assert np.isnan(result.metrics["PRESS"])
        assert np.isnan(result.metrics["RMSE_CV"])
        assert np.isnan(result.metrics["Q2"])


# ==============================================================================
#                          Test F-Test Methods
# ==============================================================================

class TestFTestMethods:
    """Test model and LOF F-test calculations."""
    
    def test_model_f_test_computation(self, analyzer, multiple_regression_data):
        """Test model significance F-test calculation."""
        X, y = multiple_regression_data
        Y = pd.DataFrame({'response': y})
        
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups=[])
        f_test = RegressionAnalyzer.compute_model_f_test('response', wrapper)
        
        assert isinstance(f_test, pd.DataFrame)
        assert 'F_value' in f_test.columns
        assert 'F_crit_95%' in f_test.columns
        assert 'F_crit_99%' in f_test.columns
        assert 'p_value' in f_test.columns
    
    def test_model_f_value_calculation(self, analyzer, simple_linear_data):
        """Test that F-value is computed as MS_reg / MS_res."""
        X, y = simple_linear_data
        Y = pd.DataFrame({'response': y})
        
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups=[])
        f_test = RegressionAnalyzer.compute_model_f_test('response', wrapper)
        
        anova = wrapper.results['response'].anova
        expected_f = anova['MS_reg'] / anova['MS_res']
        
        assert np.isclose(f_test['F_value'].values[0], expected_f, atol=1e-4)
    
    def test_model_f_p_value_matches_scipy(self, analyzer, multiple_regression_data):
        """Test that p-value matches scipy.stats.f."""
        X, y = multiple_regression_data
        Y = pd.DataFrame({'response': y})
        
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups=[])
        f_test = RegressionAnalyzer.compute_model_f_test('response', wrapper)
        
        anova = wrapper.results['response'].anova
        f_value = anova['MS_reg'] / anova['MS_res']
        expected_p = 1.0 - stats.f.cdf(f_value, anova['df_reg'], anova['df_res'])
        
        assert np.isclose(f_test['p_value'].values[0], expected_p, atol=1e-4)

    def test_model_f_test_matches_statsmodels_without_constant(self, analyzer):
        X = pd.DataFrame({"x": [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0]})
        y = pd.Series([2.0, 2.8, 4.2, 6.1, 8.0, 10.2])
        wrapper = analyzer.mlr_fit(X, pd.DataFrame({"response": y}), replicate_groups=[])
        reference = sm.OLS(y, X).fit()

        f_test = RegressionAnalyzer.compute_model_f_test("response", wrapper)

        assert np.isclose(f_test.loc[0, "F_value"], reference.fvalue)
        assert np.isclose(f_test.loc[0, "p_value"], reference.f_pvalue)

    @pytest.mark.parametrize(
        "X",
        [
            pd.DataFrame({"Int": np.ones(7), "x": np.linspace(-2.0, 2.0, 7)}),
            pd.DataFrame({
                "A": [1.0, 0.0, 0.5, 0.2, 0.8, 0.4, 0.7],
                "B": [0.0, 1.0, 0.5, 0.8, 0.2, 0.6, 0.3],
            }),
        ],
        ids=["explicit-intercept", "mixture-implicit-intercept"],
    )
    def test_model_f_test_matches_statsmodels_with_a_constant(self, analyzer, X):
        y = pd.Series([2.1, 2.8, 4.0, 3.3, 5.2, 4.1, 5.0])
        wrapper = analyzer.mlr_fit(X, pd.DataFrame({"response": y}), replicate_groups=[])
        reference = sm.OLS(y, X).fit()

        f_test = RegressionAnalyzer.compute_model_f_test("response", wrapper)

        assert reference.model.k_constant == 1
        assert np.isclose(f_test.loc[0, "F_value"], reference.fvalue)
        assert np.isclose(f_test.loc[0, "p_value"], reference.f_pvalue)
        assert np.isclose(f_test.loc[0, "F_crit_95%"], stats.f.ppf(0.95, reference.df_model, reference.df_resid))
    
    def test_lof_f_test_computation(self, analyzer, data_with_replicates):
        """Test LOF F-test calculation."""
        X, y, replicate_groups = data_with_replicates
        Y = pd.DataFrame({'response': y})
        
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups)
        f_test = RegressionAnalyzer.compute_lof_f_test('response', wrapper)
        
        assert isinstance(f_test, pd.DataFrame)
        assert 'F_value' in f_test.columns
        assert 'F_crit_95%' in f_test.columns
        assert 'F_crit_99%' in f_test.columns
        assert 'p_value' in f_test.columns
    
    def test_lof_f_value_calculation(self, analyzer, data_with_replicates):
        """Test that LOF F-value is computed as MS_lof / MS_pe."""
        X, y, replicate_groups = data_with_replicates
        Y = pd.DataFrame({'response': y})
        
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups)
        f_test = RegressionAnalyzer.compute_lof_f_test('response', wrapper)
        
        anova = wrapper.results['response'].anova
        expected_f = anova['MS_lof'] / anova['MS_pe']
        
        assert np.isclose(f_test['F_value'].values[0], expected_f, atol=1e-4)
    
    def test_f_critical_values_positive(self, analyzer, simple_linear_data):
        """Test that F critical values are positive."""
        X, y = simple_linear_data
        Y = pd.DataFrame({'response': y})
        
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups=[])
        f_test = RegressionAnalyzer.compute_model_f_test('response', wrapper)
        
        assert f_test['F_crit_95%'].values[0] > 0
        assert f_test['F_crit_99%'].values[0] > 0
        assert f_test['F_crit_99%'].values[0] > f_test['F_crit_95%'].values[0]


# ==============================================================================
#                        Test Integration Scenarios
# ==============================================================================

class TestIntegrationScenarios:
    """Test realistic end-to-end scenarios."""
    
    def test_full_workflow_simple_regression(self, analyzer):
        """Test complete workflow for simple regression analysis."""
        # Create design
        X = pd.DataFrame({
            'Int': [1, 1, 1, 1, 1],
            'Temperature': [-1, -0.5, 0, 0.5, 1],
            'Pressure': [-1, -0.5, 0, 0.5, 1]
        })
        Y = pd.DataFrame({
            'Yield': [45, 52, 60, 68, 75],
            'Purity': [85, 88, 90, 92, 95]
        })
        
        # Fit model
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups=[])
        
        # Verify structure
        assert len(wrapper.results) == 2
        assert 'Yield' in wrapper.results
        assert 'Purity' in wrapper.results
        
        # Check both responses have valid metrics
        for response in ['Yield', 'Purity']:
            result = wrapper.results[response]
            assert 0 <= result.metrics['R2'] <= 1
            assert result.metrics['RMSE'] > 0
            assert len(result.coef) == 3  # Int, Temperature, Pressure
    
    def test_full_workflow_with_replicates_and_lof(self, analyzer):
        """Test complete workflow including LOF analysis."""
        # Design with center point replicates
        X = pd.DataFrame({
            'Int': [1, 1, 1, 1, 1, 1, 1],
            'A': [-1, 1, -1, 1, 0, 0, 0],
            'B': [-1, -1, 1, 1, 0, 0, 0]
        })
        Y = pd.DataFrame({
            'Response': [20, 40, 30, 50, 34, 36, 35]
        })
        replicate_groups = [[4, 5, 6]]  # Center points are replicates
        
        # Fit with LOF
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups)
        
        result = wrapper.results['Response']
        
        # Should have LOF analysis
        assert result.lof is not None
        assert result.replicates is not None
        
        # Can compute LOF F-test
        lof_test = RegressionAnalyzer.compute_lof_f_test('Response', wrapper)
        assert lof_test is not None
        assert len(lof_test) == 1
    
    def test_quadratic_model_fit(self, analyzer, quadratic_data):
        """Test fitting a quadratic response surface model."""
        X, y = quadratic_data
        Y = pd.DataFrame({'response': y})
        
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups=[])
        
        # Quadratic model should fit well with realistic noise
        assert wrapper.results['response'].metrics['R2'] > 0.90
        
        # Check that quadratic coefficients are negative (concave surface)
        coef = wrapper.results['response'].coef
        x1_sq_coef = coef.loc[coef['Variable'] == 'x1^2', 'Coefficient'].values[0]
        x2_sq_coef = coef.loc[coef['Variable'] == 'x2^2', 'Coefficient'].values[0]
        
        # Both should be positive (true values are 0.5 and 0.8)
        assert x1_sq_coef > 0
        assert x2_sq_coef > 0
        
        # Interaction term should be negative (true value is -1.2)
        interaction_coef = coef.loc[coef['Variable'] == 'x1*x2', 'Coefficient'].values[0]
        assert interaction_coef < 0
    
    def test_curvature_detection_with_lof(self, analyzer, curvature_data_with_replicates):
        """Test that LOF detects model inadequacy when curvature is present."""
        X, y, replicate_groups = curvature_data_with_replicates
        Y = pd.DataFrame({'response': y})
        
        # Fit linear model (inadequate for quadratic data)
        wrapper = analyzer.mlr_fit(X, Y, replicate_groups)
        
        result = wrapper.results['response']
        
        # Should have LOF analysis
        assert result.lof is not None
        
        # Compute LOF F-test
        lof_test = RegressionAnalyzer.compute_lof_f_test('response', wrapper)
        
        # With curvature present and linear model, LOF should be detectable
        # (though may not always be statistically significant with noise)
        assert lof_test is not None
        assert 'F_value' in lof_test.columns
        
        # Pure error should be relatively small (from center point replicates)
        assert result.lof['MS_pe'] < result.anova['MS_res']
