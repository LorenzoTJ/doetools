.. _model_computation:

Model Computation
=================

*doetools* fits one ordinary least-squares (MLR) model for each imported response.
The same configured model matrix is used for every response. The fitted models provide
coefficients, ANOVA data, diagnostic metrics, and model-based plots.

This section covers model setup, fitting, and the available summary tables.

**Quick Navigation:**

- `Set Model Terms`_ - Define polynomial structure
- `Set Response Condition`_ - Define optimization goals
- `Compute MLR Model`_ - Fit regression models
- `Model Statistics`_ - Access regression results and diagnostics

.. note::
    Before computing models, ensure:
    
    - that you have imported your response data using :meth:`import_responses`
    - that you have defined the appropriate model terms (linear, interaction, quadratic) via :meth:`set_model_terms`

    Response conditions are optional. Set them with :meth:`set_response_conditions`
    only for Pareto analysis or plots that use response limits and objectives.

Set Model Terms
------------------
Define the terms to include in the regression model. ``"all"`` includes every
available term of that type; use ``None`` to omit a term type.

.. note::    
    If you want to visualize the leverage surface you need to set the model terms first.

.. automethod:: doetools.utils.abstract_design.Design.set_model_terms

The :class:`~doetools.ModelTerms` class provides a convenient way to specify model terms:

.. autoclass:: doetools.ModelTerms
    :no-members:

**Example:**

.. code-block:: python

   from doetools import ModelTerms
   
   # Screening: main effects only
   terms = ModelTerms(
       intercept=True,
       pro_main="all",
       pro_int2=None,
       pro_quadratic=None,
   )
   design.set_model_terms(terms)
   
   # Optimization: full quadratic model
   terms = ModelTerms(
       intercept=True,
       pro_main="all",
       pro_int2="all",
       pro_quadratic="all",
   )
   design.set_model_terms(terms)

   # Mixture design: quadratic Scheffe polynomial
   terms = ModelTerms(
       intercept=False,
       pro_main=None,
       pro_int2=None,
       pro_quadratic=None,
       scheffe_pol_order=2,
   )
   design.set_model_terms(terms)

Set Response Condition
-----------------------

Set optional optimization goals for each response: lower/upper limits and
maximize/minimize flags. These conditions are not used to fit the MLR model; they
are used by Pareto analysis and relevant plots.

.. note::
    Response names must already be defined, normally by exporting with a
    ``responses`` list. The three lists must have one value for each response.

.. automethod:: doetools.utils.abstract_design.Design.set_response_conditions

**Example:**

.. code-block:: python

   # Maximize yield, minimize cost, target range for purity
   design.set_response_conditions(
       lower_limits=[70, False, 95],    # 70% min yield, no min cost, 95% min purity
       upper_limits=[False, 200, 100],  # no max yield, $200 max cost, 100% max purity
       maximize=[True, False, True]     # maximize yield & purity, minimize cost
   )

Compute MLR Model
------------------
Fit separate MLR models using the defined model terms and imported responses.
The method also identifies replicate groups, which are used for pure-error and
lack-of-fit calculations when applicable.

.. note::
    In MLR regression, one model is computed for each response variable.

.. automethod:: doetools.utils.abstract_design.Design.compute_mlr_model

**Example:**

.. code-block:: python

   # After importing responses and setting model terms
   design.compute_mlr_model()

Model Statistics
-----------------

These methods summarize in compact `pd.DataFrame` tables the results of regression analysis, including coefficient estimates, ANOVA results, model quality metrics, and diagnostic tests.

Coefficient Summary
^^^^^^^^^^^^^^^^^^^^^^^

Get regression coefficients with standard errors and significance tests.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_coefficient_summary

**Example:**

.. code-block:: python

   coef_summary = design.get_coefficient_summary(response="Yield")
   print(coef_summary)
   
   # Output columns: Variable | Coefficient | Std_Error | Conf_Int |
   #                 Lower_CI | Upper_CI | p_value

ANOVA Summary
^^^^^^^^^^^^^^^^^

Get ANOVA decomposition table.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_anova_summary

**Example:**

.. code-block:: python

   anova_summary = design.get_anova_summary(response="Yield")
   print(anova_summary)
   
   # Output columns: Source | SS | df | MS
   # Pure Error and Lack of Fit rows are included only when available.

Diagnostic Metrics
^^^^^^^^^^^^^^^^^^

Get model quality metrics (R², Adjusted R², RMSE, etc.).

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_metric_summary

**Example:**

.. code-block:: python

   metric_summary = design.get_metric_summary(response="Yield")
   print(metric_summary)
   
   # Output columns: R2 | R2_adj | Q2 | PRESS | RMSE | RMSE_CV

Model F-test
^^^^^^^^^^^^^^^^

Get F-test results for overall model significance.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_model_f_test

**Example:**

.. code-block:: python

   f_test = design.get_model_f_test(response="Yield")
   print(f_test)
   
   # Output columns: F_value | F_crit_95% | F_crit_99% | p_value

**Interpretation:**

- **Null Hypothesis**: All coefficients (except intercept) are zero
- **p-value < 0.05**: Model is statistically significant (reject null hypothesis)
- **p-value > 0.05**: Model does not explain variance better than the mean

Lack-of-Fit Test
^^^^^^^^^^^^^^^^

Get lack-of-fit F-test results. The method returns an empty table if the fitted
model has no usable pure-error and lack-of-fit degrees of freedom.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_lof_f_test

**Example:**

.. code-block:: python

   lof_test = design.get_lof_f_test(response="Yield")
   print(lof_test)

   # Output columns: F_value | F_crit_95% | F_crit_99% | p_value

**Interpretation:**

- **Null Hypothesis**: No lack of fit (model is adequate)
- **p-value > 0.05**: Cannot detect lack of fit → model is adequate
- **p-value < 0.05**: Significant lack of fit → model inadequate (add terms or transform)
- **Requires replicates**: Pure error needed to separate model inadequacy from noise

Variance Inflation Factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get variance inflation factors to detect multicollinearity.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_vif

**Example:**

.. code-block:: python

   vif_summary = design.get_vif()
   print(vif_summary)
   
   # Output columns: Term | VIF

**Interpretation:**

- **VIF = 1**: No multicollinearity
- **VIF < 5**: Low multicollinearity (good) 
- **5 ≤ VIF < 10**: Moderate multicollinearity (potential concern)
- **VIF ≥ 10**: High multicollinearity (problematic - inflates standard errors)

Dispersion Matrix
^^^^^^^^^^^^^^^^^

Get coefficient dispersion (covariance) matrix.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_dispersion_matrix

**Example:**

.. code-block:: python

   dispersion = design.get_dispersion_matrix(response="Yield")
   print(dispersion)

**Interpretation:**

- **Diagonal elements**: Variances of coefficient estimates
- **Off-diagonal elements**: Covariances between coefficient estimates

Replicate Summary
^^^^^^^^^^^^^^^^^

Get statistics for replicate groups.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_replicate_summary

**Example:**

.. code-block:: python

   replicate_summary = design.get_replicate_summary(response="Yield")
   print(replicate_summary)
   
   # Output columns: Run Index | n* | Mean | Var | Std Dev | dof

**Interpretation:**

- **n***: Number of replicates at that design point
- **Mean**: Average response across replicates
- **Var / Std Dev**: Measurement variability at that point
- **High variance**: Inconsistent measurements