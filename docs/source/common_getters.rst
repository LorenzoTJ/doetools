.. _common_getters:

Common Getters
==============

All design classes inherit these methods from
:class:`~doetools.utils.summary.DesignSummaryMixin`. They provide design
configuration, matrices, imported responses, and fitted values without exposing
the internal DataFrames directly.

.. note::
    DataFrame getters return defensive copies. Modifying a returned table does not
    modify the design object. Methods whose required state is unavailable raise
    ``ValueError`` rather than returning ``None``.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 29 46 25

   * - Method
     - Returns
     - Required state
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_design_summary`
     - Design type and run, replicate, center-point, and factor counts
     - Initialized design
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_factor_summary`
     - Type-aware factor definitions
     - Defined factors
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_model_term_summary`
     - Terms in the compiled model specification
     - Model terms set
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_model_term_count`
     - Number of terms requested by the compiled specification
     - Model terms set
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_response_condition_summary`
     - Limits and optimization direction for each imported response
     - Responses and complete conditions
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_design_matrix`
     - Factor settings in original experimental units
     - Initialized design
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_coded_design_matrix`
     - Factor settings in coded form
     - Initialized design
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_model_matrix`
     - Expanded matrix for the configured model terms
     - Model terms set
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_responses`
     - Imported experimental responses
     - Responses imported
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_fitted_values`
     - In-sample fitted values for every imported response
     - MLR model fitted
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_leverages`
     - Model-matrix leverage for each design run
     - Model terms set
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_coefficient_summary`
     - Coefficients, uncertainty, confidence intervals, and p-values
     - MLR model fitted
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_anova_summary`
     - ANOVA decomposition, with lack-of-fit rows when available
     - MLR model fitted
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_metric_summary`
     - R2, adjusted R2, Q2, PRESS, RMSE, and RMSE_CV
     - MLR model fitted
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_model_f_test`
     - Overall fitted-model F-test
     - MLR model fitted
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_lof_f_test`
     - Lack-of-fit F-test, or an empty table when unavailable
     - Fitted model with usable replicates
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_vif`
     - Variance inflation factors for model-matrix columns
     - MLR model fitted
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_dispersion_matrix`
     - Parameter dispersion matrix for one response
     - MLR model fitted
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_replicate_summary`
     - Per-group replicate statistics for one response
     - MLR model fitted
   * - :meth:`~doetools.utils.prediction.PredictionPointsMixin.get_prediction_points`
     - Loaded prediction factor settings in actual or coded units
     - Prediction points loaded
   * - :meth:`~doetools.utils.prediction.PredictionPointsMixin.get_prediction_results`
     - Factor settings, prediction, and mean-response confidence limits
     - Fitted model and prediction points
   * - :meth:`~doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_points`
     - Loaded confirmation factor settings
     - Confirmation runs loaded
   * - :meth:`~doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_responses`
     - All confirmation responses or one selected response
     - Confirmation runs loaded
   * - :meth:`~doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_results`
     - Grouped PIMean validation results for one response
     - Fitted model and confirmation runs
   * - :meth:`~doetools.utils.summary.DesignSummaryMixin.get_model_summary_pdf`
     - Writes the model summary report; does not return data
     - Report inputs available

Design Summaries
----------------

Design Summary
^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_design_summary

The result is a one-row DataFrame with these columns:

``Design``, ``Runs``, ``Replicates``, ``Center Points``, ``Continuous``,
``Categorical``, and ``Mixture Components``.

``Replicates`` counts repeated runs beyond the first occurrence in each replicate
group; it is not the number of replicate groups.

.. code-block:: python

   summary = design.get_design_summary()
   print(summary)

Factor Summary
^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_factor_summary

The result contains:

``Factor``, ``Type``, ``Levels Count``, ``Levels``, ``Coded Levels``,
``Reference Level``, ``Lower Bound``, and ``Upper Bound``.

Level-related fields are empty for mixture components because mixture factors are
defined by bounds rather than discrete levels. ``Reference Level`` is populated
only for categorical factors.

.. code-block:: python

   factors = design.get_factor_summary()
   print(factors)

Model Summaries
---------------

Model Term Summary
^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_model_term_summary

Call :meth:`set_model_terms` first. The returned table has ``Terms`` and
``Included`` columns and lists the intercept, linear, two-factor interaction,
quadratic, and three-factor interaction terms.

.. important::
    This table reports the requested compiled specification. It does not report
    model-matrix rank, aliasing, or whether every requested coefficient is
    independently estimable.

.. code-block:: python

   terms = design.get_model_term_summary()
   print(terms)

Model Term Count
^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_model_term_count

This method returns an ``int``, not a DataFrame. The count is the number of terms
in the compiled specification and is subject to the same estimability caveat as the
term summary.

.. code-block:: python

   n_terms = design.get_model_term_count()

Response Condition Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_response_condition_summary

This getter requires imported responses and a complete condition entry for every
response. Configure them with :meth:`set_response_conditions`. The result contains
``Response``, ``Lower Limit``, ``Upper Limit``, and ``Goal``; a ``None`` limit is
displayed as ``"No"``.

.. code-block:: python

   conditions = design.get_response_condition_summary()
   print(conditions)

Design and Model Matrices
-------------------------

Design Matrix
^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_design_matrix

Returns factor settings in their original experimental representation.

.. code-block:: python

   actual_design = design.get_design_matrix()

Coded Design Matrix
^^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_coded_design_matrix

Returns the representation used to construct model terms. Continuous-factor bounds
normally map to -1 and +1, although some designs can contain axial values outside
that range. Categorical factors use their numeric codes, while mixture components
remain proportions.

.. code-block:: python

   coded_design = design.get_coded_design_matrix()

Model Matrix
^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_model_matrix

Call :meth:`set_model_terms` first. The returned matrix contains exactly the
configured intercept, linear, interaction, and quadratic columns used for fitting.

.. code-block:: python

   model_matrix = design.get_model_matrix()

Responses and Fitted Values
---------------------------

Responses
^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_responses

Returns the response columns loaded by :meth:`import_responses`, aligned to the
internal design order through ``Exp. Idx``.

.. code-block:: python

   responses = design.get_responses()

Fitted Values
^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_fitted_values

Returns full-precision, in-sample fitted values for every imported response. These
are the fitted OLS values at the original design runs, not leave-one-out
cross-validation predictions and not predictions at new factor settings.

.. code-block:: python

   fitted = design.get_fitted_values()

For new factor settings, use :meth:`get_prediction_results` as described in
:doc:`prediction`.

Leverages
^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_leverages

Call :meth:`set_model_terms` first. The result has one ``Leverage`` column and one
row per design run. Leverage depends only on the configured model matrix, so a
fitted response model is not required. See :doc:`leverage_analysis` for
interpretation and a complete obtain-and-plot workflow.

.. code-block:: python

   leverages = design.get_leverages()

Regression Result Getters
-------------------------

The following methods require :meth:`compute_mlr_model`. Methods that accept a
``response`` argument raise ``ValueError`` if that response was not fitted. Their
formulas and statistical interpretation are described alongside the relevant API.

Coefficient Summary
^^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_coefficient_summary
   :noindex:

Returns ``Variable``, ``Coefficient``, ``Std_Error``, ``Conf_Int``, ``Lower_CI``,
``Upper_CI``, and ``p_value`` for one response.

ANOVA Summary
^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_anova_summary
   :noindex:

Returns ``Source``, ``SS``, ``df``, and ``MS``. The base rows are ``Total``,
``Regression``, and ``Residuals``; ``Pure Error`` and ``Lack of Fit`` are appended
only when the fitted model has the required replicate decomposition.

Metric Summary
^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_metric_summary
   :noindex:

Returns a one-row table containing ``R2``, ``R2_adj``, ``Q2``, ``PRESS``, ``RMSE``,
and ``RMSE_CV``.

Model F-test
^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_model_f_test
   :noindex:

Returns ``F_value``, ``F_crit_95%``, ``F_crit_99%``, and ``p_value`` for the overall
model test.

Lack-of-Fit F-test
^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_lof_f_test
   :noindex:

Returns the same four F-test columns. If pure-error or lack-of-fit degrees of
freedom are unavailable, it returns an empty DataFrame with those columns.

Variance Inflation Factors
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_vif
   :noindex:

Returns ``Variable`` and ``VIF`` for model-matrix columns. An explicitly named
intercept column is omitted.

Dispersion Matrix
^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_dispersion_matrix
   :noindex:

Returns the parameter dispersion matrix for one response. Values whose absolute
magnitude is within ``tol`` are replaced with zero in the returned copy.

Replicate Summary
^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_replicate_summary
   :noindex:

Returns ``Run Index``, ``n*``, ``Mean``, ``Var``, ``Std Dev``, and ``dof``. It
returns an empty table with those columns when no replicate groups are available.

**Example:**

.. code-block:: python

   coefficients = design.get_coefficient_summary(response="Yield")
   anova = design.get_anova_summary(response="Yield")
   metrics = design.get_metric_summary(response="Yield")
   model_test = design.get_model_f_test(response="Yield")
   lof_test = design.get_lof_f_test(response="Yield")
   vif = design.get_vif()
   dispersion = design.get_dispersion_matrix(response="Yield")
   replicates = design.get_replicate_summary(response="Yield")

Prediction-Point Getters
------------------------

These getters require points loaded with :meth:`load_prediction_points`.

.. automethod:: doetools.utils.prediction.PredictionPointsMixin.get_prediction_points
   :noindex:

.. automethod:: doetools.utils.prediction.PredictionPointsMixin.get_prediction_results
   :noindex:

The result for one response includes every factor column followed by
``Predicted``, ``CI Lower``, and ``CI Upper``. Factor settings use actual units by
default and coded values when ``coded=True``. If the fitted model has no valid
residual mean square, point predictions are still returned and both confidence
interval columns are ``NaN``.

.. code-block:: python

   points = design.get_prediction_points(coded=False)
   results = design.get_prediction_results(response="Yield", alpha=0.05)

See :doc:`prediction` for loading, validation, and the confidence-interval
definition.

Confirmation-Run Getters
------------------------

These getters require data loaded with :meth:`load_confirmation_runs`.

Confirmation Points
^^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_points
   :noindex:

Returns confirmation factor rows in actual or coded form, including replicated
rows.

Confirmation Responses
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_responses
   :noindex:

With no argument, returns every confirmation response. Passing ``response`` returns
a one-column DataFrame.

Confirmation Results
^^^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_results
   :noindex:

Returns one row per distinct factor setting with the replicate mean, fitted
prediction, residual, PIMean limits, and ``Within PI`` result.

.. code-block:: python

   points = design.get_confirmation_points(coded=False)
   responses = design.get_confirmation_responses()
   results = design.get_confirmation_results(response="Yield", alpha=0.05)

See :doc:`model_validation` for the complete confirmation workflow and interval
definition.

Report Generation
-----------------

``get_model_summary_pdf`` follows the package's ``get_*`` naming convention but is
an output action rather than a data getter: it writes a PDF and returns ``None``.

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_model_summary_pdf
   :noindex:

.. code-block:: python

   design.get_model_summary_pdf(filename="design_report.pdf")

See :doc:`pdf_reports` for report contents and the complete workflow.
