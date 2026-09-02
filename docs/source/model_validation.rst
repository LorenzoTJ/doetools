.. _model_validation:

Model Validation
================

Use confirmation runs to check whether a fitted OLS model predicts correctly for new, independent
experimental results at selected factor settings. Replicates at the same setting
are grouped, averaged, and compared with a prediction interval for their future
mean (PIMean).

**Quick Navigation:**

- `Load Confirmation Runs`_ - Import and validate independent experiments
- `Evaluate Confirmation Results`_ - Compare observations with PIMean intervals
- `Confirmation Plots`_ - Visualize predictions and residuals
- `Access Confirmation Data`_ - Retrieve imported settings and responses
- `Clear Confirmation Runs`_ - Remove confirmation data

.. note::
    Fit the model with :meth:`set_model_terms`, :meth:`import_responses`, and
    :meth:`compute_mlr_model` before loading confirmation runs. Confirmation data
    are kept separate from the original design and are not used to refit the model.

Load Confirmation Runs
----------------------

Load independent confirmation experiments from an Excel or CSV file.

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.load_confirmation_runs

The file must contain one column for every factor and every response in the fitted
model. Each row represents one confirmation run; repeated factor settings are
treated as replicates. ``Exp. Order``, ``Exp. Idx``, and other laboratory metadata
are optional, and additional columns are ignored.

Set ``coded=True`` when process and categorical factors are stored using their
coded values. Mixture components remain proportions regardless of this option.

**Example:**

.. code-block:: python

   design.load_confirmation_runs(
       file_path="confirmation_runs.xlsx",
       coded=False,
   )

The loader checks factor levels, numeric responses, continuous and mixture bounds,
mixture sums, and configured domain filters. If validation fails, previously loaded
confirmation data remain unchanged.

Evaluate Confirmation Results
-----------------------------

PIMean interval
^^^^^^^^^^^^^^^

For :math:`c` future confirmation runs at a setting represented by model vector
:math:`x_0`, *doetools* calculates

.. math::

   \hat y_0 \;\pm\; t_{1-\alpha/2,df_{res}}
   \sqrt{MS_{res}\left(\frac{1}{c}+
   x_0^{\mathsf T}(X^{\mathsf T}X)^{+}x_0\right)}.

This is the PIMean interval described for confirmation experiments by A. Jensen [AJENSEN2016]_. 
The :math:`1/c` term represents the variability of the
mean of :math:`c` future observations, while the other term represents the variability of the
prediction for a new observation. For :math:`c=1`, the formula reduces to the
usual prediction interval for one future observation.

The superscript :math:`+` denotes the Moore-Penrose pseudoinverse used by the
implementation. A setting passes when its observed confirmation mean lies within
the interval. This is a pointwise check for each setting; intervals are not adjusted
for simultaneous inference across multiple settings.

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_results

**Example:**

.. code-block:: python

   results = design.get_confirmation_results(
       response="Yield",
       alpha=0.05,
   )
   print(results)

The returned table contains the setting identifier, factor values, replicate count
``n``, observed mean, prediction, residual, interval limits, and ``Within PI``.
Residuals follow ``Observed - Predicted`` and values retain full precision.

Confirmation Plots
------------------

Plot grouped observed means against predictions or inspect confirmation residuals.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_confirmation_exp_vs_pred

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_confirmation_residuals

**Example:**

.. code-block:: python

   prediction_figure = design.plot_confirmation_exp_vs_pred(response="Yield")
   residual_figure = design.plot_confirmation_residuals(response="Yield")

   prediction_figure.show()
   residual_figure.show()

Access Confirmation Data
------------------------

Retrieve confirmation factor settings and measured responses without exposing
internal DataFrames.

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_points

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.get_confirmation_responses

**Example:**

.. code-block:: python

   actual_points = design.get_confirmation_points(coded=False)
   coded_points = design.get_confirmation_points(coded=True)
   all_responses = design.get_confirmation_responses()
   yield_response = design.get_confirmation_responses(response="Yield")

All getters return defensive copies. Passing a response name returns a one-column
DataFrame.

Clear Confirmation Runs
-----------------------

Remove loaded confirmation data from the design.

.. automethod:: doetools.utils.confirmation.ConfirmationRunsMixin.clear_confirmation_runs

Clearing confirmation runs does not modify the original experiments, model terms,
or fitted OLS model.

References
----------

.. [AJENSEN2016] Jensen, A. (2016). `Confirmation Runs in Design of Experiments
   <https://doi.org/10.1080/00224065.2016.11918157>`_. *Journal of Quality Technology*, 48(2), 162-177, DOI: 10.1080/00224065.2016.11918157
