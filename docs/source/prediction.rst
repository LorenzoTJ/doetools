.. _prediction:

Prediction
==========

Use a fitted MLR model to predict one response at factor settings loaded from a
CSV or Excel workbook. The points remain stored on the design, while predictions
and confidence intervals are recalculated from the current model whenever they
are requested.

Load prediction points
----------------------

.. automethod:: doetools.utils.prediction.PredictionPointsMixin.load_prediction_points

The file must contain one column for every factor. Additional columns are
ignored, row order and repeated settings are retained, and invalid files do not
replace previously loaded points. Values use actual factor units by default:

.. code-block:: python

   design.load_prediction_points(
       source="prediction_points.xlsx",
       coded=False,
   )

An existing DataFrame can be loaded without writing an intermediate file:

.. code-block:: python

   design.load_prediction_points(source=prediction_points)

Set ``coded=True`` when continuous and categorical factors use their coded
levels. Mixture components are always supplied as proportions. The same bounds,
categorical levels, mixture constraints, and domain filters used for confirmation
runs are applied to prediction points.

Retrieve stored points
----------------------

.. automethod:: doetools.utils.prediction.PredictionPointsMixin.get_prediction_points

Both forms are available as defensive copies:

.. code-block:: python

   actual_points = design.get_prediction_points()
   coded_points = design.get_prediction_points(coded=True)

Calculate predictions and confidence intervals
----------------------------------------------

Configure and fit the model with :meth:`set_model_terms` and
:meth:`compute_mlr_model`, then select one fitted response:

.. automethod:: doetools.utils.prediction.PredictionPointsMixin.get_prediction_results

.. code-block:: python

   results = design.get_prediction_results(
       response="Yield",
       alpha=0.05,
       coded=False,
   )

The result contains the factor-setting columns followed by ``Predicted``,
``CI Lower``, and ``CI Upper``. Setting ``coded=True`` changes only the displayed
factor values; statistical results are identical.

For a model row vector :math:`x_0`, the pointwise confidence interval for the
expected mean response is

.. math::

   \hat{y}_0 \pm t_{1-\alpha/2,\,df_{res}}
   \sqrt{MS_{res}\,x_0^T(X^TX)^+x_0}.

This interval describes uncertainty in the estimated mean response. It is not a
prediction interval for an individual future observation and is not adjusted
simultaneously across multiple loaded points.

Clear prediction points
-----------------------

.. automethod:: doetools.utils.prediction.PredictionPointsMixin.clear_prediction_points

Clearing the points does not modify the fitted model, responses, design matrix,
or confirmation runs.
