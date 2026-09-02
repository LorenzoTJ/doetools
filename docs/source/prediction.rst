.. _prediction:

Prediction
==========

Use a fitted MLR model to calculate point predictions at new factor settings.
Prediction requires a model that has already been configured with
:meth:`set_model_terms` and fitted with :meth:`compute_mlr_model`.

Prediction at new points
------------------------

For process and categorical factors, provide values on the coded scale used by
the design. Mixture components remain component proportions. The input DataFrame
must contain the factor columns used by the fitted model.

.. automethod:: doetools.utils.abstract_design.Design.predict

**Example:**

.. code-block:: python

   import pandas as pd

   # Coded settings for two process factors.
   new_points = pd.DataFrame({
       "Temperature": [0.5, -0.3],
       "Pressure": [0.0, 0.8],
   })

   predictions = design.predict(
       matrix_to_pred=new_points,
       responses=["Yield", "Purity"],
   )

The returned DataFrame has one row per prediction point and one column per
requested response. These are point predictions; confidence and prediction
intervals are not calculated by :meth:`predict`.