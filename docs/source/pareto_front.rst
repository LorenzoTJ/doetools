.. _pareto_front:

Pareto Front
============

Use a Pareto front to compare trade-offs between at least two predicted
responses, such as maximizing yield while minimizing cost.

Workflow
--------

Start with a design whose model terms are configured and experimental responses
are imported (see :doc:`model_computation`). This example assumes the response
order is ``Yield``, then ``Cost``.

1. Fit the response models and set each response's optimization direction.

   .. code-block:: python

      design.compute_mlr_model()
      design.set_response_conditions(
          lower_limits=[None, None],
          upper_limits=[None, None],
          maximize=[True, False],
      )

2. Compute the front over the candidate factor settings. Replace the factor
   names and values below with those of your design; values use actual units.
   Omit ``factor_levels`` to use the declared process-factor levels.

   .. code-block:: python

      design.compute_pareto_front(
          responses=["Yield", "Cost"],
          factor_levels={
              "Temperature": [40, 50, 60],
              "Pressure": [1, 2, 3],
          },
      )

   For mixture designs, set the resolution with ``mixture_grid={"degree": 8}``.
   Response limits are plot guides; they do not filter candidates.

3. Retrieve the non-dominated settings and display the trade-offs.

   .. code-block:: python

      front = design.get_pareto_front()
      print(front)
      fig = design.plot_pareto_front(x="Yield", y="Cost")
      fig.show()

Choose a setting from ``front`` based on your priorities. Recompute the front
after changing the fitted models, optimization directions, or candidate grid.

Function Reference
------------------

Compute the front
^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.abstract_design.Design.compute_pareto_front
   :noindex:

``responses`` selects the objectives; omitting it uses all imported responses.
At least two are required, with fitted models and optimization directions set
through :meth:`~doetools.utils.abstract_design.Design.set_response_conditions`.
``factor_levels`` selects process-factor values, ``mixture_grid`` controls mixture
resolution, and ``max_candidates`` limits the candidate grid size. The method
stores the results and returns ``None``.

Retrieve the front
^^^^^^^^^^^^^^^^^^

.. automethod:: doetools.utils.pareto._ParetoAnalysisMixin.get_pareto_front
   :noindex:

Returns a DataFrame containing candidate IDs, factor settings, and predicted
objective responses. Call ``compute_pareto_front()`` first.

Plot the front
^^^^^^^^^^^^^^

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_pareto_front
   :noindex:

Returns a Plotly figure; call ``fig.show()`` to display it. Omit ``x`` and ``y``
to use the first two computed objectives, or supply ``z`` for a 3D plot. Use
``show_candidates=False`` to display only the front.
