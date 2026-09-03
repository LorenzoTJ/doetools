.. _plots:

Visualization & Plotting
========================

doetools returns interactive `Plotly <https://plotly.com/python/>`_ figures.
In a notebook, return the figure from the final cell or call ``fig.show()``.
Use ``fig.write_html("plot.html")`` for a self-contained interactive file.
Static image export through ``fig.write_image(...)`` requires Plotly's static
image export dependency.

Available plots
---------------

The table below lists every public plotting method. Methods inherited from
``GraphsMixin`` are shared by the design classes that expose the standard graph
API; candidate-set and Pareto plots are available only on the classes that
provide those workflows.

.. list-table::
   :header-rows: 1
   :widths: 28 45 27

   * - Method
     - Use and required state
     - Result
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_design`
     - Inspect process or mixture design points. Requires a generated or
       imported design.
     - One 2D, 3D, ternary, or tetrahedral figure.
   * - :meth:`~doetools.DOptDesign.plot_candidate_set`
     - Inspect a new D-optimal design's candidate set before optimization.
     - One 2D, 3D, ternary, or tetrahedral figure.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_leverage`
     - Explore model leverage. Requires model terms, but not fitted responses.
     - Contour and surface figures.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_response`
     - Explore a fitted response, optionally with a second response,
       confidence correction, or feasible region.
     - Contour and surface figures.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_confidence_interval`
     - Explore confidence half-width for the fitted mean. Requires a fitted
       model and variance statistics.
     - Contour and surface figures.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_regression_coefficients`
     - Compare fitted coefficients and their confidence intervals.
     - One figure, optionally with a response selector.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_diagnostics`
     - Select an overview, observed-versus-predicted, residual, Q-Q, or
       histogram diagnostic.
     - One figure, optionally with a response selector.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_main_effects`
     - Compare fitted main-effect profiles for all non-mixture factors.
     - One multi-panel figure.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_interactions`
     - Compare all fitted two-factor interactions for a non-mixture design.
     - One figure with a factor-pair selector.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_mixture_trace`
     - Inspect fitted component traces through the centroid blend.
     - One figure.
   * - :meth:`~doetools.graphs.plot_api_mixin.GraphsMixin.plot_confirmation`
     - Select observed-versus-predicted or residual confirmation diagnostics.
     - One figure for one response.
   * - :meth:`~doetools.utils.pareto.ParetoMixin.plot_pareto_front`
     - Inspect non-dominated solutions after ``compute_pareto_front()``.
     - One 2D or 3D figure.
   * - :meth:`~doetools.DOptDesign.compute_d_optimal`
     - Run D-optimal selection and inspect the best log-determinant by run
       count.
     - One optimization-history figure.

The D-optimal computation methods also return an optimization-history figure.
Because they run the optimization rather than only render existing data, they
are documented with the :doc:`D-optimal design <designs/optimal/doptimal>` and
:doc:`D-optimal augmentation <designs/optimal/doptimal_augmentation>` workflows.

Design-space visualization
--------------------------

Design points
^^^^^^^^^^^^^

``plot_design`` selects its geometry from the chosen factor types and number of
axes. Two or three non-mixture factors produce Cartesian plots. Two mixture
components produce a mixture-line plot, three produce a ternary plot, and four
produce a tetrahedral plot. Mixture and non-mixture factors cannot be combined
as axes, but continuous and categorical axes can be combined.

For mixture plots, ``domain="allowed"`` draws the domain covered by the design
instead of the full simplex or tetrahedron. When omitted factors make multiple
runs share the same plotted coordinates, projected points are aggregated by
default and their replicate counts are retained.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_design

.. code-block:: python

   figure = design.plot_design(
       ax1="Temperature",
       ax2="Pressure",
       ax3="Time",
       coded=False,
   )
   figure.show()

D-optimal candidate sets
^^^^^^^^^^^^^^^^^^^^^^^^

Candidate-set plots are specific to D-optimal workflows. For augmentation,
generate the candidate set before plotting it. The two methods have different
state requirements, so both signatures are shown.

.. automethod:: doetools.DOptDesign.plot_candidate_set
   :noindex:

.. code-block:: python

   candidate_figure = design.plot_candidate_set(
       ax1="Temperature",
       ax2="Pressure",
       ax3="Time",
       coded=False,
   )

Leverage and fitted surfaces
----------------------------

The surface methods return two figures in this order:
``(contour_figure, surface_figure)``. With two free process factors these are a
Cartesian contour and a 3D surface. Supplying ``ax3`` produces the corresponding
mixture contour and surface.

For process factors, grid bounds are always supplied in coded units, even when
``coded=False`` controls the displayed labels. Factors not assigned to an axis
must be supplied in ``constant_levels`` when the method cannot infer them.

Leverage
^^^^^^^^

Leverage depends on the model matrix, not on observed response values. It can
therefore be inspected after model terms are defined and before fitting a model.
Use ``domain="allowed"`` to apply saved process-domain filters or limit mixture
plots to their allowed hull.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_leverage
   :no-index:

.. code-block:: python

   contour, surface = design.plot_leverage(
       ax1="Temperature",
       ax2="Pressure",
       constant_levels={"Time": 30.0},
       coded=False,
       domain="allowed",
   )

Response surfaces
^^^^^^^^^^^^^^^^^

``plot_response`` requires a fitted model. Set ``second_response`` to overlay a
second fitted response. ``feasible_region=True`` displays the region satisfying
the limits previously configured with ``set_response_conditions()``; it is
different from ``domain="allowed"``, which controls which factor combinations
are drawn.

The optional confidence correction is conservative. The confidence half-width
is subtracted for responses configured for maximization and added for responses
configured for minimization, so response conditions are required. The
``"residuals"`` mode uses residual mean square; ``"replicates"`` uses pure error
and consequently also requires replicated runs and lack-of-fit statistics.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_response

.. code-block:: python

   contour, surface = design.plot_response(
       ax1="Temperature",
       ax2="Pressure",
       response="Yield",
       second_response="Purity",
       constant_levels={"Time": 30.0},
       feasible_region=True,
       domain="allowed",
   )

Confidence half-width
^^^^^^^^^^^^^^^^^^^^^

This plot shows the confidence half-width of the estimated mean response,
``t * sqrt(MS * leverage)``. It is not a prediction interval for a future
observation because no additional observation-error term is included.
``alpha`` is the significance level, so ``alpha=0.05`` produces a 95% confidence
level.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_confidence_interval

.. code-block:: python

   contour, surface = design.plot_confidence_interval(
       ax1="Temperature",
       ax2="Pressure",
       response="Yield",
       type="residuals",  # or "replicates"
   )

Model diagnostics
-----------------

Diagnostic plots require a fitted model. ``plot_diagnostics`` selects the view
through ``view``; supported values are ``"overview"``,
``"observed_vs_predicted"``, ``"residuals_vs_fitted"``,
``"residuals_by_run"``, ``"qq"``, and ``"histogram"``. Set ``cv=True`` to use
leave-one-out cross-validation predictions and residuals. Omitting ``response``
adds a response selector to every view.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_diagnostics

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_regression_coefficients

.. code-block:: python

   diagnostics = design.plot_diagnostics(response="Yield", cv=True)
   coefficients = design.plot_regression_coefficients(response="Yield")
   residuals_by_run = design.plot_diagnostics(
       response="Yield",
       view="residuals_by_run",
       x_axis="exp_order",
   )
   normality = design.plot_diagnostics(response="Yield", view="qq")

Model effects
-------------

Process and categorical factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Main-effect plots vary one factor while holding the others at their standard
reference settings. Interaction plots vary a pair of factors; visibly
non-parallel profiles indicate that the fitted effect of one factor depends on
the other. These methods evaluate the fitted model and are not raw response
summaries.

Omit ``factor`` or ``factors`` to build the complete dashboard. Supply one
factor or one two-factor tuple to return a focused figure. Both methods reject
mixture components; use ``plot_mixture_trace`` for mixture models.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_main_effects

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_interactions

.. code-block:: python

   all_main_effects = design.plot_main_effects("Yield", coded=False)
   temperature_effect = design.plot_main_effects(
       "Yield",
       factor="Temperature",
       coded=False,
   )
   selected_interaction = design.plot_interactions(
       "Yield",
       factors=("Temperature", "Pressure"),
   )

Mixture traces
^^^^^^^^^^^^^^

A mixture trace varies each component along a bound-respecting path through the
centroid reference blend. The remaining mixture components are adjusted using
their available capacity so that their bounds are respected and the mixture sum
remains one. Any process or categorical factors are held at their standard
reference settings.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_mixture_trace

.. code-block:: python

   trace = design.plot_mixture_trace(
       response="Viscosity",
       reference="centroid",
       n_points=75,
   )

Model-validation plots
----------------------

These plots use grouped confirmation runs loaded through
``import_confirmation_results()`` and predictions from the fitted model. See
:doc:`Model validation <model_validation>` for the complete workflow and the
formulas used for validation intervals.

.. automethod:: doetools.graphs.plot_api_mixin.GraphsMixin.plot_confirmation
   :noindex:

.. code-block:: python

   comparison = design.plot_confirmation(
       response="Yield", view="observed_vs_predicted"
   )
   residuals = design.plot_confirmation(response="Yield", view="residuals")

Pareto-front plots
------------------

Call ``compute_pareto_front()`` before plotting. Omitting ``x`` and ``y`` uses
the first two responses; supply both names to choose another pair. Adding ``z``
creates a 3D plot. Configured response limits are overlaid on 2D plots.

.. automethod:: doetools.utils.pareto.ParetoMixin.plot_pareto_front

.. code-block:: python

   design.compute_pareto_front()
   pareto_2d = design.plot_pareto_front(x="Yield", y="Cost")
   pareto_3d = design.plot_pareto_front(
       x="Yield",
       y="Cost",
       z="Purity",
   )
