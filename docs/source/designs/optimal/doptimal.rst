.. _designs-optimal-doptimal:

D-Optimal Design
================

A D-optimal design selects a fixed-size subset from a user-defined candidate
set by maximizing the determinant of the model information matrix
:math:`X^\mathsf{T}X`. It is useful when a classical design does not represent
the experimental domain or when process, categorical, and mixture factors must
be combined.

**When to use:**

* The experimental region has irregular or filtered boundaries
* Process, categorical, and mixture factors appear in the same design
* The design must support a specific model under a limited run budget

**Key characteristics:**

* Candidate points are generated when :class:`~doetools.DOptDesign` is created
* Optimization is model-dependent and selects candidates without replacement
* The optimal number of runs should be manually selected from the results

Workflow
--------

#. Define factors and create the candidate set with
   :class:`~doetools.DOptDesign`.
#. Optionally inspect it with
   :meth:`~doetools.DOptDesign.plot_candidate_set`.
#. Define the model with :meth:`~doetools.DOptDesign.set_model_terms`.
#. Optimize one or more run counts with
   :meth:`~doetools.DOptDesign.compute_d_optimal`.
#. Finalize one computed solution with
   :meth:`~doetools.DOptDesign.select_design`.

Candidate generation
--------------------

``process_strategy`` is required whenever continuous or categorical factors are
present. The available strategies are:

* ``"grid"``: Cartesian product of the declared factor levels
* ``"lhs"``: Latin-hypercube points for continuous factors, crossed with all
  categorical levels; requires ``lhs_n_samples``
* ``"ccc"``, ``"ccf"``, and ``"cci"``: central-composite candidates for
  continuous factors, crossed with all categorical levels
* ``"bb"``: Box-Behnken candidates for continuous factors, crossed with all
  categorical levels

When mixture factors are present:

* ``mixture_include="all"`` includes vertices, true edge midpoints, true face centroids, and the feasible-region centroid. A collection can select individual categories.
* ``mixture_grid={"degree": m}`` adds a bounded simplex-lattice grid. For mixed designs, process and mixture candidates are combined by Cartesian product.

Candidate filters receive the complete actual-valued candidate DataFrame and
must return a Boolean :class:`pandas.Series` with one value per row.

Class and method reference
--------------------------

.. autoclass:: doetools.DOptDesign
    :no-members:

.. automethod:: doetools.DOptDesign.plot_candidate_set

.. automethod:: doetools.DOptDesign.set_model_terms

.. note::
    See :class:`~doetools.ModelTerms` for process and mixture model-term options.

.. automethod:: doetools.DOptDesign.compute_d_optimal

.. automethod:: doetools.DOptDesign.select_design

.. autoattribute:: doetools.DOptDesign.log_det

Process-design example
----------------------

.. code-block:: python

    from doetools import ContinuousFactor, DOptDesign, ModelTerms

    factors = {
        "Temperature": ContinuousFactor(
            n_levels=5, lower_bound=60, upper_bound=90
        ),
        "Pressure": ContinuousFactor(
            n_levels=5, lower_bound=1.0, upper_bound=5.0
        ),
        "Time": ContinuousFactor(
            n_levels=3, lower_bound=10, upper_bound=60
        ),
    }

    design = DOptDesign(factors=factors, process_strategy="grid")
    candidate_figure = design.plot_candidate_set("Temperature", "Pressure", "Time")

    design.set_model_terms(ModelTerms(
        intercept=True,
        pro_main="all",
        pro_int2="all",
        pro_quadratic="all",
    ))

    solutions_figure = design.compute_d_optimal(
        n_min=10,
        n_max=25,
        random_state=42,
    )
    design.select_design(n=14)
    design_matrix = design.get_design_matrix()

Bounded-mixture example
-----------------------

.. code-block:: python

    from doetools import DOptDesign, MixtureFactor, ModelTerms

    mixture_factors = {
        "A": MixtureFactor(lower_bound=0.10, upper_bound=0.60),
        "B": MixtureFactor(lower_bound=0.15, upper_bound=0.70),
        "C": MixtureFactor(lower_bound=0.10, upper_bound=0.65),
    }

    mixture_design = DOptDesign(
        factors=mixture_factors,
        mixture_include="all",
    )

    mixture_design.set_model_terms(ModelTerms(
        intercept=False,
        pro_main=None,
        pro_int2=None,
        pro_quadratic=None,
        mix_main="all",
        mix_int2="all",
    ))
    
    mixture_design.compute_d_optimal(
        n_min=6,
        n_max=10,
        random_state=42,
    )
    mixture_design.select_design(8)
