.. _designs-optimal-doptimal-augmentation:

D-Optimal Augmentation
======================

D-optimal augmentation imports existing experiments and selects additional
candidate runs that maximize the information of the combined design. Existing
runs remain fixed during optimization; only the additional candidates participate
in the exchange search.

**When to use:**

* Existing experiments must be extended for a larger or more complex model
* The new candidate domain differs from the original experimental domain

Workflow
--------

#. Import existing experiments with :class:`~doetools.DOptAddDesign`.
#. Define the model with :meth:`~doetools.DOptAddDesign.set_model_terms`.
#. Generate candidates with :meth:`~doetools.DOptAddDesign.generate_cp`.
#. Optionally inspect them with
   :meth:`~doetools.DOptAddDesign.plot_candidate_set`.
#. Optimize additional-run counts with
   :meth:`~doetools.DOptAddDesign.compute_d_optimal`.
#. Finalize one solution with :meth:`~doetools.DOptAddDesign.select_design`.
#. Optionally export the combined design with
   :meth:`~doetools.DOptAddDesign.export_experiments`.

Class and method reference
--------------------------

.. autoclass:: doetools.DOptAddDesign
    :no-members:

.. automethod:: doetools.DOptAddDesign.set_model_terms

.. automethod:: doetools.DOptAddDesign.generate_cp

.. automethod:: doetools.DOptAddDesign.plot_candidate_set

.. automethod:: doetools.DOptAddDesign.compute_d_optimal

.. automethod:: doetools.DOptAddDesign.select_design

.. automethod:: doetools.DOptAddDesign.export_experiments

.. autoattribute:: doetools.DOptAddDesign.log_det

Existing-run policy
-------------------

The default ``existing_runs_policy="within_candidate_domain"`` retains only
existing rows satisfying the currently declared continuous and mixture bounds
and all active filters. ``existing_runs_policy="all"`` retains every valid
imported row, even when it lies outside the new candidate domain. Such continuous
values are coded outside ``[-1, 1]`` when they exceed the declared bounds.

Candidate settings already present among the imported runs remain eligible by
default. Set ``allow_existing_replicates=False`` to remove all such settings from
the additional-run candidate set.

Existing categorical values must be declared by their
:class:`~doetools.CategoricalFactor`. Existing mixture values must lie in
``[0, 1]`` and sum to 1, although they may lie outside the component bounds used
for the new candidate domain when ``existing_runs_policy="all"``.

Process-augmentation example
----------------------------

.. code-block:: python

    from doetools import ContinuousFactor, DOptAddDesign, ModelTerms

    factors = {
        "Temperature": ContinuousFactor(5, 100, 200, decimals=1),
        "Pressure": ContinuousFactor(5, 1.0, 5.0, decimals=1),
        "Time": ContinuousFactor(5, 60, 120, decimals=0),
    }
    design = DOptAddDesign(
        factors=factors,
        source="existing_data.xlsx",
        responses=["Yield"],
    )

    design.set_model_terms(ModelTerms(
        intercept=True,
        pro_main="all",
        pro_int2="all",
        pro_quadratic="all",
    ))

    def operating_constraint(frame):
        return (frame["Temperature"] <= 190) & (frame["Pressure"] <= 4.5)

    design.generate_cp(
        process_strategy="grid",
        filters=[operating_constraint],
        existing_runs_policy="within_candidate_domain",
        allow_existing_replicates=True,
    )
    candidate_figure = design.plot_candidate_set("Temperature", "Pressure")
    solutions_figure = design.compute_d_optimal(
        n_min=1,
        n_max=20,
        random_state=42,
    )
    design.select_design(5)
    design.export_experiments(
        responses=["Yield"],
        destination="augmented_design.xlsx",
    )

``source`` can also be a pandas DataFrame. In either form, ``factors`` remains
the first constructor parameter and the input data are copied defensively.

Imported response values are retained for existing rows during export. Response
cells for the selected new runs are initialized to zero.
