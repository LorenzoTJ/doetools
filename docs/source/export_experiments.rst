.. _export_design:

Export Experiments
==================

In this section, we cover how to export the experimental design matrix for execution in the laboratory.
The :meth:`export_experiments` method writes an Excel workbook (``.xlsx``) with
options for randomization and response columns. CSV export is not supported.

If you plan to import measured results with :meth:`import_responses`, define the
response names when exporting. Those names are retained by the design and identify
the response columns expected during import.

Before exporting the design, you can modify it by adding replicates to improve precision
and enable lack-of-fit testing.

Design Modification
-------------------

add_replicates
^^^^^^^^^^^^^^

Add replicate runs to improve precision and enable lack-of-fit testing. This method supports four types of replicates:

- **Center Point Replicates**: Add replicates at the design center to estimate pure error and detect curvature.
- **Leverage-Based Replicates**: Add replicates at high-leverage points to improve model stability and prediction accuracy in influential regions.
- **Manual Replicates**: Add replicates at specific run indices based on experimental priorities or constraints.
- **All Runs Replicates**: Add replicates to all runs for a fully replicated design (not recommended for large designs).

.. automethod:: doetools.utils.abstract_design.Design.add_replicates

**Example:**

.. code-block:: python

   # Add 3 center point replicates
   design.add_replicates(type="center", n_replicates=3)
   
   # Replicate high-leverage points
   # Add 1 replicate to the top 2 leverage points
   design.add_replicates(type="leverage", n_replicates=2)
   
   # Replicate specific runs
   # Add 1 replicate to runs at indices 0, 5, and 10
   design.add_replicates(type="manual", n_replicates=1, indices=[0, 5, 10])

|

Export Experiments
-------------------

export_experiments
^^^^^^^^^^^^^^^^^^

Export a design matrix to an Excel workbook for laboratory execution.

.. note::
    ``Exp. Order`` is the planned sequence in which to perform the runs. Follow it
    during laboratory execution; it is randomized when ``randomize=True``.

    ``Exp. Idx`` is the stable identifier of the original internal design row. Do
    not edit it: :meth:`import_responses` uses it to associate measured responses
    with the correct run and restore the internal design order.

.. automethod:: doetools.utils.abstract_design.Design.export_experiments

**Example:**

.. code-block:: python

   # Export with randomization for blind experiments
   design.export_experiments(
       responses=["Yield", "Purity", "Cost"],
       randomize=True,
       destination="experiment_plan.xlsx",
       coded=False  # Use actual factor units
   )


