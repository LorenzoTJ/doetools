.. _export_design:

Export Experiments
==================

In this section, we cover how to export the experimental design matrix for execution in the laboratory.
The :meth:`export_experiments` method writes a CSV file by default, with
options for randomization and response columns. Use a ``.xlsx`` destination to
write an Excel workbook instead. The extension selects the format; supported
extensions are ``.csv`` and ``.xlsx`` (case-insensitive). Omitting the destination
writes ``design_matrix.csv``.

Both identifiers start at 1. ``Run`` is the laboratory execution order, while
``Idx`` is the stable identifier of a row in the internal design. For example,
randomized output may begin with ``Run=1, Idx=3`` followed by ``Run=2, Idx=1``.
The internal pandas matrix indices remain zero-based. The names ``Run`` and
``Idx`` are reserved and cannot be used for exported factors or responses.

If you plan to import measured results with :meth:`import_responses`, define the
response names when exporting. Those names are retained by the design and identify
the response columns expected during import. Unmeasured responses are exported
as empty cells in CSV and XLSX, using missing values internally rather than
zeros. A measured zero remains a valid response value.

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

Export a design matrix to CSV or XLSX for laboratory execution.

.. note::
    ``Run`` is the planned sequence in which to perform the runs. Follow it
    during laboratory execution; it is randomized when ``randomize=True``.

    ``Idx`` is the stable identifier of the original internal design row. Do
    not edit it: :meth:`import_responses` uses it to associate measured responses
    with the correct run and restore the internal design order.

.. automethod:: doetools.utils.abstract_design.Design.export_experiments

**Example:**

.. code-block:: python

   # Export with randomization for blind experiments
   design.export_experiments(
       responses=["Yield", "Purity", "Cost"],
       randomize=True,
       destination="experiment_plan.csv",
       coded=False  # Use actual factor units
   )


Excel output is available by specifying an XLSX destination:

.. code-block:: python

   design.export_experiments(
       responses=["Yield", "Purity", "Cost"],
       destination="experiment_plan.xlsx",
   )

CSV files use UTF-8, a comma separator, and a decimal point. Both formats contain
column headers without an additional pandas index column. Excel support is
included in the standard installation; legacy ``.xls`` files are not supported.
