.. _import_responses:

Import Results
==============

Once the experimental design has been executed in the laboratory, the next step is to import the observed responses back into *doetools* for analysis and modeling.
The :meth:`import_responses` method loads response data from a pandas DataFrame or
an Excel/CSV file and matches it to the corresponding runs in the design matrix.

import_responses
^^^^^^^^^^^^^^^^

Import experimental results after lab execution.

.. note::
    Export the design with a ``responses`` list before importing results. The
    imported file must contain all of those named response columns and the
    unmodified ``Idx`` column.

``Idx`` must contain each integer from 1 to the number of experiments exactly
once. Results are matched by ``Idx``, regardless of file row order. ``Run`` is
optional; when present, it must also be a permutation of 1 through N and is
retained as execution-order metadata.

Migrating older files
---------------------

The old ``Exp. Order`` and ``Exp. Idx`` headers are rejected. For files exported
with the previous zero-based numbering, rename both columns and add 1 to their
values before importing:

.. code-block:: python

   import pandas as pd

   completed = pd.read_csv("old_results.csv")
   completed = completed.rename(columns={"Exp. Order": "Run", "Exp. Idx": "Idx"})
   for column in ("Run", "Idx"):
       if column in completed:
           completed[column] = completed[column] + 1
   design.import_responses(completed)

The same migration applies to existing-design and confirmation-run inputs.

Examples
--------

.. automethod:: doetools.utils.abstract_design.Design.import_responses

**Example:**

.. code-block:: python

   # Import completed experiments
   design.import_responses(source="completed_experiments.csv")

An in-memory table can be passed directly and is copied defensively:

.. code-block:: python

   design.import_responses(source=completed_experiments)

The extension selects the reader. Excel workbooks (``.xlsx``) are also accepted:

.. code-block:: python

   design.import_responses(source="completed_experiments.xlsx")

For CSV files with a different separator, decimal convention, or encoding, load
with pandas and pass the resulting DataFrame:

.. code-block:: python

   import pandas as pd

   completed = pd.read_csv("completed_experiments.csv", sep=";", decimal=",")
   design.import_responses(source=completed)

The same CSV/XLSX and DataFrame inputs are supported by ``ImportDesign``,
``DOptAddDesign``, ``load_prediction_points``, and ``load_confirmation_runs``.
