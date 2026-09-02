.. _import_responses:

Import Results
==============

Once the experimental design has been executed in the laboratory, the next step is to import the observed responses back into *doetools* for analysis and modeling.
The :meth:`import_responses` method loads response data from Excel or CSV files and
matches it to the corresponding runs in the design matrix.

import_responses
^^^^^^^^^^^^^^^^

Import experimental results after lab execution.

.. note::
    Export the design with a ``responses`` list before importing results. The
    imported file must contain all of those named response columns and the
    unmodified ``Exp. Idx`` column.

.. automethod:: doetools.utils.abstract_design.Design.import_responses

**Example:**

.. code-block:: python

   # Import completed experiments
   design.import_responses(file_path="completed_experiments.xlsx")  
