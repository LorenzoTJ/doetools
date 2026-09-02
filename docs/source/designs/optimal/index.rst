.. _designs-optimal:

Optimal Designs
===============

Optimal designs are computer-generated experimental designs that maximize specific 
statistical criteria, most commonly D-optimality (maximizing the determinant of 
the information matrix X'X). Unlike classical designs with fixed geometric structures, 
optimal designs are flexible and can accommodate irregular experimental regions and complex constraints. 

Optimal designs are ideal for:

* **Complex constraints** - Handling irregular experimental regions with boundaries that don't fit classical design geometries
* **Augmentation** - Extending existing experimental data efficiently by adding strategically selected new runs
* **Model-specific optimization** - Creating designs tailored to fit specific model terms with maximum statistical efficiency
* **Mixed variable types** - Accommodating continuous, categorical, and mixture factors in the same design

Optimal designs use iterative algorithms to select the best 
subset of experimental runs from a candidate set. *doetools* provides D-optimal design generation 
for both new experiments and augmentation of already collected data.

|

Available Optimal Designs
-------------------------

The optimal designs supported by *doetools* include:

* :ref:`D-Optimal Design <designs-optimal-doptimal>`
* :ref:`D-Optimal Augmentation <designs-optimal-doptimal-augmentation>`

.. note::
   Select D-Optimal Augmentation when you have existing experimental data and want to add new runs that maximize the overall design efficiency.

.. toctree::
   :maxdepth: 1
   :hidden:

   doptimal
   doptimal_augmentation
