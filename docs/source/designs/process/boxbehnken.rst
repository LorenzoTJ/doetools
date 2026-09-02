.. _designs-process-boxbehnken:

Box-Behnken Design
===================

Box-Behnken designs are efficient response surface designs that avoid testing at
the extreme corners of the factor space. All design points are at the midpoints of
edges or at the center, making them ideal when extreme factor combinations are
expensive, dangerous, or physically impossible.

**When to use:**

* You need quadratic models but want to avoid extreme combinations
* Factors are continuous 

**Key characteristics:**

* Requires continuous factors with exactly 3 levels each
* Minimum of 3 factors required
* No runs at the vertices (extreme corners)
* Number of runs: 2k(k-1) + center points
* More economical than 3-level factorial designs

.. note::
    For more information on Box-Behnken designs, see the 
    `NIST Engineering Statistics Handbook <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm>`_.

|

Class Reference
---------------

.. autoclass:: doetools.BoxBehnkenDesign
    :no-members:

|

Example Usage
-------------

**Three-factor Box-Behnken design**

.. code-block:: python

   from doetools import BoxBehnkenDesign, ContinuousFactor
   
   # Define 3 factors with 3 levels each
   factors = {
       'Temperature': ContinuousFactor(n_levels=3, lower_bound=60, upper_bound=80),
       'Pressure': ContinuousFactor(n_levels=3, lower_bound=1.0, upper_bound=3.0),
       'pH': ContinuousFactor(n_levels=3, lower_bound=5.0, upper_bound=7.0)
   }
   
   # Create Box-Behnken design with center points
   design = BoxBehnkenDesign(factors=factors, center_points=3)
   
   print(f"Runs: {len(design.get_design_matrix())}")
   # 2×3×(3-1) = 12 edge points + 3 center = 15 runs
   # Compare to 3^3 = 27 runs for full factorial!