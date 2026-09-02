.. _designs-process-fullfactorial:

Full Factorial Design
======================

A full factorial design systematically evaluates every combination of the selected factor levels,
enabling the estimation of main effects and interactions between factors. It is best suited to
experiments with relatively few factors and levels, as the number of required runs grows rapidly
with design size—for example, a two-level design requires \(2^k\) runs for \(k\) factors.

**When to use:**

* You need complete information about all main effects and interactions
* The number of factors is manageable (typically 2-4 factors)
* Resources permit testing all factor combinations

**Key characteristics:**

* Explores all possible combinations of factor levels (typically 2 or 3 levels)
* Provides unbiased estimates of all effects
* Can be used for both screening (2 levels) and response surface modeling (3 levels)
* Expensive in terms of experimental runs

|

.. note::
   For more information on full factorial designs, see the 
   `NIST Engineering Statistics Handbook <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm>`_.

|

Class Reference
---------------

.. autoclass:: doetools.FullFactorialDesign
    :no-members:

|

Example Usage
-------------

**Mixed factor types (continuous and categorical)**

.. code-block:: python

   from doetools import FullFactorialDesign, ContinuousFactor, CategoricalFactor
   
   factors = {'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=40),
              'Catalyst': CategoricalFactor(levels=['A', 'B', 'C']),
              'pH': ContinuousFactor(n_levels=2, lower_bound=5.0, upper_bound=7.0)}
   
   # Create design with replicates
   design = FullFactorialDesign(factors=factors, replicates=1)
   
   # Each design point is run twice (original + 1 replicate)
   print(f"Total runs: {len(design.get_design_matrix())}")
   # Total: 2 × 3 × 2 × 2 = 24 runs