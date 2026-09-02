.. _designs-mixture-simplexcentroid:

Simplex Centroid Design
========================

Simplex Centroid designs are efficient designs for mixture experiments
where the components must sum to a constant 1.0. The design contains the 
centroids of every non-empty subset of components, including simplex vertices,
edge midpoints, higher-dimensional face centroids, and the overall centroid,
providing balanced coverage of the mixture simplex.

**When to use:**

* Components form a mixture (sum to 100% or fixed total)
* You want to screen mixture effects efficiently
* Equal blend proportions are practical and meaningful

**Key characteristics:**

* Minimum of 2 mixture components
* Tests simplex vertices
* Tests equal-proportion binary, ternary, and higher blends
* Number of runs: 2^q - 1 for q components (excluding replicates)

|

.. note::
    For more information on simplex centroid designs, see the 
    `NIST Engineering Statistics Handbook <https://www.itl.nist.gov/div898/handbook/pri/section5/pri5.htm>`_.

.. note::
    For a three- or four-component formulation that requires lower or upper
    bounds, use the :ref:`Constrained Mixture Design
    <designs-mixture-constrained>`.


Class Reference
---------------

.. autoclass:: doetools.SimplexCentroidDesign
    :no-members:

|

Example Usage
-------------

**Three-component mixture screening**

.. code-block:: python

   from doetools import SimplexCentroidDesign, MixtureFactor
   
   # Define 3 mixture components
   factors = {
       'Polymer_A': MixtureFactor(lower_bound=0.1, upper_bound=1),
       'Polymer_B': MixtureFactor(lower_bound=0.1, upper_bound=1),
       'Polymer_C': MixtureFactor(lower_bound=0.1, upper_bound=1)
   }
   
   # Create simplex centroid design
   design = SimplexCentroidDesign(factors=factors)
   
   print(f"Runs: {len(design.get_design_matrix())}")
   # 2^3 - 1 = 7 design points
   # Points: 3 vertices + 3 binary + 1 centroid
