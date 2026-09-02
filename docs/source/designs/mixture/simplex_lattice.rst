.. _designs-mixture-simplexlattice:

Simplex Lattice Design
=======================

Simplex Lattice designs provide systematic and complete coverage of the mixture
simplex space. By specifying a degree parameter, you control the granularity of
the lattice grid.

**When to use:**

* Components form a mixture (sum to 100% or fixed total)
* You need systematic coverage of the mixture space

**Key characteristics:**

* Minimum of 2 mixture components
* Degree parameter m controls lattice resolution
* Each component takes values in an unconstrained simplex: 0, 1/m, 2/m, ..., (m-1)/m, 1
* Number of runs: C(q+m-1, m) for q components and degree m

| 

.. note::
    For more information on simplex lattice designs, see the 
    `NIST Engineering Statistics Handbook <https://www.itl.nist.gov/div898/handbook/pri/section5/pri52.htm>`_.
.. note::
    For a three- or four-component formulation that requires lower or upper
    bounds, use the :ref:`Constrained Mixture Design
    <designs-mixture-constrained>`.

|

Class Reference
---------------

.. autoclass:: doetools.SimplexLatticeDesign
    :no-members:

|

Example Usage
-------------

**Three-component mixture with {3,3} lattice**

.. code-block:: python

   from doetools import SimplexLatticeDesign, MixtureFactor
   
   # Define 3 mixture components
   factors = {
       'Component_A': MixtureFactor(lower_bound=0.0, upper_bound=1),
       'Component_B': MixtureFactor(lower_bound=0.0, upper_bound=1),
       'Component_C': MixtureFactor(lower_bound=0.0, upper_bound=1)
   }
   
   # Create {3,3} simplex lattice design
   design = SimplexLatticeDesign(factors=factors, m=3, center_points=2)
   
   print(f"Runs: {len(design.get_design_matrix())}")
   # C(3+3-1, 3) = C(5,3) = 10 lattice points + 2 center = 12 runs
   # Each component: 0, 1/3, 2/3, 1

