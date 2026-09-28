.. _designs-mixture-constrained:

Constrained Mixture Design
==========================

Constrained mixture designs explore a feasible mixture region defined by lower
and upper bounds on each component. Every generated point satisfies the component
bounds and the constant-sum condition :math:`\sum x_i = 1`.

**When to use:**

* The mixture contains three or four components
* One or more components have practical lower or upper bounds
* You want to explore the vertices, edges, and faces of the feasible region

The default ``structure="complete"`` includes all supported boundary structures.
Alternatively, pass ``"vertices"`` or a collection containing ``"vertices"``,
``"edge_midpoints"``, and ``"face_centroids"``.

Class Reference
---------------

.. autoclass:: doetools.ConstrainedMixtureDesign
    :no-members:

|

Example Usage
-------------

**Three-component formulation with upper bounds**

.. code-block:: python

   from doetools import ConstrainedMixtureDesign, MixtureFactor

   factors = {
       "Component_A": MixtureFactor(lower_bound=0.0, upper_bound=0.7),
       "Component_B": MixtureFactor(lower_bound=0.1, upper_bound=0.8),
       "Component_C": MixtureFactor(lower_bound=0.1, upper_bound=0.7),
   }

   design = ConstrainedMixtureDesign(
       factors=factors,
       structure="complete",
   )

   design_matrix = design.get_design_matrix()

**Selecting only vertices and edge midpoints**

.. code-block:: python

   design = ConstrainedMixtureDesign(
       factors=factors,
       structure={"vertices", "edge_midpoints"},
       replicates=1,
   )

