.. _designs-mixture:

Mixture Designs
===============

Mixture designs are specialized experimental designs for situations where the factors 
represent proportions of a mixture that must sum to a constant 1.0 (100%).
Unlike process factors that can be independently varied, mixture components are 
inherently constrained and dependent on each other.

Mixture designs are ideal for:

* **Formulation optimization** - Optimizing blends, composites, alloys, or multi-component systems
* **Blending effects** - Identifying synergistic or antagonistic interactions between mixture components

Mixture designs use particular polynomial models (Scheffé models) where the simplex 
constraint (:math:`\sum x_i = 1`) is built into the model structure. *doetools* provides classical 
mixture designs that efficiently explore the compositional space to identify optimal formulations.

|

Available Mixture Designs
-------------------------

The mixture designs supported by *doetools* include:

* :ref:`Simplex Centroid Design <designs-mixture-simplexcentroid>`
* :ref:`Simplex Lattice Design <designs-mixture-simplexlattice>`
* :ref:`Constrained Mixture Design <designs-mixture-constrained>`

.. toctree::
   :maxdepth: 1
   :hidden:

   simplex_centroid
   simplex_lattice
   constrained_mixture
