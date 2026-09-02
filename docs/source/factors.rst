.. _factors:

Factors
=========== 

*doetools* factor classes for Design of Experiments (DoE).

This module provides three types of factors commonly used in experimental design:

* **CategoricalFactor**: represent qualitative variables that assume only a finite set of discrete levels. Examples include the type of catalyst, the solvent used, or the reactor type.
* **ContinuousFactor**: are variables that can take an infinite number of values within a given interval, neglecting instrumental resolution. Typical examples of continuous factors include temperature, pressure, pH, and reaction time.
* **MixtureFactor**: Unlike process factors, mixture factors cannot vary independently because their values are constrained by a constant-sum condition. They represent the proportions of components in a mixture, such as the composition of a solvent blend or the ratio of components in a formulation.

Each factor type handles automatic level generation and coding for use in DoE matrices.

|

CategoricalFactor
-----------------

.. autoclass:: doetools.CategoricalFactor
    :no-members:

| 

ContinuousFactor
-----------------

.. autoclass:: doetools.ContinuousFactor
    :no-members:

|

MixtureFactor
-----------------

.. autoclass:: doetools.MixtureFactor
    :no-members:


Example Usage
-----------------
Here is an example of how to define factors for a DoE:

.. code-block:: python

    # Basic Import of the factor classes
    from doetools import ContinuousFactor, CategoricalFactor, MixtureFactor

    # Define factors for a process design with only continuous factors
    factors = {"Temperature": ContinuousFactor(n_levels= 3, lower_bound=50, upper_bound=150, decimals=1),
               "Pressure": ContinuousFactor(n_levels=4, lower_bound=1, upper_bound=10, decimals=2),
               "pH": ContinuousFactor(n_levels=3, lower_bound=0, upper_bound=14, decimals=0)}

    # Define factors for a process design with a two continuous factors and a categorical factor
    factors = {"Pressure" : ContinuousFactor(n_levels=4, lower_bound=1, upper_bound=10),
               "Time" : ContinuousFactor(n_levels=3, lower_bound=10, upper_bound=60),
               "Catalyst" : CategoricalFactor(levels=["A", "B", "C"], reference_level="B")}

    # Define factors for a mixture design with three components
    factors = {"Component A" : MixtureFactor(lower_bound=0.2, upper_bound=1),
               "Component B" : MixtureFactor(lower_bound=0.2, upper_bound=1),
               "Component C" : MixtureFactor(lower_bound=0.2, upper_bound=1)}

For categorical factors, ``reference_level`` selects the operational default used
for center points, fixed plot levels, and prediction profiles. If it is omitted,
the first item in ``levels`` is used. Selecting a reference level does not reorder
or otherwise change the numerical coding of the levels.
