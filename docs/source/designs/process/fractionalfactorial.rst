.. _designs-process-fractionalfactorial:

Fractional Factorial Design
============================

Fractional factorial designs use a carefully chosen subset of the full factorial
design to reduce the number of experimental runs while maintaining the ability
to estimate important effects. These designs are particularly useful for screening
many factors efficiently.

**When to use:**

* You need to screen many factors (typically 4-5 or more)
* Resources are limited and you need to minimize runs
* Main effects and low-order (two-factor) interactions are of primary interest
* Higher-order interactions are assumed negligible

**Key characteristics:**

* Requires 2-level factors only
* Uses fraction of full factorial (e.g., half-fraction: 2^(k-1))
* Resolution determines which effects are confounded:
  
  * **Resolution III**: Main effects clear, but confounded with 2-factor interactions
  * **Resolution IV**: Main effects clear of 2-factor interactions, but 2-factor interactions may be confounded with each other
  * **Resolution V**: Main effects and two-factor interactions are clear of each other but may be aliased with three-factor interactions.

.. note::
    For more information on fractional factorial designs, see the 
    `NIST Engineering Statistics Handbook <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm>`_.

    `doetools` generates fractional factorial designs using the
    `doe-toolbox <https://pypi.org/project/doe-toolbox/>`_ package. It uses
    `doe_box.fracfactgen` to select a generator for the requested resolution,
    then `doe_box.fracfact` to create the coded two-level design matrix.

Class Reference
---------------

.. autoclass:: doetools.FractionalFactorialDesign
    :no-members:

| 

Example Usage
-------------

**Resolution IV design for 5 factors**

.. code-block:: python

   from doetools import FractionalFactorialDesign, ContinuousFactor
   
   # Define 5 two-level factors
   factors = {
       'Temperature': ContinuousFactor(n_levels=2, lower_bound=60, upper_bound=80),
       'Pressure': ContinuousFactor(n_levels=2, lower_bound=1.0, upper_bound=2.0),
       'Time': ContinuousFactor(n_levels=2, lower_bound=30, upper_bound=60),
       'pH': ContinuousFactor(n_levels=2, lower_bound=5.0, upper_bound=7.0),
       'Catalyst': ContinuousFactor(n_levels=2, lower_bound=0.1, upper_bound=0.5)
   }
   
   # Create Resolution IV design
   design = FractionalFactorialDesign(factors=factors, resolution=4)

    