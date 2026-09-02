.. _designs-process-plackettburman:

Plackett-Burman Design
======================

Plackett-Burman designs are highly efficient screening designs used to identify
the most important factors among a large number of potential factors with minimal
experimental runs. The design columns are orthogonal and factors are balanced. 
They are ideal for early-stage screening.

**When to use:**

* You need to screen many factors (supporting 4-24) economically
* Only main effects are of interest (interactions assumed negligible)
* You want to maximize factors studied with minimal runs

**Key characteristics:**

* Requires 2-level factors only
* Supports 4 to 24 factors efficiently
* Number of runs is always a multiple of 4
* Estimates main effects only (interactions confounded)
* Highly economical: n factors can be screened in n+1 runs (e.g. 7 factors in 8 runs)
* Dummy factors automatically added if needed

.. note::
    For more information on Plackett-Burman designs, see the 
    `NIST Engineering Statistics Handbook <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm>`_.

|

Class Reference
---------------

.. autoclass:: doetools.PlackettBurmanDesign
    :no-members:

|

Example Usage
-------------

**Screening 7 factors in 8 runs**

.. code-block:: python

   from doetools import PlackettBurmanDesign, ContinuousFactor
   
   # Define 7 factors to screen
   factors = {
       'Temperature': ContinuousFactor(n_levels=2, lower_bound=50, upper_bound=90),
       'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=3),
       'Time': ContinuousFactor(n_levels=2, lower_bound=30, upper_bound=90),
       'pH': ContinuousFactor(n_levels=2, lower_bound=4, upper_bound=8),
       'Catalyst': ContinuousFactor(n_levels=2, lower_bound=0.1, upper_bound=1.0),
       'Stirring': ContinuousFactor(n_levels=2, lower_bound=100, upper_bound=500),
       'Concentration': ContinuousFactor(n_levels=2, lower_bound=0.5, upper_bound=2.0)
   }
   
   # Create Plackett-Burman design
   design = PlackettBurmanDesign(factors=factors)
   # Only 8 runs to screen 7 factors!
   # Compare to 2^7 = 128 runs for full factorial
    
