.. _designs-process-centralcomposite:

Central Composite Design
=========================

Central Composite Designs (CCDs) are the most popular response surface designs
for fitting second-order polynomial models. They consist of factorial points,
axial (star) points, and center points, enabling estimation of quadratic effects
and interactions with fewer runs than a 3-level factorial.

**When to use:**

* You need to fit quadratic (second-order) models
* Optimizing a process with continuous factors
* Building response surface models for prediction
* Factors are continuous and can be varied beyond factorial levels

**Key characteristics:**

* Requires continuous factors only (minimum 2 factors)
* Three types of points: factorial, axial, and center
* Number of runs: 2^k + 2k + center points
* Three design variants:
  
  * **CCC (Circumscribed)**: Axial points outside factorial space, rotatable
  * **CCF (Face-centered)**: Axial points at face centers (±1)
  * **CCI (Inscribed)**: Factorial points scaled inward

.. note::
    For more information on Central Composite designs, see the 
    `NIST Engineering Statistics Handbook <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm>`_.

Class Reference
---------------

.. note::
    The number of levels specified for each factor is not important since the design will generate the appropriate levels based on the design type. The n_levels parameter is ignored for CCDs, but must be provided when creating ContinuousFactor objects.
    It is updated after the design is generated to reflect the actual levels used in the design matrix.

.. autoclass:: doetools.CentralCompositeDesign
    :no-members:

|

Example Usage
-------------

**Circumscribed CCD for 3 factors**

.. code-block:: python

   from doetools import CentralCompositeDesign, ContinuousFactor
   
   # Define continuous factors
   factors = {
       'Temperature': ContinuousFactor(n_levels=5, lower_bound=60, upper_bound=80),
       'Pressure': ContinuousFactor(n_levels=5, lower_bound=1.0, upper_bound=3.0),
       'Time': ContinuousFactor(n_levels=5, lower_bound=30, upper_bound=90)
   }
   
   # Create circumscribed CCD (default, rotatable)
   design = CentralCompositeDesign(factors=factors, design='ccc', center_points=3)
   # 2^3 = 8 factorial points 
   # 2×3 = 6 axial points
   # 3 center points
   # Total = 8 + 6 + 3 = 17 runs