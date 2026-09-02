.. _leverage_analysis:

Leverage Analysis
=================

Leverage describes how strongly a factor setting can influence the estimated
model coefficients. It is determined by the experimental design and the terms in
the configured model matrix; it does not depend on measured responses. You can
therefore examine leverage before importing results or fitting a response model.

For the model-matrix row :math:`x_i` corresponding to run :math:`i` and the full
model matrix :math:`X`, doetools computes

.. math::

   h_i = x_i^{\mathsf{T}} (X^{\mathsf{T}}X)^{+} x_i,

where :math:`+` denotes the Moore--Penrose pseudoinverse. The leverage values sum
to the rank of :math:`X`, and their average is
:math:`\operatorname{rank}(X)/n` for :math:`n` runs.

Interpretation
--------------

Higher leverage means that a setting has greater influence on coefficient
estimation. Interpret values relative to the other points in the same design and
for the same model terms: changing the model changes the leverages. A
high-leverage point is not necessarily erroneous or an outlier.

Leverage values are directly related to model uncertainty. The variance of the 
predicted response at setting :math:`x_0` is proportional to the square root of the leverage 
:math:`\sqrt{h_0}` of that setting. 

Obtain and plot leverage
------------------------

.. automethod::  doetools.utils.abstract_design.Design.get_leverages

.. automethod::  doetools.graphs.plot_api_mixin.GraphsMixin.plot_leverage

Example
-------

Define the model terms first. Imported responses and a fitted model are not
required.

.. code-block:: python

   from doetools import ModelTerms

   design.set_model_terms(
       ModelTerms(
           intercept=True,
           pro_main="all",
           pro_int2=None,
           pro_quadratic=None,
       )
   )

   # One leverage value for each experimental run
   leverages = design.get_leverages()

   # Leverage across a two-factor slice of the design space
   contour, surface = design.plot_leverage(
       ax1="Temperature",
       ax2="Pressure",
       constant_levels={"Time": 30.0},
       coded=False,
       domain="allowed",
   )

``ax1`` and ``ax2`` are the free factors. For mixture surfaces, pass the third
free component as ``ax3``. Use ``constant_levels`` to fix factors that are not
shown; its values are expressed in actual units. The grid bounds remain coded,
while ``coded`` controls whether plot labels use coded or actual units.
``domain="allowed"`` applies saved process-domain filters or restricts a mixture
plot to its allowed hull.
