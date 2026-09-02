.. _designs-process:

Process Designs
===============

Process designs are experimental designs used to study the relationship between process factors 
(such as temperature, pressure, concentration, time) and response variables. These factors are 
typically continuous or categorical variables that can be independently controlled within specified 
ranges or levels.

Process designs are ideal for:

* **Factor screening** - Identifying which factors significantly affect the response among many potential variables
* **Process optimization** - Finding optimal operating conditions to maximize yield, minimize cost, or achieve target specifications
* **Response surface modeling** - Building mathematical models to predict outcomes and understand factor interactions

*doetools* provides a comprehensive suite of different process designs, from efficient screening with few runs to detailed 
response surface exploration.

|

Available Process Designs
-------------------------

The process designs supported by *doetools* include:

* :ref:`Full Factorial Design <designs-process-fullfactorial>`
* :ref:`Fractional Factorial Design <designs-process-fractionalfactorial>`
* :ref:`Plackett-Burman Design <designs-process-plackettburman>`
* :ref:`Central Composite Design <designs-process-centralcomposite>`
* :ref:`Box-Behnken Design <designs-process-boxbehnken>`

.. toctree::
   :maxdepth: 1
   :hidden:

   fullfactorial
   fractionalfactorial
   plackettburman
   centralcomposite
   boxbehnken   