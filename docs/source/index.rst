*doetools* Documentation
====================================

*doetools* is a Python package for *Design of Experiments (DoE)*, supporting the
full workflow from design generation to modelling, diagnostics, visualisation and validation.

It includes **factorial**, **screening**, **response-surface**, **mixture**, and **D-optimal designs** for
constrained or complex experimental regions.

*doetools* helps users plan efficient experiments and turn results into actionable insights for process optimisation.

Key Features
-----------------

* **Design Reccomendations** from high-level inputs;
* **Design Generation** for process and mixture factors;
* **Model Fitting** with ordinary least squares (OLS);
* **Regression analysis** and **interactive visualizations** in Plotly;
* **Multi-objective optimisation**;
* **Model validation** through confirmation runs;
* **PDF reports** for a comprehensive summary of the experimental workflow.

Getting Started
-----------------

To get started with *doetools*, please refer to the following sections:

* :doc:`getting_started/installation` - Instructions for installing doetools
* :doc:`getting_started/overview` - A quick start guide to using doetools
* :doc:`examples/index` - Example notebooks demonstrating the use of doetools

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/overview

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Design Creation
   :hidden:

   factors
   design_advisor
   designs/import_design
   designs/process/index
   designs/mixture/index
   designs/optimal/index

.. toctree::
   :maxdepth: 1
   :caption: Experimental Workflow
   :hidden:

   leverage_analysis
   export_experiments
   import_responses
   model_computation
   prediction
   model_validation
   common_getters
   plots
   pdf_reports

