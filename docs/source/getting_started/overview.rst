.. _overview:

Quick Start Guide
=================

This guide provides a quick introduction to *doetools* and demonstrates the basic workflow for creating and analyzing an experimental design.

What is doetools?
-----------------

*doetools* is a comprehensive Python package for design of experiments (DoE). It covers the complete DoE workflow:

1. **Design Suggestion** - The built-in design advisor recommends feasible designs based on high-level requirements;
2. **Design Creation** - Generate experimental design matrices for process and mixture factors;
3. **Data Collection** - Export design matrices for laboratory execution (CSV/Excel);
4. **Analysis** - Fit regression models (MLR) and analyze results with regression statistics and ANOVA;
5. **Visualization** - Create interactive plots (leverage, residual analysis, confidence intervals, main effects, interaction plots) and response surfaces (contour, 3D) in Plotly;
6. **Optimization** - Identify optimal conditions using pareto front analysis and response surface feasible region plotting (for both single and multiple responses);
7. **Validation** - Perform confirmation runs to validate the model and assess its predictive power;
8. **Reporting** - Generate comprehensive PDF reports

Supported Design Types
----------------------

*doetools* supports a wide range of **design types** for both process factors (temperature, pressure, concentration) and mixture factors (component proportions).

- **Process Designs**: Full Factorial, Fractional Factorial, Plackett-Burman, Box-Behnken, and Central Composite Designs;
- **Mixture Designs**: Simplex Lattice, Simplex Centroid, Constrained Mixture Designs;
- **Optimal Designs**: D-Optimal, D-Optimal Augmentation;
- **Import Designs**: Import existing data for analysis;

Supported Plots
---------------
*doetools* provides a variety of **interactive plots** to help you analyze and visualize your experimental data:

- **Design Visualization**: 2D and 3D scatter plots of design points;
- **Leverage Plots**: Identify influential points in your design (contour and 3D);
- **Coefficient Plots**: Visualize model coefficients with confidence intervals;
- **Residual Analysis**: Analyze the residuals of the fitted model;
- **Confidence Interval Plots**: Show confidence intervals for predictions;
- **Response Surfaces**: Contour and 3D surface plots for visualizing response behavior;
- **Pareto plots**: Identify good compromises for multi-response optimization;

Basic Workflow
--------------

The typical doetools workflow consists of four main steps:

1. **Design Selection and Creation**: Define your factors and create a design matrix (optional: use the design advisor);
2. **Data Collection**: Export the design matrix for execution in the lab;
3. **Model Fitting**: Import results and fit a regression model;
4. **Analysis and Visualization**: Analyze the fitted model and generate plots and reports;

.. code-block:: python

    # Import the doetools package and specific modules
    import doetools
    from doetools import ContinuousFactor as CF
    from doetools import FullFactorialDesign, ModelTerms

    # 1. Design Selection and Creation
    factors = {"Temperature": CF(n_levels=3, lower_bound=60, upper_bound=80),
               "Pressure": CF(n_levels=3, lower_bound=1, upper_bound=3),
               "Concentration": CF(n_levels=3, lower_bound=0.1, upper_bound=0.5)}
    design = FullFactorialDesign(factors)

    # 2. Data Collection
    design.export_experiments(responses = ["Yield"],
                              randomize=True,
                              save_path="design_matrix.xlsx")
    design.import_responses(file_path="experiments_with_responses.xlsx")

    # 3. Model Fitting
    design.set_model_terms(terms=ModelTerms())
    design.compute_mlr_model()

    # 4. Analyze the fitted model
    design.get_model_summary_pdf(filename="design_report.pdf")
    design.plot_regression_coefficients()
    design.plot_response(ax1="Temperature", ax2="Pressure", response="Yield")

Next Steps
----------

Now that you understand the basics:

* Explore :doc:`../factors` to learn about factor types in detail

* Review design-specific pages for your experimental scenario:

  * :doc:`../designs/process/index`
  * :doc:`../designs/mixture/index`
  * :doc:`../designs/optimal/index`

* Follow the experimental workflow, starting with :doc:`../leverage_analysis`
  and :doc:`../export_experiments`
* See complete real examples in :doc:`../examples/index`
