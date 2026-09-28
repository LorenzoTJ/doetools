PDF Reports
===========

``get_model_summary_pdf()`` writes a static PDF summary of an experimental
design and its fitted response models. Use it to archive an analysis or share a
consistent report with collaborators. The report contains tables, diagnostic
summaries, and statistical interpretations; it does not embed interactive Plotly
figures or confirmation-run analysis.

Generating a report
-------------------

The usual workflow is to create a design, define the model terms, export and
perform the runs, import the measured responses, fit the model, and then write
the report.

.. code-block:: python

   from doetools import CentralCompositeDesign, ContinuousFactor, ModelTerms

   factors = {
       "Temperature": ContinuousFactor(n_levels=3, lower_bound=60, upper_bound=80),
       "Pressure": ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=3),
   }
   design = CentralCompositeDesign(factors)

   design.set_model_terms(
       ModelTerms(
           intercept=True,
           pro_main="all",
           pro_int2="all",
           pro_quadratic="all",
       )
   )
   design.export_experiments(
       responses=["Yield"],
       destination="design_matrix.xlsx",
   )

   # Perform the experiments, then import the completed workbook.
   design.import_responses(source="completed_experiments.xlsx")
   design.compute_mlr_model()

   design.get_model_summary_pdf(filename="design_report.pdf")

.. automethod:: doetools.utils.summary.DesignSummaryMixin.get_model_summary_pdf
   :noindex:

Report contents
---------------

The report is organized as follows:

- **Design Overview** — design type, run, replicate, center-point, and factor
  summaries. Process/categorical factors and mixture components are shown in
  separate tables.
- **Model Structure** — requested included terms and effective model-matrix
  diagnostics: observations, columns, rank, aliased columns, intercept status,
  regression degrees of freedom, residual degrees of freedom, and model status.
- **General Diagnostics** — VIF with a qualitative flag, run-level leverage and
  average leverage, and configured response conditions.
- **Response Analysis** — one section per response with model metrics,
  coefficients, model and lack-of-fit F-tests with interpretations, ANOVA,
  replicate summary, and observed-versus-predicted values.
- **Report Notes** — only when optional sections could not be populated.
- **Appendix A** — the design matrix in actual factor units. It uses landscape
  pages and splits wide matrices into column groups while repeating the run
  column.
