# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where it is compatible with Python's versioning rules.

## [Unreleased]

### Added

- Public `export_plot(fig, destination, scale=3, width=None, height=None)` helper
  for high-resolution PNG/JPEG/WebP and SVG/PDF exports, preserving figure layout
  dimensions by default. Install `doetools[export]` for optional Kaleido support;
  Chrome/Chromium is also required. Interactive plotting requires no extra.

### Changed

- Rename `get_model_summary_pdf` to `save_model_summary` without a compatibility
  alias. Replace existing calls with `design.save_model_summary(...)`; the
  `filename` argument, default `design_report.pdf`, PDF output, and `None`
  return value are unchanged.

- Documentation notebooks now use CSV experiment inputs and outputs with
  one-based `Run`/`Idx` identifiers and empty unmeasured responses. D-optimal
  examples use a fixed random seed for reproducibility.

- Unmeasured responses in experimental-plan exports are now empty CSV/XLSX
  cells instead of zeros. D-optimal augmentation preserves historical
  measurements, including zeros and missing values, and leaves new responses
  empty. Model fitting continues to require complete, finite responses.

- Experiment tables now use `Run` for laboratory execution order and `Idx` for
  the stable experiment identifier, both numbered from 1. Internal pandas
  indices remain zero-based. Imports reject the old `Exp. Order` and `Exp. Idx`
  headers: rename them and add 1 to their previous zero-based values. Diagnostic
  plots and PDF experiment tables now use the same identifiers; PDFs show `Run`
  only when execution metadata was imported.

- Experimental-plan exports now default to `design_matrix.csv` for all designs,
  including D-optimal augmentation. The destination extension selects CSV or XLSX;
  pass an explicit `.xlsx` destination to retain Excel output. CSV uses UTF-8,
  comma-separated fields, and a decimal point. Excel support remains included.

## [0.1.10] - 2026-11-3

- Link updates in the README file

[0.1.10]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.10

## [0.1.9] - 2026-11-1

- Documentation Updates

[0.1.9]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.9

## [0.1.8] - 2026-09 - 28

- Small Documentation Updates

[0.1.8]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.8


## [0.1.7] - 2026-09-17

- Replace the legacy Pareto mixin API with common design methods:
  ``compute_pareto_front``, ``get_pareto_front``, and ``plot_pareto_front``.
  Pareto grids now accept exact continuous and categorical ``factor_levels``,
  reuse bounded-mixture candidates, enforce persistent domain filters, and keep
  response limits as plot-only guides. Pareto plots now use the library's shared
  graph-object styling and can show dominated domain-valid candidates for context.

[0.1.7]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.7


## [0.1.6] - 2026-09-16

- Changed one phrase in the documentation landing page

[0.1.6]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.6

## [0.1.5] - 2026-09-15

- Rename the ``suggest_design`` boolean parameter
  ``non_rectangular_constraints`` to ``constraints``. The old keyword is no
  longer accepted.

[0.1.5]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.5

## [0.1.4] - 2026-09-15

- Modified CHANGELOG 
- Updated landing page of PyPi

[0.1.4]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.4

## [0.1.3] - 2026-09-15

### Statistical definitions and interval API

- Correct RMSE to use `sqrt(SS_res / n)`. RMSE_CV remains `sqrt(PRESS / n)`.
- Add `SD = sqrt(MS)` to all ANOVA rows; residual SD uses residual degrees of
  freedom and is not a model quality metric.
- Replace `plot_confidence_interval` with `plot_interval`. Replace plot options
  `corrected` and `type` with keyword-only `interval` and `variance_source`;
  `pure_error` replaces the old `replicates` variance option. No aliases remain.
- Add confidence intervals for the mean and prediction intervals for one new
  observation to plots and external prediction tables. Prediction tables use
  `PI Lower`/`PI Upper` when `interval="prediction"` is selected.
- Example migration: `plot_response(..., corrected="residuals")` becomes
  `plot_response(..., interval="confidence", variance_source="residuals")`.

[0.1.3]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.3


## [0.1.2] - 2026-09-14

- Changed the project license from MIT to GNU General Public License v3.0 only
  (`GPL-3.0-only`) to align with the `doe-toolbox` dependency.

- ZIP downloads on the Examples index, including each example folder and its    supporting files.

- GitHub logo and repository link beside the documentation's page-source icons.

[0.1.2]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.2

## [0.1.1] - 2026-09-05

- Initial public release of `doetools`.
- Experimental design generation and design recommendations.
- Process, mixture, constrained-mixture, and D-optimal designs.
- Import of externally generated experimental designs.
- OLS model fitting, diagnostics, prediction, and model validation.
- Interactive Plotly visualizations.
- Multi-objective optimisation.
- Excel-based experimental workflows.
- PDF report generation.

[0.1.1]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.1
