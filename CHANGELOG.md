# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where it is compatible with Python's versioning rules.

## [Unreleased]

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


### Changed

## [0.1.2] - 2026-09-14

- Changed the project license from MIT to GNU General Public License v3.0 only
  (`GPL-3.0-only`) to align with the `doe-toolbox` dependency.

- ZIP downloads on the Examples index, including each example folder and its    supporting files.

- GitHub logo and repository link beside the documentation's page-source icons.

[0.1.2]: https://github.com/LorenzoTJ/doetools/releases/tag/v0.1.2

## [0.1.1] - 2026-09-05

### Added

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
