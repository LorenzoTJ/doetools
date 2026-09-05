# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where it is compatible with Python's versioning rules.

## [Unreleased]

### Added

- Initial public release preparation.
- Documentation and validation for importing external process and mixture designs.
- Design generation for process, mixture, constrained-mixture, and D-optimal
  experiments.
- Regression analysis, interactive visualization, prediction, validation,
  optimization, Excel workflows, and PDF reporting.
- Sphinx documentation and end-to-end example notebooks.
- Persistent CSV/XLSX prediction-point workflows with actual/coded getters and
  pointwise confidence intervals for the expected mean response.

### Changed

- `ImportDesign` now accepts a factor mapping instead of separate `vars` and
  `vars_type` lists. Imported continuous levels are taken from the file and
  may be non-equally spaced; coded imports are decoded with the supplied factor
  definitions.
- The plotting API now exposes ten standard `plot_*` methods. Model diagnostics
  are selected through `plot_diagnostics(view=...)`, focused main effects and
  interactions use optional selectors on their collection methods, and
  confirmation plots use `plot_confirmation(view=...)`.
- In-sample model predictions are now exposed as `get_fitted_values()`, while
  predictions at loaded external settings use `get_prediction_results(...)`.
- Tabular loaders use a common `source` argument and accept paths or pandas
  DataFrames; `export_experiments(...)` uses `destination` for its output path.
- `ImportDesign(factors, source, ...)` and `DOptAddDesign(factors, source, ...)`
  now follow the same convention and accept in-memory DataFrames.

### Removed

- Removed the redundant diagnostic, singular-effect, singular-interaction, and
  confirmation-specific plotting methods, together with accidentally public
  renderer and plotting-data helpers. This is an intentional pre-1.0 breaking
  change.
- Removed the public `predict(...)` method and `get_predicted_responses()` name as
  intentional pre-1.0 breaking changes; internal graph prediction remains private.
- Removed the standalone `simulate_responses(...)` development utility, which was
  not used by the library workflows or documentation.

[Unreleased]: https://github.com/LorenzoTJ/doetools/commits/dev
