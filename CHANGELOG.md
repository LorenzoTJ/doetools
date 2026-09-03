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

### Changed

- `ImportDesign` now accepts a factor mapping instead of separate `vars` and
  `vars_type` lists. Imported continuous levels are taken from the file and
  may be non-equally spaced; coded imports are decoded with the supplied factor
  definitions.

[Unreleased]: https://github.com/LorenzoTJ/doetools/commits/dev
