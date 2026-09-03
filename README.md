<p align="center">
  <img src="https://raw.githubusercontent.com/LorenzoTJ/doetools/main/docs/source/_static/doetools_logo.png" alt="doetools logo" width="320">
</p>

# doetools

[![CI](https://github.com/LorenzoTJ/doetools/actions/workflows/ci.yml/badge.svg)](https://github.com/LorenzoTJ/doetools/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`doetools` is a Python library for Design of Experiments (DoE). It supports the complete workflow
from choosing and generating a design to fitting response models, inspecting
diagnostics, exploring response surfaces, and validating predictions.

`doetools` is designed for scientists and engineers seeking a free, reproducible, and integrated alternative to manually assembling experimental design and analysis workflows.

To access the full documentation click [here](https://doetools.readthedocs.io/en/latest/).

## Features

* **Design recommendations** from high-level inputs;
* **Design Generation** for process and mixture factors;
* **Model Fitting** with ordinary least squares (OLS);
* **Regression analysis** and **interactive visualizations** in Plotly;
* **Multi-objective optimisation**;
* **Model validation** through confirmation runs;
* **PDF reports** for a comprehensive summary of the experimental workflow.

## Supported designs

| Category | Designs |
| --- | --- |
| Screening | Plackett-Burman, fractional factorial, two-level full factorial |
| Response surface | Three-level full factorial, Central composite (CCC, CCF, CCI), Box-Behnken |
| Mixture | Simplex lattice, simplex centroid, constrained mixture |
| Optimal | D-optimal, D-optimal augmentation |
| External generated designs | Import design |

## Requirements and platforms

`doetools` requires Python 3.10 or newer and is distributed as a pure-Python,
platform-independent package. Continuous integration is configured for CPython
3.10 through 3.14 on Linux, plus a current Python version on Windows and macOS.
Actual platform availability also depends on the scientific Python dependencies
listed in [`pyproject.toml`](pyproject.toml).

## Installation

Install the latest stable release from PyPI:

```bash
python -m pip install doetools
```

To install the current development version from GitHub:

```bash
python -m pip install "git+https://github.com/LorenzoTJ/doetools.git@dev"
```

For an editable development installation, see [Development](#development).

## Quick start

The following example creates a face-centered central composite design for two
continuous factors and exports a randomized experiment sheet for measuring yield:

```python
from doetools import CentralCompositeDesign, ContinuousFactor

factors = {
    "Temperature": ContinuousFactor(
        n_levels=3,
        lower_bound=60,
        upper_bound=80,
    ),
    "Pressure": ContinuousFactor(
        n_levels=3,
        lower_bound=1,
        upper_bound=3,
    ),
}

design = CentralCompositeDesign(
    factors=factors,
    design="ccf",
    center_points=3,
)

print(design.get_design_summary())
print(design.get_design_matrix())

design.export_experiments(
    responses=["Yield"],
    randomize=True,
    save_path="experiments.xlsx",
)
```

A typical analysis continues by completing the exported workbook, importing the
responses with `design.import_responses(...)`, defining `ModelTerms`, and calling
`design.compute_mlr_model()`.

## Documentation and examples

The documentation is built with Sphinx.

- [Full documentation source](https://doetools.readthedocs.io/en/latest/)
- [Complete example notebooks](https://doetools.readthedocs.io/en/latest/examples/index.html)

## Development

The development version of doetools is available on [GitHub](https://github.com/LorenzoTJ/doetools). Bug reports and feature suggestions are welcome through GitHub Issues. Detailed contributor and development guidelines will be provided in a future release.

## Project status and scope

- `doetools` is currently in its initial public-release stage (`0.1.x`). Backward compatibility is not guaranteed until version `1.0`; API changes and other notable updates are documented in the [CHANGELOG](CHANGELOG.md)
- Statistical results should be interpreted in the context of the experimental design, model assumptions, and relevant domain expertise. `doetools` does not replace experimental review or laboratory quality procedures.
- Models are currently fitted independently for each response using ordinary least squares (OLS). Correlated-response, time-series, Bayesian, mixed-effects, and other specialized modelling approaches are outside the current scope.

## Citation

If `doetools` contributes to published work, cite the software and the exact
version used. GitHub can generate citation metadata from [`CITATION.cff`](CITATION.cff).
No archival DOI has been assigned yet.

## License

`doetools` is released under the [MIT License](LICENSE). Third-party packages and
the publications cited by the documentation remain under their respective
licenses and terms.
