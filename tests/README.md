# doetools testing guide

The suite contains unit and integration tests for the public API, design generators,
analysis utilities, plotting, file import/export, and end-to-end design-advisor
workflows.

## Set up a development environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you use `uv`, the equivalent setup is:

```bash
uv sync --all-extras
```

## Run the suite

```bash
python -m pytest
```

Useful variations:

```bash
# Coverage report
python -m pytest --cov=doetools --cov-report=term-missing --cov-report=html

# Only unit or integration tests
python -m pytest tests/unit
python -m pytest tests/integration

# One module or one test
python -m pytest tests/unit/utils/test_design_advisor.py
python -m pytest tests/unit/utils/test_design_advisor.py::TestFactorAnalyzer
```

Pytest is configured in `pyproject.toml`. The repository includes a few historical
test modules without a `test_` prefix, so collection intentionally accepts every
Python file under `tests/`; test functions and classes must still use the standard
`test_*` and `Test*` names.

Markers available for new tests are `unit`, `integration`, and `slow`. Run, for
example, `python -m pytest -m "not slow"` to omit tests explicitly marked as slow.

The HTML coverage report, when requested, is written to `htmlcov/index.html`.
