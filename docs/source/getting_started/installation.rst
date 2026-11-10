.. _installation:

Installation
============

*doetools* requires Python 3.10 or newer. Install it from
`PyPI <https://pypi.org/project/doetools/>`_ with pip or
`uv <https://docs.astral.sh/uv/>`_.

Installing a release
--------------------

Install the latest stable release with pip in your Python environment:

.. code-block:: bash

    python -m pip install doetools

This installs *doetools* and its required dependencies.

In a project managed by uv, add *doetools* as a dependency:

.. code-block:: bash

    uv add doetools

This records the dependency in your project's ``pyproject.toml``, updates its
lockfile, and installs it in the project environment.

For a standalone virtual environment, create one with ``uv venv``, then activate
it using the command for your shell:

.. code-block:: powershell

    # Windows PowerShell
    .\.venv\Scripts\Activate.ps1

.. code-block:: bash

    # Linux/macOS (bash or zsh)
    source .venv/bin/activate

Install into that environment:

.. code-block:: bash

    uv pip install doetools

.. _installation-extras:

Optional dependencies
---------------------

The base installation includes analysis, interactive plotting, table import and
export, and PDF reports. Extras add the following tools; each also includes the
base dependencies:

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Extra
     - Purpose
     - Main tools
   * - ``export``
     - Static plot export with :func:`doetools.export_plot`
     - Kaleido
   * - ``dev``
     - Tests, linting, package builds, and distribution validation
     - pytest, pytest-cov, Ruff, build, Twine
   * - ``docs``
     - Building the documentation
     - Sphinx, Furo, nbsphinx, IPython

For example, add static export to a uv project with
``uv add "doetools[export]"``. In a standalone environment use
``uv pip install "doetools[export]"``, or
``python -m pip install "doetools[export]"`` with pip.
Combine extras with commas, for example ``doetools[dev,docs,export]``.
There are no extras named ``extra`` or ``all``.

Development installation from source
------------------------------------

Clone the repository and select the dependencies needed for your work:

.. code-block:: bash

    git clone https://github.com/LorenzoTJ/doetools.git
    cd doetools

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Command
   * - Base dependencies only
     - ``uv sync``
   * - Development and tests
     - ``uv sync --extra dev``
   * - Documentation
     - ``uv sync --extra docs``
   * - Static plot export
     - ``uv sync --extra export``
   * - All extras
     - ``uv sync --all-extras``

``uv sync`` creates ``.venv`` and installs the local package in editable mode,
so source changes are available without reinstalling. Here, ``dev`` and ``docs``
are extras in ``[project.optional-dependencies]``, not dependency groups:
``dev`` is not installed by default.

Repeat the desired extra flags when synchronizing; ``uv sync`` removes packages
that are not selected. Run tools through uv without activating the environment:

.. code-block:: bash

    uv run --extra dev pytest
    uv run --extra dev ruff check .
    uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/_build/html

Building the notebook documentation also requires
`Pandoc <https://pandoc.org/installing.html>`_, installed separately, and the
example files stored in Git LFS (``git lfs pull``).

Verifying the environment
-------------------------

In an activated environment, check the installed version:

.. code-block:: bash

    python -c "import doetools; print(doetools.__version__)"

For a uv-managed project or source checkout, prefix the command with ``uv run``.
In VS Code or Jupyter, select the Python kernel from the same environment
where you installed *doetools* and its extras. Restart the kernel after installing
new dependencies.
