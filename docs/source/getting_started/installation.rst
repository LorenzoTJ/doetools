.. _installation:

Installation
============

*doetools* can be installed from `PyPI <https://pypi.org/project/doetools/>`_
using pip or `uv <https://docs.astral.sh/uv/>`_. A source checkout is also
available for development.

*Requires Python 3.10 or higher.*

Installing with pip
-------------------

The simplest way to install *doetools* is using pip:

.. code-block:: bash

    python -m pip install doetools

This will install *doetools* and all required dependencies.

Installing with uv
------------------

If you use uv to manage your Python project, add *doetools* as a project
dependency:

.. code-block:: bash

    uv add doetools

This adds *doetools* to your project's ``pyproject.toml``, updates its lockfile,
and installs the package in the project environment.

If you only want to install *doetools* into an existing virtual environment
without adding it to a project, use:

.. code-block:: bash

    uv pip install doetools

Verifying the installation
--------------------------

Verify the installation:

.. code-block:: bash

    python -c "import doetools; print(doetools.__version__)"

Importing doetools
------------------

Once installed, you can import *doetools* in your Python code:

.. code-block:: python

    import doetools

Or import **specific modules**:

.. code-block:: python

    # Import Factors (Continuous, Categorical, Mixture)
    from doetools import ContinuousFactor, CategoricalFactor, MixtureFactor

    # Import specific designs 
    from doetools import FullFactorialDesign, CentralCompositeDesign, DOptDesign

    # Import model terms data structure
    from doetools import ModelTerms

Development installation from source
------------------------------------

To create a local development installation from the repository:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/LorenzoTJ/doetools.git
    cd doetools

2. Create and synchronize the project environment:

.. code-block:: bash

    uv sync --all-extras

The command creates the project's ``.venv`` environment and installs *doetools*
with all development and documentation dependencies. Commands can be run inside
that environment without activating it explicitly, for example:

.. code-block:: bash

    uv run python -c "import doetools; print(doetools.__version__)"
