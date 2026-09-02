.. _installation:

Installation
============

*doetools* can be installed from `PyPI <https://pypi.org/>`_ using pip, or from source for development.

*Requires Python 3.10 or higher.*

Installing with pip
-------------------

The simplest way to install *doetools* is using pip:

.. code-block:: bash

    python -m pip install doetools

This will install *doetools* and all required dependencies.

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

Installing from Source
----------------------

To install *doetools* from source (e.g., for development or to use the latest version):

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/LorenzoTJ/doetools.git
    cd doetools

2. Create a virtual environment:

.. code-block:: bash

    uv venv .venv

3. Activate the virtual environment:

.. code-block:: bash

    # On Windows:
    .venv\Scripts\activate
    
    # On macOS/Linux:
    source .venv/bin/activate

4. Install the dependencies:

.. code-block:: bash

    uv sync --all-extras
