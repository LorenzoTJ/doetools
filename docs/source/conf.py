# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from doetools import __version__

# -- Project information -----------------------------------------------------
project = 'doetools'
copyright = '2026, Lorenzo Teja'
author = 'Lorenzo Teja'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate from docstrings
    'sphinx.ext.napoleon',          # Support NumPy/Google docstring styles
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx_autodoc_typehints',     # Better type hints
    'sphinxcontrib.mermaid',
    'nbsphinx',                     # Jupyter notebook support
]

templates_path = ['_templates']
root_doc = 'index'
language = 'en'
exclude_patterns = [
    # Auxiliary notebooks kept in the repository but not published in the
    # examples navigation.
    'examples/05_mix_4_comp.ipynb',
    'examples/example.ipynb',
    'examples/tutorial_dopt.ipynb',
]

# These DOI targets are valid but their publisher landing pages reject automated
# link-check clients with HTTP 403 responses.
linkcheck_ignore = [
    r'https://doi\.org/10\.1002/9780470027318\.a9646',
    r'https://doi\.org/10\.1080/00224065\.2016\.11918157',
]

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = 'never'  # Use pre-executed notebooks with saved outputs
nbsphinx_allow_errors = False  # Fail build if notebook has errors
nbsphinx_timeout = 60  # Timeout for notebook execution (if enabled)
nbsphinx_codecell_lexer = 'ipython3'  # Syntax highlighting for code cells

# Plotly graphs work with execute='never' as long as notebooks have saved outputs
# Just run notebooks in Jupyter/VS Code, save with outputs, then build docs

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_logo = "_static/doetools_logo.png"
html_theme_options = {
    "sidebar_hide_name": True,
}
pygments_style = "monokai"  # Use Visual Studio style for syntax highlighting

html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'none'  # Don't auto-generate parameter docs from type hints
autodoc_default_options = {
    'members': False,
    'member-order': 'bysource',
    'undoc-members': False,
    'private-members': False,
    'exclude-members': '__weakref__'
}

# Napoleon settings for better section rendering
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True  # Wrap examples in highlighted box
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True
