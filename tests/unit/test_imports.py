"""Unit tests for __init__ files to ensure proper imports."""

import pytest


def test_doetools_package_imports():
    """Test that main package imports work correctly."""
    try:
        import doetools
        assert doetools is not None
    except ImportError as e:
        pytest.fail(f"Failed to import doetools package: {e}")


def test_design_submodules():
    """Test design submodule imports."""
    try:
        from doetools.design.process import full_factorial
        from doetools.design.process import plackett_burman
        assert full_factorial is not None
        assert plackett_burman is not None
    except ImportError as e:
        pytest.fail(f"Failed to import design modules: {e}")


def test_utils_submodules():
    """Test utils submodule imports."""
    try:
        from doetools.utils import factors
        from doetools.utils import design_wizard
        assert factors is not None
        assert design_wizard is not None
    except ImportError as e:
        pytest.fail(f"Failed to import utils modules: {e}")
