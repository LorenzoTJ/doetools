"""Public package interface for :mod:`doetools`."""

__version__ = "0.1.0"

from .design import (
    BoxBehnkenDesign,
    CentralCompositeDesign,
    ConstrainedMixtureDesign,
    DOptAddDesign,
    DOptDesign,
    FractionalFactorialDesign,
    FullFactorialDesign,
    ImportDesign,
    PlackettBurmanDesign,
    SimplexCentroidDesign,
    SimplexLatticeDesign,
)
from .utils import (
    CategoricalFactor,
    ContinuousFactor,
    MixtureFactor,
    ModelTerms,
    simulate_responses,
    suggest_design,
)

__all__ = [
    "BoxBehnkenDesign",
    "CategoricalFactor",
    "CentralCompositeDesign",
    "ConstrainedMixtureDesign",
    "ContinuousFactor",
    "DOptAddDesign",
    "DOptDesign",
    "FractionalFactorialDesign",
    "FullFactorialDesign",
    "ImportDesign",
    "MixtureFactor",
    "ModelTerms",
    "simulate_responses",
    "PlackettBurmanDesign",
    "SimplexCentroidDesign",
    "SimplexLatticeDesign",
    "suggest_design",
    "__version__",
]
