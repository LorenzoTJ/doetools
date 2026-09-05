from .abstract_design import Design
from .factors import CategoricalFactor, ContinuousFactor, MixtureFactor
from .model_spec import ModelSpec, ModelTerms, compile_model_spec
from .pareto import ParetoMixin
from .upload import FileUploaderMixin


def __getattr__(name):
    if name == "suggest_design":
        from .design_advisor import suggest_design

        return suggest_design
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CategoricalFactor",
    "ContinuousFactor",
    "Design",
    "FileUploaderMixin",
    "MixtureFactor",
    "ModelSpec",
    "ModelTerms",
    "ParetoMixin",
    "compile_model_spec",
    "suggest_design"
]
