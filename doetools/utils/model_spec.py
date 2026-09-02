from dataclasses import dataclass, field
import itertools
import pandas as pd
from typing import Iterable, Literal

@dataclass
class ModelSpec:
    """
    Specification of a statistical model for design of experiments.
    
    This class defines the structure of a regression model by specifying which
    terms should be included: intercept, main effects, interaction terms, and
    quadratic terms. It automatically calculates the total number of model terms.
    
    Attributes
    ----------
    intercept : bool
        Whether to include an intercept term in the model.
    main : list[str] or None
        List of factor names for main effects (linear terms).
    interaction2 : list[tuple[str, str]] or None
        List of 2-way interaction terms as tuples of factor names.
    quadratic : list[str] or None
        List of factor names for quadratic terms.
    interaction3 : list[tuple[str, str, str]] or None
        List of 3-way interaction terms as tuples of factor names.
    model_terms : int
        Total number of terms in the model (calculated automatically).
    
    Examples
    --------
    >>> spec = ModelSpec(
    ...     intercept=True,
    ...     main=["A", "B", "C"],
    ...     interaction2=[("A", "B"), ("B", "C")],
    ...     quadratic=["A", "B"]
    ... )
    >>> spec.model_terms
    8
    """

    intercept : bool = True
    main : list[str] | None = None
    interaction2 : list[tuple[str, str]] | None = None
    quadratic : list[str] | None = None
    interaction3 : list[tuple[str, str, str]] | None = None
    
    model_terms : int = field(init=False)

    def __post_init__(self):
        
        self.model_terms = 0
        if self.intercept:
            self.model_terms += 1
        if self.main is not None:
            self.model_terms += len(self.main)
        if self.interaction2 is not None:
            self.model_terms += len(self.interaction2)
        if self.quadratic is not None:
            self.model_terms += len(self.quadratic)
        if self.interaction3 is not None:
            self.model_terms += len(self.interaction3)

    def to_vertical_df(self) -> pd.DataFrame:
        """
        Convert the model specification to a vertical DataFrame.
        
        Creates a DataFrame with model components as rows and their values
        in a single column, suitable for display and reporting.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with model components as index and values in
            a "Value" column.
        
        Examples
        --------
        >>> spec = ModelSpec(intercept=True, main=["A", "B"])
        >>> df = spec.to_vertical_df()
        >>> print(df.loc["Main Effects", "Value"])
        ['A', 'B']
        """
        data = {
            "Intercept": self.intercept,
            "Main Effects": self.main,
            "2-Term Interactions": self.interaction2,
            "Quadratic": self.quadratic,
            "3-Term Interactions": self.interaction3,
            "Total Model Terms": self.model_terms,
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=["Value"])

@dataclass
class ModelTerms:
    """
    Request specification for model terms in process and mixture designs.
    
    This class is used to specify which terms should be included in a model
    before it is compiled into a ModelSpec. It supports both process factors
    (with quadratic terms) and mixture factors (with Scheffe polynomials).
    
    Args:
        intercept (bool): Whether to include an intercept. It cannot be used with
            mixture factors. Defaults to ``True``.
        pro_main ("all" | list[str] | None): Process main effects. Defaults to
            ``"all"``.
        pro_int2 ("all" | list[tuple[str, str]] | None): Two-factor process
            interactions. Defaults to ``"all"``.
        pro_quadratic ("all" | list[str] | None): Process quadratic terms.
            Defaults to ``"all"``.
        pro_int3 ("all" | list[tuple[str, str, str]] | None): Three-factor
            process interactions. Defaults to ``None``.
        scheffe_pol_order (int | None): Scheffe polynomial order for mixture
            factors (1, 2, or 3). Defaults to ``None``.
        mix_main ("all" | list[str] | None): Mixture main effects. Defaults to
            ``None``.
        mix_int2 ("all" | list[tuple[str, str]] | None): Two-component mixture
            terms. Defaults to ``None``.
        mix_int3 ("all" | list[tuple[str, str, str]] | None): Three-component
            mixture terms. Defaults to ``None``.

    Notes:
        Setting ``scheffe_pol_order`` fills in unspecified mixture terms. Order 1
        includes main effects, order 2 also includes two-component terms, and order
        3 also includes three-component terms.
    
    """
    
    intercept: bool = True

    pro_main: Literal["all"] | list[str] | None = "all"
    pro_int2: Literal["all"] | list[tuple[str, str]] | None = "all"
    pro_quadratic: Literal["all"] | list[str] | None = "all"
    pro_int3: Literal["all"] | list[tuple[str, str, str]] | None = None

    scheffe_pol_order: int | None = None
    mix_main: Literal["all"] | list[str] | None = None
    mix_int2: Literal["all"] | list[tuple[str, str]] | None = None
    mix_int3: Literal["all"] | list[tuple[str, str, str]] | None = None

def _validate_all_or_list(x, name: str):
    """
    Validate that input is either 'all', a list, or None.
    
    Parameters
    ----------
    x : str, list, or None
        The value to validate.
    name : str
        Parameter name for error messages.
    
    Raises
    ------
    ValueError
        If x is a string other than 'all'.
    """
    if isinstance(x, str) and x != "all":
        raise ValueError(f"Invalid input for {name}, write 'all' or provide a list/None")

def _validate_vars_exist(items: Iterable[str], allowed: set[str], kind: str):
    """
    Validate that all items exist in the allowed set of variables.
    
    Parameters
    ----------
    items : Iterable[str]
        Variable names to validate.
    allowed : set[str]
        Set of allowed variable names.
    kind : str
        Type of factors ("process" or "mixture") for error messages.
    
    Raises
    ------
    ValueError
        If any item is not in the allowed set.
    """
    for v in items:
        if v not in allowed:
            raise ValueError(f"Variable {v!r} not defined in the {kind} factors")

def _flatten_tuples(tuples_list):
    """
    Flatten a list of tuples into a single list.
    
    Parameters
    ----------
    tuples_list : list[tuple]
        List of tuples to flatten.
    
    Returns
    -------
    list
        Flattened list containing all elements from all tuples.
    
    Examples
    --------
    >>> _flatten_tuples([("A", "B"), ("C", "D")])
    ['A', 'B', 'C', 'D']
    """
    return [e for tup in tuples_list for e in tup]

def _expand_main(factors: list[str], spec):
    """
    Expand main effect specification into a list of factor names.
    
    Parameters
    ----------
    factors : list[str]
        List of available factor names.
    spec : {"all", list[str], None}
        Specification - "all" for all factors, list for specific factors,
        or None for no factors.
    
    Returns
    -------
    list[str]
        List of factor names to include as main effects.
    
    Examples
    --------
    >>> _expand_main(["A", "B", "C"], "all")
    ['A', 'B', 'C']
    >>> _expand_main(["A", "B", "C"], ["A", "C"])
    ['A', 'C']
    """
    # spec is "all" or list[str] or None
    if spec == "all":
        return list(factors)
    return list(spec) if spec is not None else []

def _expand_int(factors: list[str], r: int, spec):
    """
    Expand interaction specification into a list of factor tuples.
    
    Parameters
    ----------
    factors : list[str]
        List of available factor names.
    r : int
        Order of interaction (2 for 2-way, 3 for 3-way).
    spec : {"all", list[tuple], None}
        Specification - "all" for all combinations, list of tuples for
        specific interactions, or None for no interactions.
    
    Returns
    -------
    list[tuple]
        List of tuples representing interaction terms.
    
    Examples
    --------
    >>> _expand_int(["A", "B", "C"], 2, "all")
    [('A', 'B'), ('A', 'C'), ('B', 'C')]
    >>> _expand_int(["A", "B", "C"], 2, [("A", "B")])
    [('A', 'B')]
    """
    # spec is "all" or list[tuple] or None
    if spec == "all":
        return list(itertools.combinations(factors, r))
    return list(spec) if spec is not None else []

def compile_model_spec(
    req : ModelTerms,
    pro_factors = list[str],
    mix_factors = list[str],
    ) -> ModelSpec:
    """
    Compile a ModelTerms request into a concrete ModelSpec.
    
    This function expands the high-level model term specifications in ModelTerms
    into explicit lists of terms, validates all specifications, and returns a
    complete ModelSpec ready for use in model fitting.
    
    Parameters
    ----------
    req : ModelTerms
        ModelTerms object specifying desired model structure.
    pro_factors : list[str]
        List of process factor names.
    mix_factors : list[str]
        List of mixture factor names.
    
    Returns
    -------
    ModelSpec
        Compiled model specification with explicit term lists.
    
    Raises
    ------
    ValueError
        If intercept is used with mixture models.
    ValueError
        If Scheffé order is specified without mixture factors.
    ValueError
        If mixture factors exist but no mixture terms are specified.
    ValueError
        If Scheffé polynomial order is not 1, 2, or 3.
    ValueError
        If any specified variable doesn't exist in the factor lists.
    ValueError
        If string input is not "all".
    
    Examples
    --------
    >>> req = ModelTerms(intercept=True, pro_main="all", pro_int2="all")
    >>> spec = compile_model_spec(req, pro_factors=["A", "B"], mix_factors=[])
    >>> spec.main
    ['A', 'B']
    >>> spec.interaction2
    [('A', 'B')]
    """
        
    # Basic Validations 
    if req.intercept and len(mix_factors) > 0:
        raise ValueError("Intercept cannot be included in a mixture model.")
    if req.scheffe_pol_order is not None and not mix_factors:
        raise ValueError("Scheffé polynomial order can be set only for mixture designs.")
    if req.scheffe_pol_order is None and mix_factors and all(v is None for v in (req.mix_main, req.mix_int2, req.mix_int3)):
        raise ValueError("No mixture model terms specified. Specify mix_* or scheffe_pol_order.")
    
    # Validate "all" or list inputs
    _validate_all_or_list(req.pro_main, "pro_main")
    _validate_all_or_list(req.pro_int2, "pro_int2")
    _validate_all_or_list(req.pro_int3, "pro_int3")
    _validate_all_or_list(req.pro_quadratic, "pro_quadratic")
    _validate_all_or_list(req.mix_main, "mix_main")
    _validate_all_or_list(req.mix_int2, "mix_int2")
    _validate_all_or_list(req.mix_int3, "mix_int3")
    
    # Scheffè polynomial order expansions
    if req.scheffe_pol_order is not None:
        if req.scheffe_pol_order < 1 or req.scheffe_pol_order > 3:
            raise ValueError("Scheffé polynomial order must be 1, 2, or 3.")
        if req.scheffe_pol_order >= 1 and req.mix_main is None:
            req.mix_main = "all"
        if req.scheffe_pol_order >= 2 and req.mix_int2 is None:
            req.mix_int2 = "all"
        if req.scheffe_pol_order == 3 and req.mix_int3 is None:
            req.mix_int3 = "all"
            
    # Expand Terms - Process 
    pro_main = _expand_main(pro_factors, req.pro_main)
    pro_int2 = _expand_int(pro_factors, 2, req.pro_int2)
    pro_int3 = _expand_int(pro_factors, 3, req.pro_int3)
    pro_quadratic = _expand_main(pro_factors, req.pro_quadratic)
    # Expand Terms - Mixture
    mix_main = _expand_main(mix_factors, req.mix_main)
    mix_int2 = _expand_int(mix_factors, 2, req.mix_int2)
    mix_int3 = _expand_int(mix_factors, 3, req.mix_int3)

    # Validate that specified variables exist
    _validate_vars_exist(pro_main, set(pro_factors), "process")
    _validate_vars_exist(_flatten_tuples(pro_int2), set(pro_factors), "process")
    _validate_vars_exist(_flatten_tuples(pro_int3), set(pro_factors), "process")
    _validate_vars_exist(pro_quadratic, set(pro_factors), "process")
    _validate_vars_exist(mix_main, set(mix_factors), "mixture")
    _validate_vars_exist(_flatten_tuples(mix_int2), set(mix_factors), "mixture")
    _validate_vars_exist(_flatten_tuples(mix_int3), set(mix_factors), "mixture")
    
    # Combine Process and Mixture terms 
    main = pro_main + mix_main
    int2 = pro_int2 + mix_int2
    int3 = pro_int3 + mix_int3 
    quadratic = pro_quadratic
    
    # Build ModelSpec
    return ModelSpec(
        intercept= req.intercept,
        main= main,
        interaction2= int2,
        interaction3= int3,
        quadratic= quadratic
    )
