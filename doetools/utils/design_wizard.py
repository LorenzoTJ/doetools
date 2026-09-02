"""
Design Advisor - DoE Recommendation System.

This module provides an expert system for recommending experimental designs
based on factor types, experimental phase, budget constraints, and modeling goals.

The Advisor analyzes the problem structure (mixture, process, or mixed factors)
and suggests appropriate classical designs (Full Factorial, Fractional Factorial,
Plackett-Burman, RSM designs) or D-Optimal designs for constrained spaces.

Key Functions
-------------
suggest_design : Main entry point for design recommendations

Key Classes
-----------
DesignRecommendation : Data structure for design suggestions
FactorAnalyzer : Analyzes and categorizes experimental factors

Examples
--------
    from doetools import ContinuousFactor as ContF
    
    factors = {
        'Temperature': ContF(n_levels = 3, lower_bound=20, upper_bound=80),
        'Pressure': ContF(n_levels=3, lower_bound=1, upper_bound=5),
        "pH" : ContF(n_levels=3, lower_bound=3, upper_bound=9)
     }
     
    recommendations = suggest_design(factors,
                                     phase='optimization',
                                     model_order='quadratic',
                                     max_experiments=20,
                                     non_rectangular_constraints=False,
                                     performed_exp=False)

"""

# Imports
from typing import Literal, List, Optional, Union
from dataclasses import dataclass
from ..design.process.plackett_burman import PlackettBurmanDesign   
from ..design.process.fractional_factorial import FractionalFactorialDesign
import math


# =========================
# Constants
# =========================
DEFAULT_CENTER_POINTS = 3  # Default number of center points for RSM designs


# =========================
# Data structures
# =========================
@dataclass
class DesignRecommendation:
    
    """Structure to hold experimental design recommendation details.
    
    Attributes
    ----------
    design_name : str
        Name of the recommended design (e.g., "Plackett-Burman Design")
    n_runs : Optional[Union[int, str]]
        Number of experimental runs required, or descriptive text for flexible designs
    pros : Optional[List[str]]
        List of advantages of this design approach
    cons : Optional[List[str]]
        List of limitations or drawbacks
    use_case : Optional[List[str]]
        Specific scenarios where this design is most appropriate
    additional_info : Optional[List[str]], default=None
        Additional notes, implementation tips, or follow-up recommendations
    """
    design_name: str
    n_runs: Optional[Union[int, str]]
    pros: Optional[List[str]]
    cons: Optional[List[str]]
    use_case: Optional[List[str]]
    additional_info: Optional[List[str]] = None

    def __str__(self):
        """Format recommendation for display.
        
        Returns
        -------
        str
            Formatted, human-readable string with design details
        """
        result = [f"{'=' * 100}"]
        result.append(f"Design: {self.design_name}")
        if self.n_runs is not None:
            result.append(f"Number of runs: {self.n_runs}")
        if self.pros is not None:
            result.append("")
            result.append("Pros:")
            for pro in self.pros:
                result.append(f"  • {pro}")
        if self.cons is not None:
            result.append("")
            result.append("Cons:")
            for con in self.cons:
                result.append(f"  • {con}")
        if self.use_case is not None:
            result.append("")
            result.append("Use case:")
            for case in self.use_case:
                result.append(f"  • {case}")
        if self.additional_info is not None:
            result.append("")
            result.append("Note:")
            for info in self.additional_info:
                result.append(f"  • {info}")
        result.append(f"{'=' * 100}")
        return "\n".join(result)


# =========================
# Factor analyzer
# =========================
class FactorAnalyzer:
    """Analyze and categorize experimental factors.
    
    Separates factors into process (continuous/categorical) and mixture types,
    providing convenient properties for querying factor structure.
    
    Parameters
    ----------
    factors : dict
        Dictionary mapping factor names to Factor objects with type attributes
    
    Attributes
    ----------
    factors : dict
        Original factor dictionary
    cont_factors : List[str]
        Names of continuous process factors
    cat_factors : List[str]
        Names of categorical factors
    pro_factors : List[str]
        Names of all process factors (continuous + categorical)
    mix_factors : List[str]
        Names of mixture component factors
    """

    def __init__(self, factors: dict):
        
        self.factors = factors
        self.cont_factors = [f for f in factors.keys() if factors[f].type == "cont"]
        self.cat_factors = [f for f in factors.keys() if factors[f].type == "cat"]
        self.pro_factors = self.cont_factors + self.cat_factors
        self.mix_factors = [f for f in factors.keys() if factors[f].type == "mix"]

    @property
    def n_continuous(self) -> int:
        return len(self.cont_factors)

    @property
    def n_categorical(self) -> int:
        return len(self.cat_factors)

    @property
    def n_process(self) -> int:
        return len(self.pro_factors)

    @property
    def n_mixture(self) -> int:
        return len(self.mix_factors)

    @property
    def has_mixture(self) -> bool:
        return self.n_mixture > 0

    @property
    def has_process(self) -> bool:
        return self.n_process > 0

    @property
    def has_categorical(self) -> bool:
        return self.n_categorical > 0

    def is_mixed_problem(self) -> bool:
        """Check if problem has both mixture and process factors."""
        return self.has_mixture and self.has_process

    def validate(self) -> None:
        """Validate factor configuration."""
        if not self.factors:
            raise ValueError("No factors provided.")
        if self.n_mixture == 0 and self.n_process == 0:
            raise ValueError("No valid factors found. Factors must be 'cont', 'cat', or 'mix' type.")
    
    def has_upper_bounds(self) -> bool:
        """Check if any mixture factor has an upper bound < 1.0 (constrained simplex)."""
        if not self.mix_factors:
            return False
        return any(
            hasattr(self.factors[f], 'upper_bound') and self.factors[f].upper_bound < 1.0
            for f in self.mix_factors
        )


# =========================
# Small helperS
# =========================
def _process_levels(fa: FactorAnalyzer) -> List[int]:
    """Return level count for each process factor.
    
    Parameters
    ----------
    fa : FactorAnalyzer
         Factor analyzer with categorized factors
    
    Returns
    -------
    List[int]
        Number of levels for each process factor (continuous and categorical)
    """
    f = fa.factors
    lv = []
    for name in fa.pro_factors:
        if f[name].type == "cont":
            lv.append(f[name].n_levels)
        elif f[name].type == "cat":
            lv.append(len(f[name].levels))
        else:
            raise ValueError("Unexpected factor type in process factors.")
    return lv

def _cat_levels(fa: FactorAnalyzer) -> List[int]:
    """Return level count for each categorical factor.
    
    Parameters
    ----------
    fa : FactorAnalyzer
        Factor analyzer with categorized factors
    
    Returns
    -------
    List[int]
        Number of levels for each categorical factor
    """
    f = fa.factors
    lv = []
    for name in fa.cat_factors:
            lv.append(len(f[name].levels))
    return lv

def _cont_levels(fa: FactorAnalyzer) -> List[int]:
    """Return level count for each continuous factor.
    
    Parameters
    ----------
    fa : FactorAnalyzer
        Factor analyzer with categorized factors
    
    Returns
    -------
    List[int]
        Number of levels for each continuous factor
    """
    f = fa.factors
    lv = []
    for name in fa.cont_factors:
            lv.append(f[name].n_levels)
    return lv

def _next_pow2(n: int) -> int:
    """Calculate the smallest power of two greater than or equal to n.
    
    Parameters
    ----------
    n : int
        Input value
    
    Returns
    -------
    int
        Smallest power of 2 that is >= n
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _min_runs_fractional(k: int, resolution: int) -> int:
    """
    Calculate minimum runs for 2-level fractional factorial designs.
    
    Uses wizard-grade sufficient bounds for regular 2-level fractional factorials:
    - Resolution III: N >= k + 1 (rounded to power of 2)
    - Resolution IV:  N >= 2k (rounded to power of 2)
    
    Parameters
    ----------
    k : int
        Number of factors (must be >= 1)
    resolution : int
        Design resolution (3 or 4)
    
    Returns
    -------
    int
        Minimum number of runs (power of 2)
    
    Raises
    ------
    ValueError
        If k <= 0 or resolution not in {3, 4}
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if resolution == 3:
        return _next_pow2(k + 1)
    if resolution == 4:
        return _next_pow2(2 * k)
    raise ValueError("resolution must be 3 or 4")


# =========================
# Public Advisor entrypoint (Public API)
# =========================
def suggest_design(
    factors: dict,
    phase: Literal["screening", "optimization"] = "optimization",
    model_order: Optional[Literal["linear", "2FI", "quadratic"]] = None,
    max_experiments: int = None,
    non_rectangular_constraints: bool = False,
    performed_exp: bool = False,
    print_output: bool = True,
) -> List[DesignRecommendation]:
    
    """
    Suggests experimental design(s) based on the provided factors and constraints.
    
    Parameters
    ----------
    factors : dict
        Dictionary of factor objects with their constraints
    phase : Literal["screening", "optimization"], default "optimization"
        Experimental phase
    model_order : Optional[Literal["linear", "2FI", "quadratic"]], optional
        Expected model complexity. Automatically determines interaction needs.
        - "linear": Main effects only
        - "2FI": Two-factor interactions
        - "quadratic": RSM quadratic model
    max_experiments : int, optional
        Maximum number of experimental runs
    performed_exp : bool, default False
        Whether prior experiments have been performed
    print_output : bool, default True
        Whether to print formatted recommendations to stdout
    
    Returns
    -------
    List[DesignRecommendation]
        List of recommended designs
    """

    factor_analyzer = FactorAnalyzer(factors)
    factor_analyzer.validate()

    if max_experiments is not None and max_experiments < 1:
        raise ValueError(f"max_experiments must be positive, got {max_experiments}")
    
    # Derive interactions from model_order
    interactions = model_order in ["2FI", "quadratic"] if model_order else False

    if phase == "screening":
        recommendations = _suggest_screening_design(factor_analyzer = factor_analyzer,
                                                    interactions = interactions,
                                                    max_experiments = max_experiments,
                                                    model_order = model_order,
                                                    non_rectangular_constraints = non_rectangular_constraints)
    elif phase == "optimization":
        recommendations = _suggest_optimization_design(factor_analyzer = factor_analyzer,
                                                       max_experiments = max_experiments,
                                                       performed_exp = performed_exp,
                                                       model_order = model_order,
                                                       non_rectangular_constraints = non_rectangular_constraints)
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'screening' or 'optimization'.")

    # Display recommendations
    if recommendations and print_output:
        # Format model order for display
        if model_order == "linear":
            model_order_text = "Linear"
        elif model_order == "2FI":
            model_order_text = "2FI"
        elif model_order == "quadratic":
            model_order_text = "Quadratic"
        else:
            model_order_text = "Not specified"
        
        print(f"\n{'#' * 100}")
        print("# DoE Advisor Recommendations")
        print(f"# Phase: {phase.upper()}")
        print(f"# Process Factors: {factor_analyzer.n_process} (Continuous: {factor_analyzer.n_continuous}, Categorical: {factor_analyzer.n_categorical})")
        print(f"# Mixture Factors: {factor_analyzer.n_mixture}")
        print(f"# Model Order: {model_order_text}")
        print(f"# Performed Experiments: {'Yes' if performed_exp else 'No'}")
        print(f"# Non-rectangular Constraints: {'Yes' if non_rectangular_constraints else 'No'}")
        if max_experiments:
            print(f"# Max Experiments: {max_experiments}")
        print(f"{'#' * 100}\n")

        for rec in recommendations:
            print(rec)
            print()

    return recommendations


# =========================
# Screening routing
# =========================

def _suggest_screening_design(
    factor_analyzer: FactorAnalyzer,
    interactions: bool,
    max_experiments: Optional[int],
    model_order: Optional[str],
    non_rectangular_constraints: bool
    ) -> List[DesignRecommendation]:
    
    """Route to appropriate screening design recommendations.
    
    Delegates to specialized functions based on factor types (mixture, process, or mixed).
    
    Parameters
    ----------
    factor_analyzer : FactorAnalyzer
        Analyzed factor structure
    interactions : bool
        Whether two-factor interactions should be considered
    max_experiments : Optional[int]
        Budget constraint on number of runs
    model_order : Optional[str]
        Expected model complexity
    non_rectangular_constraints : bool
        Whether the feasible region has non-rectangular constraints
    
    Returns
    -------
    List[DesignRecommendation]
        Screening design recommendations
    """
    
    # Process and Mixture Factors
    if factor_analyzer.has_process and factor_analyzer.has_mixture:
        return _screening_mixed_problem()

    # Process-only screening
    if factor_analyzer.has_process and not factor_analyzer.has_mixture:
        return _screening_process_only(factor_analyzer, interactions, max_experiments, non_rectangular_constraints)
    # Mixture-only screening
    elif factor_analyzer.has_mixture and not factor_analyzer.has_process:
        return _screening_mixture_only(factor_analyzer, max_experiments, non_rectangular_constraints)

    return []

def _screening_mixed_problem() -> List[DesignRecommendation]:
    
    """Suggest screening strategy for mixed process-mixture problems.
    
    Recommends separate screening of process and mixture factors followed by
    joint optimization using D-Optimal design.
    
    Returns
    -------
    List[DesignRecommendation]
        Single recommendation explaining the separate screening strategy
    """
    
    intro_recommendation = DesignRecommendation(
        design_name="Separate Screening Strategy: Process + Mixture",
        n_runs= None,
        
        pros=[
            "Screens process and mixture factors independently",
            "Clear interpretation of main effects within each category (process and mixture)",
            "Simplifies experimental planning and analysis",
        ],
        
        cons=[
            "Does not capture process-mixture interactions in screening phase",
            "Requires two separate experimental designs"
        ],
        
        use_case=[
            "Initial screening when process-mixture interactions are not expected to dominate",
            "When independent understanding of each factor type is important before joint optimization"
        ],
        
        additional_info=[
            "Recommended workflow:",
            "1) Screen separately (call suggest_design for process factors and mixture factors independently)",
            "2) Identify key factors",
            "3) Use D-Optimal design for joint optimization",
        ]
    )
    
    return [intro_recommendation]

def _screening_process_only(
    factor_analyzer: FactorAnalyzer,
    interactions: bool,
    max_experiments: Optional[int],
    non_rectangular_constraints: bool
    ) -> List[DesignRecommendation]:
    
    """
    Suggest screening designs for process factors only.
    
    For screening, continuous factors are expected to be configured with 2 levels (low/high);
    categorical factors keep their native number of levels.
    
    Parameters
    ----------
    factor_analyzer : FactorAnalyzer
        Analyzed factor structure
    interactions : bool
        Whether two-factor interactions should be considered
    max_experiments : Optional[int]
        Budget constraint on number of runs
    non_rectangular_constraints : bool
        Whether the feasible region has non-rectangular constraints
        
    Returns
    -------
    List[DesignRecommendation]
        Recommended screening designs for process factors
    """
    # Extract factor information
    factors_dict = factor_analyzer.factors
    cat_lv = _cat_levels(factor_analyzer)   
    cont_lv = _cont_levels(factor_analyzer)    

    n_cont = len(cont_lv)
    cat_prod = math.prod(cat_lv) if cat_lv else 1
    n_grid = (2 ** n_cont) * cat_prod

    # Check if categorical factors have more than 2-levels
    has_multilevel_cat = any(L > 2 for L in cat_lv)
    
    # Check if process factors have more than 2-levels
    has_multilevel_process = any(L > 2 for L in cont_lv)
    
    if has_multilevel_process:
        raise ValueError("Screening designs require all continuous factors to be 2-level (low/high).")
    
    if non_rectangular_constraints:
        return [DesignRecommendation(
            design_name="D-Optimal Screening Design (Constrained Process Domain)",
            n_runs= "Flexible, up to budget",
            pros=[
                "Respects feasibility constraints (avoids invalid factor combinations)",
                "Efficient main-effects screening under a fixed run budget",
                "Works in irregular or coupled constraint regions via a feasible candidate set"
            ],
            cons=[
                "Model-based: requires specifying screening terms upfront",
                "Limited ability to detect effects not included in the candidate model"
            ],
            use_case=[
                "Screening of process factors when constraints create a non-rectangular feasible region\n"
                "    and classical designs (full/fractional/PB) are infeasible or waste many runs"
            ],
            additional_info=[
                "Typical screening model: main effects; add only the most plausible two-factor interactions (2FI)."
            ]
        )]
        
    # Full Factorial Design - 2-levels or mixed-levels
    if max_experiments is None or n_grid <= max_experiments:

        # Mixed-levels FF
        # n_grid < max_experiment and has_multilevel_cat -> mixed-level ff
        if has_multilevel_cat:
            cat_levels_str = " × ".join(map(str, cat_lv)) if cat_lv else "1"

            return [DesignRecommendation(
                design_name="Full Factorial Design (Mixed Levels)",
                n_runs=n_grid,
                pros=[
                    "Exhaustive coverage of categorical levels (all combinations are tested)",
                    "Continuous factors are screened at low/high while preserving categorical structure",
                    "No confounding within the tested grid"
                ],
                cons=[
                    "Run count grows multiplicatively with the number of categorical levels",
                    "May be inefficient if you only need main-effect ranking (especially with many categorical levels)"
                ],
                use_case=[
                    "Screening with multi-level categorical factors when a complete grid fits the budget\n"
                    "    and you want clean, on-grid estimates of categorical effects (and their interactions)"
                ],
                additional_info=[
                    "Continuous factors are collapsed to 2 levels (low/high) for screening.",
                    f"Run count: 2^{n_cont} × ({cat_levels_str}) = 2^{n_cont} × {cat_prod} = {n_grid}"
                ]
            )]

        k2 = n_cont + len(cat_lv)

        # Two-level FF
        # n_grid < max_exp and not has_multilevel_cat -> two-level ff
        return [DesignRecommendation(
            design_name="Full Factorial Design (2-level)",
            n_runs=2**k2,
            pros=[
                "Estimates all main effects and interactions without aliasing",
                "Clean interpretation: no confounding within the design"
            ],
            cons=[
                f"Requires {2**k2} runs (2^{k2})",
                "Run count grows rapidly (2^k) as you add factors"
            ],
            use_case=[
                "Screening when the factor count is small and you want unambiguous visibility of\n"
                "    two-factor interactions (2FI) alongside main effects"
            ],
            additional_info=[
                "Continuous factors are collapsed to 2 levels (low/high) for screening.",
                "If you only need main-effect prioritization, consider Plackett–Burman.",
                "If interactions may matter but budget is limited, use a Fractional Factorial (Resolution IV/III)."
            ]
        )]

    # If the number of n_grid > max_exp and has_multilevel_cat -> No feasible design
    if has_multilevel_cat:
        return [DesignRecommendation(
            design_name="No Suitable Design Available - Budget Exceeded",
            n_runs=None,
            pros=None,
            cons=None,
            use_case=None,
            additional_info=[
                f"Full mixed-level grid would require {n_grid} runs, which exceeds the budget ({max_experiments}).",
                "Because at least one categorical factor has >2 levels, 2-level screening designs\n"
                "    (Plackett–Burman, Fractional Factorial Resolution III/IV) are not directly applicable.",
                "1) If feasible, increase the budget to run the full mixed-level grid.",
                "2) For screening, collapse categorical levels into 2 levels (then PB or fractional factorial becomes valid).",
                "3) Otherwise, use a custom main-effects screening design over a candidate set (e.g. D-optimal),\n"
                "       optionally adding a few targeted runs to probe suspected interactions."
            ]
        )]
        
    # Plackett-Burman Design
    # No interactions requested and 2-level factors only -> Plackett–Burman
    if not interactions:
        pb_design = PlackettBurmanDesign(factors=factors_dict)
        n_runs = pb_design.get_design_matrix().shape[0]
        max_factors = n_runs - 1

        return [DesignRecommendation(
            design_name="Plackett–Burman Design",
            n_runs=n_runs,
            pros=[
                "Efficient main-effects screening design (minimal runs).",
                f"Supports up to {max_factors} factors in {n_runs} runs (main effects only).",
                "Well-suited for rapid factor prioritization."
            ],
            cons=[
                "Does not estimate interaction effects.",
                "Main-effect estimates can be biased if interactions or curvature are non-negligible."
            ],
            use_case=[
                "Early-stage screening when the goal is to rank/prioritize factors and \n"
                "    interactions are expected to be small relative to main effects."
            ],
            additional_info=[
                "Recommended follow-up: confirm the top factors with a higher-resolution design."
            ])]
        
    # Fractional Factorial Design
    min_n4 = _min_runs_fractional(factor_analyzer.n_process, 4)
    min_n3 = _min_runs_fractional(factor_analyzer.n_process, 3)

    # Resolution IV
    # If min_n4 =< max_exp -> Res IV
    # main effects clear from 2FI and 2FI could be aliased with each other
    if max_experiments is not None and min_n4 <= max_experiments:
        frac_design_4 = FractionalFactorialDesign(factors=factors_dict, resolution=4)
        n_runs_4 = frac_design_4.get_design_matrix().shape[0]
        return [DesignRecommendation(
            design_name="Fractional Factorial Design (Resolution IV)",
            n_runs=n_runs_4,
            pros=[
                "Efficient screening with interaction awareness.",
                "Main effects are not aliased with two-factor interactions (2FI).",
                f"Fits within budget ({n_runs_4} ≤ {max_experiments} runs)."
            ],
            cons=[
                "Two-factor interactions (2FI) may be aliased with each other.",
                "Only a subset of interactions can be interpreted unambiguously."
            ],
            use_case= ["Screening when interactions are plausible but a full factorial is too costly."]
        )]
        
    # Resolution III
    # If min_n3 =< max_exp -> Res III
    # main effects could be aliased with 2FI
    if max_experiments is not None and min_n3 <= max_experiments:
        frac_design_3 = FractionalFactorialDesign(factors=factors_dict, resolution=3)
        n_runs_3 = frac_design_3.get_design_matrix().shape[0]
        return [DesignRecommendation(
            design_name="Fractional Factorial Design (Resolution III)",
            n_runs=n_runs_3,
            pros=[
                "Efficient screening with minimal runs.",
                f"Fits {factor_analyzer.n_process} factors in {n_runs_3} runs."
            ],
            cons=[
                "Main effects may be aliased with two-factor interactions (2FI).",
                "Effect estimates can be misleading if interactions are active."
            ],
            use_case=["Budget-limited screening when the primary goal is main-effect prioritization."],
            additional_info=[
                "Recommended follow-up: if any factors look important, move to an higher resolution Design."

            ]
        )]
        
    # Insufficient budget for fractional designs
    return [DesignRecommendation(
        design_name="Insufficient Budget",
        n_runs=None,
        pros=None,
        cons=None,
        use_case= None,
        additional_info=[
            f"Minimum runs needed: {min_n3} (Resolution III)",
            f"Current budget: {max_experiments} runs",
            "Consider (one or multiple):",
            f"1) Increasing budget to at least {min_n3} runs;",
            "2) Reducing the number of factors;",
            "3) Setting interactions=False to use Plackett–Burman."
        ]
    )]


def _screening_mixture_only(
    factor_analyzer: FactorAnalyzer,
    max_experiments: Optional[int],
    non_rectangular_constraints : bool,
) -> List[DesignRecommendation]:
    
    """Suggest screening designs for mixture factors only.
    
    Recommends Simplex-Centroid for unconstrained mixtures or D-Optimal
    for constrained mixture spaces.
    
    Parameters
    ----------
    factor_analyzer : FactorAnalyzer
        Analyzed factor structure
    max_experiments : Optional[int]
        Budget constraint on number of runs
    non_rectangular_constraints : bool
        Whether additional constraints exist beyond component bounds
    
    Returns
    -------
    List[DesignRecommendation]
        Screening design recommendations for mixture factors
    """

    q = factor_analyzer.n_mixture
    n_runs = (2 ** q) - 1

    if factor_analyzer.has_upper_bounds() or non_rectangular_constraints:
        return [DesignRecommendation(
            design_name="D-Optimal Design for Constrained Mixtures",
            n_runs= "Flexible, up to budget",
            pros=[
                "Handles constrained mixture spaces efficiently",
                "Accommodates component upper bounds and other constraints",
                "Flexible number of runs",
                "Tailored to the specified screening model (typically linear or 2FI)"
            ],
            cons=[
                "Requires explicit model specification upfront",
                "Less structured than classical designs (not vertex/centroid-based)",
                "May be less intuitive to interpret than Simplex-Centroid"
            ],
            use_case=[
                "Mixture screening when component bounds restrict the simplex, when standard\n"
                "    mixture designs are not feasible due to constraints"
            ],
            additional_info=[
                "Specify a linear or 2FI mixture model for screening purposes"
            ]
        )]
    
    # Simplex-Centroid for unconstrained mixtures
    if max_experiments is None or n_runs <= max_experiments:
        return [DesignRecommendation(
            design_name="Simplex-Centroid Mixture Design",
            n_runs=n_runs,
            pros=[
                "Well-suited for early mixture screening and component prioritization",
                "Includes vertices and balanced blends (centroid / equal-proportion points)",
                "Provides a clear first look at blending behavior within the simplex"
            ],
            cons=[
                "Run count grows quickly with the number of components",
                f"Requires {n_runs} runs for {q} components (may exceed budget for large q)"
            ],
            use_case=["Mixture screening to identify influential components and broad blending behavior"]
        )]
    
    # Insufficient budget
    return [DesignRecommendation(
        design_name="Insufficient Budget for Simplex-Centroid Design",
        n_runs=None,
        pros=None,
        cons=None,
        use_case=None,
        additional_info=[
            f"Simplex-Centroid design for {q} components requires {n_runs} runs, which exceeds the budget ({max_experiments}).",
            "Consider one of the following:",
            "1) Increase the budget to accommodate the Simplex-Centroid design",
            "2) Use a D-Optimal screening design tailored to your budget and model",
            "3) Reduce the number of mixture components to lower the run count"
        ]
    )]
    
# =========================
# Optimization routing
# =========================
def _suggest_optimization_design(
    factor_analyzer: FactorAnalyzer,
    max_experiments: Optional[int],
    performed_exp: bool,
    model_order: Optional[str],
    non_rectangular_constraints: bool
    ) -> List[DesignRecommendation]:
    
    """Route to appropriate optimization design recommendations.
    
    Delegates to specialized functions based on factor types (mixture, process, or mixed).
    
    Parameters
    ----------
    factor_analyzer : FactorAnalyzer
        Analyzed factor structure
    max_experiments : Optional[int]
        Budget constraint on number of runs
    performed_exp : bool
        Whether prior experiments exist (triggers augmentation)
    model_order : Optional[str]
        Expected model complexity
    non_rectangular_constraints : bool
        Whether the feasible region has non-rectangular constraints
    
    Returns
    -------
    List[DesignRecommendation]
        Optimization design recommendations
    """

    # Mixed problem (mixture + process)
    if factor_analyzer.is_mixed_problem():
        return _optimization_mixed_problem(performed_exp = performed_exp,
                                           max_experiments = max_experiments)

    # Pure mixture problem
    if factor_analyzer.has_mixture and not factor_analyzer.has_process:
        return _optimization_mixture_only(factor_analyzer=factor_analyzer,
                                          max_experiments=max_experiments,
                                          model_order=model_order,
                                          performed_exp=performed_exp,
                                          non_rectangular_constraints=non_rectangular_constraints)

    # Pure process problem
    if factor_analyzer.has_process and not factor_analyzer.has_mixture:
        return _optimization_process_only(factor_analyzer = factor_analyzer,
                                          max_experiments = max_experiments,
                                          performed_exp=performed_exp,
                                          model_order=model_order,
                                          non_rectangular_constraints=non_rectangular_constraints)

    return []


def _optimization_mixed_problem(performed_exp: bool, max_experiments: Optional[int]) -> List[DesignRecommendation]:
    
    """Suggest optimization designs for mixed mixture-process problems.
    
    Recommends D-Optimal design (or augmentation if prior data exists) to handle
    both mixture and process factors in a single design.
    
    Parameters
    ----------
    performed_exp : bool
        Whether prior experiments exist (triggers D-Optimal augmentation)
    max_experiments : Optional[int]
        Budget constraint on number of runs
    
    Returns
    -------
    List[DesignRecommendation]
        D-Optimal design or augmentation recommendation
    """
    if performed_exp:
        rec = DesignRecommendation(
            design_name="D-Optimal Augmentation (Mixture + Process)",
            n_runs= "Flexible, up to budget",
            pros=[
                "Adds new runs to maximize information gain given the existing data",
                "Handles mixture constraints and process factors in a single feasible design space",
                "Improves estimation precision for the specified model terms",
                "Well-suited for sequential experimentation (no need to restart the design)"
            ],
            cons=[
                "Model-based: added runs are optimal only for the assumed model structure",
                "May not reveal unmodeled effects unless those terms are included"
            ],
            use_case=[
                "Sequential optimization when prior runs exist and you want to refine the model \n"
                "    and improve precision without discarding data"
            ],
            additional_info=[
                "Typical workflow: fit/update the current model → build a feasible candidate set (mixture bounds\n"
                "    + process constraints) → select augmentation runs using D-optimality.",
                "Recommended practice: include any suspected missing terms before generating the augmentation."
            ]
        )
        return [rec]
    else:
        rec = DesignRecommendation(
            design_name="D-Optimal Design (Mixture + Process)",
            n_runs= "Flexible, up to budget",
            pros=[
                "Efficient initial run allocation targeted to the specified model under a fixed budget",
                "Supports mixture constraints together with process factors in a single feasible design space",
                "Works well in constrained or irregular domains where classical grids are infeasible"
            ],
            cons=[
                "Model-based: requires specifying the key model terms upfront",
                "Design quality depends on the candidate set and how constraints are represented"
            ],
            use_case=[
                "Optimization design when both mixture and process factors are present"
            ],
            additional_info=[
                "Important: include the terms you want to estimate before generating the design",            ]
        )
    return [rec]


def _optimization_mixture_only(
    factor_analyzer: FactorAnalyzer,
    max_experiments: Optional[int],
    model_order: Optional[str],
    performed_exp : bool,
    non_rectangular_constraints: bool
    ) -> List[DesignRecommendation]:
    """
    Suggest optimization designs for mixture-only problems.
    
    Recommends simplex-lattice for unconstrained mixtures or D-Optimal
    for constrained mixture spaces.
    
    Parameters
    ----------
    factor_analyzer : FactorAnalyzer
        Analyzed factor structure
    max_experiments : Optional[int]
        Budget constraint on number of runs
    model_order : Optional[str]
        Expected model complexity ("linear" or "quadratic")
    performed_exp : bool
        Whether prior experiments exist (triggers augmentation)
    non_rectangular_constraints : bool
        Whether additional constraints exist beyond component bounds
        
    Returns
    -------
    List[DesignRecommendation]
        Recommended optimization designs for mixture factors
    """

    mix_factors = factor_analyzer.mix_factors
    factors_dict = factor_analyzer.factors
    ub = [factors_dict[f].upper_bound for f in mix_factors]
    
    unrestricted = all(u == 1.0 for u in ub)
    
    # Add model order guidance
    model_note: Optional[List[str]] = None
    if model_order == "linear":
        model_note = ["Linear mixture model specified - simplex lattice with m=1 or m=2 is sufficient."]
    elif model_order == "quadratic":
        model_note = ["Quadratic mixture model specified - use simplex lattice with m=2 or m=3."]

    if unrestricted and not non_rectangular_constraints:
        
        if not performed_exp:
            return [DesignRecommendation(
                design_name="Simplex-Lattice Mixture Design",
                n_runs=None, 
                pros=[
                    "Systematic, grid-like coverage of the simplex (regular spacing in component proportions)",
                    "Well-suited for fitting standard Scheffé mixture models in an unconstrained simplex",
                    "Provides uniform exploration that supports model building and visualization"
                ],
                cons=[
                    "Run count grows quickly with the number of components and lattice degree",
                    "Less flexible than D-optimal designs when practical constraints or forbidden blends exist"
                ],
                use_case=[
                    "Mixture optimization when the feasible region is an unconstrained simplex and you \n"
                    "    want structured coverage to fit a chosen mixture model"
                ],
                additional_info=(
                    model_note if model_note else
                    ["Choose the lattice degree based on the intended model complexity and budget\n"
                    "    (e.g., degree 2 for quadratic Scheffé; higher degrees for more complex blending behavior)"]
                )
            )]
        
        else: 
            return [DesignRecommendation(
                design_name="D-Optimal Augmentation (Mixture)",
                n_runs= "Flexible, up to budget",
                pros=[
                    "Adds new runs to maximize information gain given the existing data",
                    "Improves estimation precision for the specified model terms",
                    "Well-suited for sequential experimentation (no need to restart the design)"
                ],
                cons=[
                    "Model-based: added runs are optimal only for the assumed model structure",
                    "May not reveal unmodeled effects unless those terms are included"
                ],
                use_case=[
                    "Sequential optimization when prior runs exist and you want to refine the model \n"
                    "    and improve precision without discarding data"
                ],
                additional_info=[
                    "Typical workflow: fit/update the current model → build a feasible candidate set (simplex) \n"
                    "    → select augmentation runs using D-optimality.",
                    "Recommended practice: include any suspected missing terms before generating the augmentation."
                ]
            )]
    
    else: 
    
        if not performed_exp:
            return [DesignRecommendation(
                design_name="D-Optimal Design for Constrained Mixture",
                n_runs= "Flexible, up to budget",
                pros=[
                    "Efficient initial run allocation targeted to the specified model under a fixed budget",
                    "Works well in constrained or irregular feasible regions where classical grids are infeasible"
                ],
                cons=[
                    "Model-based: requires specifying the key model terms upfront",
                    "Design quality depends on the candidate set and how constraints are represented"
                ],
                use_case=[
                    "Mixture optimization when component bounds or constraints restrict the simplex"
                ],
                additional_info=[
                    "Important: include the terms you want to estimate before generating the design"
                ]
            )]
            
        else: 
            rec = DesignRecommendation(
                design_name="D-Optimal Augmentation (Mixture)",
                n_runs= "Flexible, up to budget",
                pros=[
                    "Adds new runs to maximize information gain given the existing data",
                    "Improves estimation precision for the specified model terms",
                    "Well-suited for sequential experimentation (no need to restart the design)",
                    "Handles mixture constraints efficiently"
                ],
                cons=[
                    "Model-based: added runs are optimal only for the assumed model structure",
                    "May not reveal unmodeled effects unless those terms are included"
                ],
                use_case=[
                    "Sequential optimization when prior runs exist and you want to refine the model \n"
                    "    and improve precision without discarding data"
                ],
                additional_info=[
                    "Typical workflow: fit/update the current model → build a feasible candidate set (mixture bounds\n"
                    "    and constraints) → select augmentation runs using D-optimality.",
                    "Recommended practice: include any suspected missing terms before generating the augmentation."
                ]
            )
            return [rec]

def _optimization_process_only(
    factor_analyzer: FactorAnalyzer,
    max_experiments: Optional[int],
    performed_exp: bool,
    non_rectangular_constraints: bool,
    model_order: Optional[str]
) -> List[DesignRecommendation]:
    """
    Suggest optimization designs for process-only problems.
    
    Handles both pure continuous and mixed continuous/categorical scenarios.
    Recommends response surface methods (CCD, BBD) for continuous factors,
    or D-Optimal/Full Factorial for mixed scenarios.
    
    Parameters
    ----------
    factor_analyzer : FactorAnalyzer
        Analyzed factor structure
    max_experiments : Optional[int]
        Budget constraint on number of runs
    performed_exp : bool
        Whether prior experiments exist (triggers augmentation)
    non_rectangular_constraints : bool
        Whether the feasible region has non-rectangular constraints
    model_order : Optional[str]
        Expected model complexity ("linear", "2FI", or "quadratic")
        
    Returns
    -------
    List[DesignRecommendation]
        Recommended optimization designs for process factors
    """

    recommendations: List[DesignRecommendation] = []
    
    if non_rectangular_constraints:
        
        if not performed_exp:
        
            recommendations.append(DesignRecommendation(
                design_name="D-Optimal Design for Constrained Process Domain",
                n_runs= "Flexible, up to budget",
                pros=[
                    "Respects feasibility constraints (avoids invalid factor combinations) where classical designs fail",
                    "Efficient initial run allocation targeted to the specified model under a fixed budget"
                ],
                cons=[
                    "Model-based: requires specifying the key model terms upfront",
                    "Limited ability to reveal effects not represented in the model",
                    "Design quality depends on the candidate set and how constraints are represented"
                ],
                use_case=[
                    "Optimization of process factors when constraints create a non-rectangular feasible region\n"
                    "    and classical designs (full-factorial or RSM) are infeasible or waste many runs"
                ],
                additional_info=["Important: include the terms you want to estimate before generating the design."]
            ))
            return recommendations

        else:
            
            recommendations.append(DesignRecommendation(
                design_name="D-Optimal Augmentation for Constrained Process Domain",
                n_runs= "Flexible, up to budget",
                pros=[
                    "Adds new runs to maximize information gain given the existing data",
                    "Respects feasibility constraints (avoids invalid factor combinations)",
                    "Improves estimation precision for the specified model terms",
                    "Well-suited for sequential experimentation (no need to restart the design)"
                ],
                cons=[
                    "Model-based: added runs are optimal only for the assumed model structure",
                    "May not reveal unmodeled effects unless those terms are included"
                ],
                use_case=[
                    "Sequential optimization of process factors when prior runs exist and you want to refine the model \n"
                    "    and improve precision without discarding data"
                ],
                additional_info=[
                    "Typical workflow: fit/update the current model → build a feasible candidate set (process constraints)\n"
                    "    → select augmentation runs using D-optimality.",
                    "Recommended practice: include any suspected missing terms before generating the augmentation."
                ]
            ))
            return recommendations
    
    
    # Categorical Factors Present
    # Full Factorial or D-Opt Design -> Conditional Response Surface
    if factor_analyzer.has_categorical:
        lv = _process_levels(factor_analyzer)
        n_grid = math.prod(lv)
        
        if performed_exp:
            recommendations.append(DesignRecommendation(
                design_name="D-Optimal Augmentation (Continuous × Categorical)",
                n_runs= "Flexible, up to budget",
                pros=[
                    "Adds new runs to maximize information gain given the existing data",
                    "Improves estimation precision for the specified model terms",
                    "Well-suited for sequential experimentation (no need to restart the design)"
                ],
                cons=[
                    "Model-based: added runs are optimal only for the assumed model structure",
                    "May not reveal unmodeled effects unless those terms are included"
                ],
                use_case=[
                    "Sequential optimization with continuous and categorical factors when prior runs exist\n"
                    "    and you want to refine the model and improve precision without discarding data"
                ],
                additional_info=[
                    "Typical workflow: fit/update the current model → build a feasible candidate set \n"
                    "    (continuous settings × categorical levels) → select augmentation runs using D-optimality.",
                    "Recommended practice: include any suspected missing terms before generating the augmentation."
                ]
            ))
            
            return recommendations
        
        if max_experiments is None or n_grid <= max_experiments:
            recommendations.append(DesignRecommendation(
                design_name="Full Factorial Design (Continuous × Categorical)",
                n_runs=n_grid,
                pros=[
                    "Tests all combinations of the specified continuous settings and categorical levels",
                    "No confounding within the tested grid (effects are identifiable on-grid)",
                    f"Fits within budget ({n_grid} ≤ {max_experiments} runs)"
                ],
                cons=[
                    "Run count grows multiplicatively with the number of factors and levels",
                    "Can consume a large fraction of the budget, leaving limited room for replication or follow-up"
                ],
                use_case=[
                    "Optimization when the full grid is feasible and you want complete coverage of categorical levels\n"
                    "    together with the chosen continuous settings"
                ],
                additional_info=[
                    "Interpretation: fit a single model across all runs including categorical terms. Conditional \n"
                    "    response surfaces are then obtained by fixing the categorical levels and evaluating the model \n"
                    "    over the continuous factors.",
                    "Consider the D-Optimal design to save runs for replication or augmentation"
                ]
            ))
            
        recommendations.append(DesignRecommendation(
            design_name="D-Optimal Design (Continuous × Categorical)",
            n_runs= "Flexible, up to budget",
            pros=[
                "Efficient run allocation targeted to the specified model under a fixed budget",
                "Handles categorical factors without requiring a full factorial grid",
                "Works well when constraints restrict feasible combinations or continuous settings"
            ],
            cons=[
                "Model-based: requires specifying the key model terms upfront",
                "Effects not represented in the model cannot be reliably identified (e.g., missing interactions)",
                "Design quality depends on the candidate set and how categorical combinations are encoded"
            ],
            use_case=["Optimization with continuous and categorical factors when the full grid is infeasible or wasteful"],
            additional_info=[
                "Recommended practice: include categorical main effects and add continuous×categorical interactions \n"
                "    when you expect different continuous-factor trends across categorical levels. Conditional response \n"
                "    surfaces are obtained by fixing categorical levels in the fitted model and evaluating predictions \n"
                "    over the continuous factors."
            ]
            ))
        
        return recommendations

    # Pure continuous factors -> RSM suggestions
    else:
        k = factor_analyzer.n_continuous

        if k < 2:
            recommendations.append(DesignRecommendation(
                design_name="Insufficient Factors for RSM",
                n_runs=None,
                pros=[],
                cons=["Response surface designs require at least 2 continuous factors"],
                use_case=["Not applicable"],
                additional_info=["For a single factor, use a 1D design (e.g., multi-level sweep) or space-filling strategy."]
            ))
        else:
            n_ccd = (2 ** k) + (2 * k) + DEFAULT_CENTER_POINTS
            recommendations.append(DesignRecommendation(
                design_name="Central Composite Design (CCD)",
                n_runs=n_ccd,
                pros=[
                    "Efficient for fitting second-order (quadratic) response surface models",
                    "Good precision for estimating curvature and locating a stationary point (candidate optimum)",
                    "Standard, widely adopted design with a clear analysis workflow"
                ],
                cons=[
                    "May require axial (star) points outside the original factor ranges unless face-centered",
                    "Extremes can be impractical or infeasible under tight operating constraints"
                ],
                use_case=["Optimization when a quadratic response surface is expected and you want a well-established \n"
                "    design to locate and characterize an optimum"],
                additional_info=[
                    f"Run count assumes a factorial core plus {DEFAULT_CENTER_POINTS} center points.",
                    "If factor bounds are tight, consider a face-centered CCD or switch to a constrained\n"
                    "    (e.g., D-optimal) design or Box–Behnken design."
                ]
            ))

        if k >= 3:
            n_bb = (2 * k * (k - 1)) + DEFAULT_CENTER_POINTS
            recommendations.append(DesignRecommendation(
                design_name="Box–Behnken Design (BBD)",
                n_runs=n_bb,
                pros=[
                    "Efficient for fitting second-order (quadratic) response surface models",
                    "Avoids corner points: combinations stay within the factor bounds",
                    "No axial (star) points outside the specified factor ranges"
                ],
                cons=[
                    "Requires at least 3 factors",
                    "Provides limited information at the extremes of the design region (corners are not sampled)"
                ],
                use_case=["Optimization when a quadratic model is appropriate and running extreme combinations is undesirable or infeasible"],
                additional_info=[
                    f"Run count assumes {DEFAULT_CENTER_POINTS} center points.",
                    "Recommended when you want an interior quadratic design without star points",
                    "Choose CCD instead if corner/extreme behavior is critical"
                ]
            ))

        return recommendations
