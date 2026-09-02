"""
Factor classes for Design of Experiments (DoE).

This module provides three types of factors commonly used in experimental design:
- CategoricalFactor: For discrete, non-numeric levels (e.g., material types)
- ContinuousFactor: For numeric factors with a defined range and number of levels
- MixtureFactor: For mixture components with proportions summing to 1

Each factor type handles level generation and coding for use in DoE matrices.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CategoricalFactor:
    """
    Represents a categorical factor with discrete levels.
    
    Categorical factors are used for qualitative variables such as material types,
    suppliers, or process conditions. Levels are automatically coded to a numeric
    scale from -1 to 1 for use in design matrices and accessed through the
    ``coded_levels`` attribute.
    
    Parameters:
        levels (list[str]): List of category names for the factor.
        reference_level (str, optional): Level used as the operational reference
            for center points, plot defaults, and prediction profiles. If omitted,
            the first declared level is used.
            
    Raises:
        ValueError: If levels is empty or None.
        ValueError: If reference_level is not in levels.
    
    Example:
        >>> factor = CategoricalFactor(levels=["Low", "Medium", "High"])
    """
    levels: list[str]
    reference_level: str = None
    
    type: str = field(init=False, default="cat")
    coded_levels: np.ndarray = field(init=False)

    
    def __post_init__(self):
        if self.levels is None or len(self.levels) == 0:
            raise ValueError("levels must contain at least one categorical level")

        if self.reference_level is None:
            self.reference_level = self.levels[0]
        elif self.reference_level not in self.levels:
            raise ValueError(
                f"reference_level {self.reference_level!r} must be one of "
                f"the declared levels: {self.levels}"
            )

        k = len(self.levels)
        self.coded_levels = np.linspace(-1, 1, k)

    @property
    def reference_code(self) -> float:
        """Return the coded value associated with ``reference_level``."""
        try:
            index = self.levels.index(self.reference_level)
        except ValueError as exc:
            raise ValueError(
                f"reference_level {self.reference_level!r} must be one of "
                f"the declared levels: {self.levels}"
            ) from exc
        return float(self.coded_levels[index])
    
@dataclass
class ContinuousFactor:
    """
    Represents a continuous numeric factor with defined bounds and precision.
    
    Continuous factors are quantitative variables that can take on any value within
    a specified range (e.g., temperature, pressure, concentration). Levels are
    evenly spaced between the bounds and rounded to the specified decimal precision. 
    Levels can be accessed through the ``levels`` attribute, and standardized coded 
    levels from -1 to 1 can be accessed through the ``coded_levels`` attribute.
    
    Parameters:
        n_levels (int): Number of distinct levels to generate for the factor.
        lower_bound (float): Minimum value for the factor.
        upper_bound (float): Maximum value for the factor.
        decimals (int): Number of decimal places for rounding (default: 2).
            Set to 0 for integer levels.
    
    Raises:
        ValueError: If n_levels is less than one.
        ValueError: If lower_bound > upper_bound.
        ValueError: If decimals < 0.
        ValueError: If rounding causes bounds to collapse.
        ValueError: If integer levels cannot be evenly distributed.
        ValueError: If rounding creates duplicate levels.
    
    Example:
        >>> factor = ContinuousFactor(n_levels=5, lower_bound=0.0, 
        ...                           upper_bound=10.0, decimals=1)
    """
    n_levels: int
    lower_bound: float
    upper_bound: float
    decimals: int = 2
    
    type: str = field(init=False, default="cont")
    levels: np.ndarray = field(init=False)
    coded_levels: np.ndarray = field(init=False)
    
    def __post_init__(self):
        
        # Sanity Check
        if self.n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        if self.lower_bound > self.upper_bound:
             raise ValueError("lower_bound needs to be smaller than the upper bound")
        if self.decimals < 0:
            raise ValueError("decimals must be >= 0")
        
        self.upper_bound = np.round(self.upper_bound, decimals=self.decimals)
        self.lower_bound = np.round(self.lower_bound, decimals=self.decimals)
        
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                "Bounds collapse after rounding; "
               "adjust decimals, bounds or n_levels.")

        if self.lower_bound == self.upper_bound:
            if self.n_levels != 1:
                raise ValueError(
                    "Equal bounds describe one fixed level; set n_levels to 1."
                )
            self.levels = np.array([self.lower_bound])
            self.coded_levels = np.array([0.0])
            return
               
        # Decimal number 
        if self.decimals == 0:
            lb_i  = int(self.lower_bound)
            ub_i  = int(self.upper_bound)
            span = ub_i - lb_i
            if span % (self.n_levels - 1) != 0: 
                raise ValueError(f"It is not possible to find {self.n_levels} integer levels between {self.lower_bound} and {self.upper_bound}")
            else:
                self.levels = np.round(np.linspace(self.lower_bound, self.upper_bound, self.n_levels), decimals = self.decimals)

        else:
            self.levels = np.round(np.linspace(self.lower_bound, self.upper_bound, self.n_levels), decimals=self.decimals)
            
            if np.unique(self.levels).size != self.n_levels:
                raise ValueError(
                    "Rounding collapsed some levels into duplicates. "
                    "Adjust bounds, n_levels or decimals.")
        
        cp = (self.lower_bound + self.upper_bound) / 2
        self.coded_levels = (self.levels - cp) * (2 / (self.upper_bound - self.lower_bound))
        
@dataclass
class MixtureFactor:
    """
    Represents a mixture component factor with upper/ lower constraints.
    
    Mixture factors are used in mixture experiments where factors represent
    proportions of components that must sum to 1 (e.g., chemical compositions,
    blends). Each factor is constrained to a valid range within [0, 1].
    
    Parameters:
        lower_bound (float): Minimum proportion for the component (must be >= 0).
        upper_bound (float): Maximum proportion for the component (must be <= 1).
        decimals (int): Number of decimal places for rounding (default: 2).
            Must be > 0 for mixture factors.
    
    Raises:
        ValueError: If bounds are not in [0, 1].
        ValueError: If lower_bound > upper_bound.
        ValueError: If decimals < 0.
        ValueError: If decimals == 0 (proportions require decimal precision).
    
    Example:
        >>> factor = MixtureFactor(lower_bound=0.1, upper_bound=0.8, decimals=2)
    """
    lower_bound: float
    upper_bound: float
    decimals: int = 2
    
    type: str = field(init=False, default="mix")
    levels: np.ndarray = field(init=False)

    def __post_init__(self):
        
        # Sanity Check
        if self.lower_bound < 0 or self.upper_bound > 1:
            raise ValueError("Bounds for mixture factors must be in [0, 1]")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound needs to be smaller than the upper bound")
        if self.decimals < 0:
            raise ValueError("decimals must be >= 0")
        if self.decimals == 0:
            raise ValueError("Mixture factors cannot have decimals = 0")
