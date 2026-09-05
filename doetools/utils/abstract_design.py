from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union  # noqa: UP035

import numpy as np
import pandas as pd

from .confirmation import ConfirmationRunsMixin
from .model_spec import ModelSpec, ModelTerms, compile_model_spec
from .prediction import PredictionPointsMixin
from .regression import RegressionAnalyzer, RegressionWrapper
from .summary import DesignSummaryMixin
from .upload import FileUploaderMixin


class Design(
    ABC,
    FileUploaderMixin,
    DesignSummaryMixin,
    ConfirmationRunsMixin,
    PredictionPointsMixin,
):
  
  """
  Abstract base class for experimental design of experiments (DoE).
  
  This class provides the core functionality for managing experimental designs,
  including design matrix generation, coding/decoding transformations, model
  specification, and multiple linear regression analysis. It supports three
  factor types: continuous (cont), categorical (cat), and mixture (mix).
  
  The class implements a comprehensive workflow for:
  - Design matrix construction and manipulation
  - Factor level coding and decoding
  - Model matrix generation with interactions and polynomial terms
  - Replicate handling and leverage computation
  - Response data management and MLR model fitting
  - Experimental data export and import
  
  Attributes
  ----------
  TOLERANCE : float
      Numerical tolerance for floating-point comparisons (1e-12).
  EXP_IDX_COL : str
      Column name for experiment index in exported files.
  EXP_ORDER_COL : str
      Column name for experiment execution order in exported files.
      
  Notes
  -----
  This is an abstract base class and should not be instantiated directly.
  Subclasses must initialize the design-specific attributes, particularly
  `_factors`, `_coded_design_matrix`, and `_design_matrix`.
  
  See Also
  --------
  FileUploaderMixin : Mixin for file upload functionality
  DesignSummaryMixin : Mixin for design summary and reporting
  ModelSpec : Model specification class for term management
  """

  # Class constants
  TOLERANCE: float = 1e-12
  EXP_IDX_COL: str = "Exp. Idx"
  EXP_ORDER_COL: str = "Exp. Order"

  def __init__(self):
    
    """
    Initialize the Design object with default attribute values.
    
    Sets up internal data structures for design matrices, factors, model
    specifications, responses, and regression analysis components.
    """

    # ==========================================================================
    #                              Initialization
    # ==========================================================================
    
    self._analyzer: RegressionAnalyzer = RegressionAnalyzer()

    # ---------------------------- Design Information ------------------------ #
    self._factors: Optional[Dict[str, Any]] = None
    self._design_type: Optional[str] = None
    self._replicates: int = 0
    self._center_points: int = 0
    self._coded_design_matrix: Optional[pd.DataFrame] = None
    self._design_matrix: Optional[pd.DataFrame] = None
    self._domain_filters: List[Callable[[pd.DataFrame], pd.Series]] = []
    
    # ---------------------------- Model Terms ------------------------------- #
    # Model specification (main, interaction, quadratic terms)
    self._model_spec: Optional[ModelSpec] = None
    # Model Matrix
    self._model_matrix: Optional[pd.DataFrame] = None
    
    # --------------------------- Leverages ---------------------------------- #
    self._leverages: Optional[pd.DataFrame] = None
    
    # --------------------------- Responses ---------------------------------- #
    self._response_list: Optional[List[str]] = None
    self._responses: Optional[pd.DataFrame] = None
    self._response_conditions: Dict[str, Dict[str, Any]] = {}
    self._experimental_metadata: Optional[pd.DataFrame] = None
    
    # --------------------------- MLR Model ---------------------------------- #
    self._mlr_wrapper: Optional[RegressionWrapper] = None

    # ------------------------ Confirmation Runs ---------------------------- #
    self._initialize_confirmation_runs()

    # ------------------------- Prediction Points --------------------------- #
    self._initialize_prediction_points()

  def set_domain_filters(
      self,
      filters: list[Callable[[pd.DataFrame], pd.Series]] | None,
  ) -> None:
    """Set constraints describing the feasible experimental domain.

    Each filter receives a design matrix in actual factor units and must
    return one boolean value per row. Passing ``None`` clears the filters.
    """
    if filters is None:
      self._domain_filters = []
      return
    if not isinstance(filters, (list, tuple)) or not all(
        callable(domain_filter) for domain_filter in filters
    ):
      raise TypeError("filters must be a list of callable functions or None.")
    self._domain_filters = list(filters)

  # ===========================================================================
  #                               Internal Methods
  # ===========================================================================
  
  def _add_center_points(self, 
                         design_matrix: pd.DataFrame,
                         center_points: int) -> pd.DataFrame:
    
    """
    Add center point runs to the design matrix.
    
    Center points are computed at the midpoint for continuous factors (coded value 0),
    at the configured reference level for categorical factors, and at the centroid
    of the simplex region for mixture factors.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        The coded design matrix to which center points will be added.
    center_points : int
        Number of center point replicates to add.
    
    Returns
    -------
    pd.DataFrame
        Design matrix with center points appended.
        
    Raises
    ------
    ValueError
        If the sum of mixture component lower bounds exceeds 1.
    """
    
    factors = design_matrix.columns
    pro_factors = [f for f in factors if self._factors[f].type in ["cont", "cat"]]
    mix_factors = [f for f in factors if self._factors[f].type == "mix"]
    
    center_row = {
        name: [0]
        if self._factors[name].type == "cont"
        else [self._factors[name].reference_code]
        for name in pro_factors
    }
        
    if len(mix_factors) > 0:
      
      n_vars = len(mix_factors)
      # dictionary of lower bounds
      L = {name: self._factors[name].lower_bound for name in mix_factors}

      # slack available to reach the simplex sum 1
      S = 1.0 - float(sum(L.values()))

      # if lower bounds already exceed 1 (beyond tolerance), fail
      if S < -self.TOLERANCE:
          raise ValueError("Sum of lower_bounds must be ≤ 1.")
      S = max(S, 0.0)

      # equal share of the remaining slack across variables
      z = 1.0 / n_vars

      # center row on the simplex slice: L + S*z
      for name in mix_factors:
        center_row[name] = [L[name] + S * z]
    
    df = pd.DataFrame(center_row, columns=factors)
    center_df = df.loc[df.index.repeat(center_points)].reset_index(drop=True)
    design_matrix = pd.concat([design_matrix.reset_index(drop=True), center_df], ignore_index=True)
    
    return design_matrix
  
  def _number_of_center_points(self, design_matrix: pd.DataFrame) -> int:
    
    """
    Count the number of center point runs in the design matrix.
    
    Identifies rows that match the center point conditions for all factor types:
    continuous factors at 0, categorical factors at their configured reference
    level, and mixture factors at their simplex centroid.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        The coded design matrix to analyze.
    
    Returns
    -------
    int
        Number of center point runs found in the design matrix.
        
    Raises
    ------
    ValueError
        If the sum of mixture component lower bounds exceeds 1.
    """
    mix_factors = [f for f in self._factors.keys() if self._factors[f].type == "mix"]
      
    if len(mix_factors) > 0:
      n_vars = len(mix_factors)
      lb = [self._factors[f].lower_bound for f in mix_factors]
      S = 1.0 - sum(lb)
      if S < -self.TOLERANCE:
          raise ValueError("Sum of lower_bounds must be ≤ 1.")
      S = max(S, 0.0)
      z = 1.0 / n_vars
        
    vt = []
    for f in self._factors.keys():
      if self._factors[f].type == "cont":
        vt.append(0)
      if self._factors[f].type == "cat":
        vt.append(self._factors[f].reference_code)
      if self._factors[f].type == "mix":
        cp = self._factors[f].lower_bound + S * z
        vt.append(cp)
          
    cp = 0
    for i in range(design_matrix.shape[0]):
        row = design_matrix.iloc[i, :].to_numpy()
        if np.array_equal(row, np.array(vt)):
            cp += 1
    return cp
  
  def _decode_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Transform coded factor levels to their original experimental values.
    
    Converts the coded design matrix (typically in [-1, 1] range for continuous
    factors) back to the actual experimental units. Categorical factors are
    mapped to their original level names, and mixture factors remain unchanged.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Coded design matrix to decode.
    
    Returns
    -------
    pd.DataFrame
        Design matrix with factors in original experimental units.
        
    Notes
    -----
    The transformation for continuous factors is:
        actual_value = center_point + (coded_value * span / 2)
    where span = upper_bound - lower_bound.
    """
    
    X = matrix.copy()
    # Create a list of the columns in the design matrix
    cols = [c for c in X.columns if c in self._factors.keys()]
    for col in cols:
      if self._factors[col].type == "cat":
        code = {}
        k = len(self._factors[col].levels)
        for i in range(k):
          code[self._factors[col].coded_levels[i]] = self._factors[col].levels[i]
        X[col] = X[col].map(code)
      elif self._factors[col].type == "cont":
        lb = self._factors[col].lower_bound
        ub = self._factors[col].upper_bound
        span = ub - lb
        cp = (lb + ub) / 2
        X[col] = X[col].map(lambda v: cp + (v * span / 2))
      elif self._factors[col].type == "mix":
        pass

    return X
  
  def _code_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Transform experimental factor levels to coded values.
    
    Converts the design matrix from original experimental units to coded values.
    Continuous factors are scaled to the [-1, 1] range, categorical factors are
    mapped to their coded levels, and mixture factors remain unchanged.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Design matrix with factors in original experimental units.
    
    Returns
    -------
    pd.DataFrame
        Coded design matrix.
        
    Notes
    -----
    The transformation for continuous factors is:
        coded_value = ((actual_value - center_point) * 2) / span
    where span = upper_bound - lower_bound.
    """
    
    X = matrix.copy()
    # Create a list of the columns in the design matrix
    cols = [c for c in X.columns if c in self._factors.keys()]
    for col in cols:
      if self._factors[col].type == "cat":
        code = {}
        k = len(self._factors[col].levels)
        for i in range(k):
          code[self._factors[col].levels[i]] = self._factors[col].coded_levels[i]
        X[col] = X[col].map(code)
      elif self._factors[col].type == "cont":
        lb = self._factors[col].lower_bound
        ub = self._factors[col].upper_bound
        span = ub - lb
        cp = (lb + ub) / 2
        X[col] = X[col].map(lambda v: ( (v - cp) * 2 ) / span)
      elif self._factors[col].type == "mix":
        pass

    return X
    
  def _code_dict(self, levels: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Transform a dictionary of factor levels to coded values.
    
    Converts individual factor level specifications from experimental units to
    coded values. Used primarily for setting constant levels in plotting and
    optimization routines.
    
    Parameters
    ----------
    levels : dict of {str: float} or None
        Dictionary mapping factor names to their experimental values.
        If None, returns None.
    
    Returns
    -------
    dict of {str: float} or None
        Dictionary with coded factor values, or None if input is None.
        
    Notes
    -----
    Categorical factors are mapped using their predefined coding scheme,
    continuous factors are scaled to [-1, 1], and mixture factors are
    returned unchanged.
    """
    
    if levels is None:
      return None
    coded_levels = {}
    for var, val in levels.items():
      if self._factors[var].type == "cat":
        code = {}
        k = len(self._factors[var].levels)
        for i in range(k):
          code[self._factors[var].levels[i]] = self._factors[var].coded_levels[i]
        coded_levels[var] = code[val]
      elif self._factors[var].type == "cont":
        lb = self._factors[var].lower_bound
        ub = self._factors[var].upper_bound
        span = ub - lb
        cp = (lb + ub) / 2
        coded_levels[var] = ( (val - cp) * 2 ) / span
      elif self._factors[var].type == "mix":
        coded_levels[var] = val

    return coded_levels
  
  def _build_model_matrix(self, matrix: pd.DataFrame, model_spec: ModelSpec) -> pd.DataFrame:
    """
    Construct the model matrix from the design matrix and model specification.
    
    Generates the full model matrix including intercept, main effects, two-way
    and three-way interactions, and quadratic terms as specified in the model
    specification.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Coded design matrix containing factor columns.
    model_spec : ModelSpec
        Model specification object defining which terms to include.
    
    Returns
    -------
    pd.DataFrame
        Model matrix with all specified terms as columns.
        
    Raises
    ------
    KeyError
        If a quadratic term is requested for a factor with only 2 levels.
        
    Notes
    -----
    The model matrix construction follows this order:
    1. Filter main effects
    2. Add intercept (if specified)
    3. Add two-way interactions (e.g., A:B)
    4. Add three-way interactions (e.g., A:B:C)
    5. Add quadratic terms (e.g., A^2)
    """

    model_matrix = matrix.copy()
    
    # 0) Check for main terms
    for term in model_matrix.columns:
      if term not in model_spec.main:
        model_matrix.drop(columns=term, inplace=True)
        
    # 1) Add Intercept
    if model_spec.intercept:
        model_matrix.insert(0, "Int", 1)

    # 2) Build 2-way interactions
    for a, b in model_spec.interaction2:
      model_matrix[f"{a}:{b}"] = model_matrix[a] * model_matrix[b]

    # 3) Build 3-way interactions
    for a, b, c in model_spec.interaction3:
      model_matrix[f"{a}:{b}:{c}"] = model_matrix[a] * model_matrix[b] * model_matrix[c]

    # 4) Build quadratics
    for a in model_spec.quadratic:
      if len(self._factors[a].levels) <= 2:
            raise KeyError(f"{a!r} has only {len(self._factors[a].levels)} levels; cannot fit a quadratic model")
      model_matrix[f"{a}^2"] = model_matrix[a] ** 2

    return model_matrix

  def _compute_leverage(self,
                        points: pd.DataFrame) -> np.ndarray:
    """
    Calculate the leverage values for given experimental points.
    
    Computes the hat matrix diagonals (leverage values) which measure the
    influence of each point on the fitted model. High leverage points are
    located at extreme positions in the design space and are candidates for
    replication to improve model precision.
    
    Parameters
    ----------
    points : pd.DataFrame
        Design points in coded units for which to calculate leverage.
    
    Returns
    -------
    np.ndarray
        Array of leverage values, one for each point.
        
    Raises
    ------
    ValueError
        If the model matrix has not been defined (call set_model_terms first).
        
    Notes
    -----
    Leverage is computed as h_i = x_i^T (X^T X)^+ x_i, where X is the model
    matrix and (X^T X)^+ is the Moore-Penrose pseudo-inverse for numerical
    stability.
    """

    # Ensure the design (model) matrix exists
    if self._model_matrix is None:
        raise ValueError("No model matrix defined, please call set_model_terms first.")

    # Extract the raw numpy array from the model matrix
    Xv = self._model_matrix.copy()
    # Build the model terms of the points provided
    x0 = self._build_model_matrix(points, self._model_spec)
    x0 = x0.reindex(columns=Xv.columns, fill_value=0.0).values
    Xv = Xv.values
    # Compute the Moore–Penrose pseudo-inverse of XTX for numerical stability
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    # Calculate the leverage for each point in points
    leverage = np.einsum("ij,jk,ik->i", x0, XtX_inv, x0)
    
    return leverage

  def _number_of_replicates(self) -> int:
    """
    Count the total number of replicate runs in the design.
    
    Computes the number of replicated experimental runs by identifying groups
    of identical design points and counting the additional replicates beyond
    the first occurrence.
    
    Returns
    -------
    int
        Total number of replicate runs (not including the original runs).
        
    Notes
    -----
    For example, if a design point appears 3 times, it contributes 2 replicates
    to the total count.
    """
    
    repl_groups = self._group_by_replicates()
    repl = 0
    for i in range(len(repl_groups)):
      repl += len(repl_groups[i]) - 1
    return repl
  
  def _group_by_replicates(self) -> List[List[int]]:
    """
    Identify and group replicated experimental runs.
    
    Scans the coded design matrix to find groups of identical design points
    (pure replicates). Each group contains the row indices of runs with
    identical factor level settings.
    
    Returns
    -------
    list of list of int
        List of replicate groups, where each group is a list of row indices
        corresponding to identical design points. Only groups with 2 or more
        points are included.
        
    Notes
    -----
    Replicates are identified by exact equality of all factor levels in the
    coded design matrix. This method is used for pure error estimation in
    lack-of-fit testing.
    """
    
    d_matrix = self._coded_design_matrix.copy()
    n_sample = d_matrix.shape[0]
    # Group candidate points by replicates
    replicate_groups = []
    used_indices = set()
    for i in range(n_sample):
      if i in used_indices:
        continue
      group = [i]
      used_indices.add(i)
      for j in range(i+1, n_sample):
        if j in used_indices:
          continue
        if np.array_equal(d_matrix.iloc[i], d_matrix.iloc[j]):
          group.append(j)
          used_indices.add(j)
      if len(group) > 1:
        replicate_groups.append(group)
        
    return replicate_groups

  def _mlr_predict(self,
                  matrix_to_pred: pd.DataFrame,
                  responses: List[str]) -> pd.DataFrame:
    """
    Predict response values using fitted MLR models.
    
    Internal method to generate predictions for specified responses using the
    fitted multiple linear regression models.
    
    Parameters
    ----------
    matrix_to_pred : pd.DataFrame
        Model matrix (with all terms) for prediction points.
    responses : list of str
        Names of response variables to predict.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing predicted values for each specified response.
        
    Notes
    -----
    This method assumes that MLR models have already been fitted for all
    requested response variables.
    """

    predictions = {}
    for variable in responses:
      y_pred = self._mlr_wrapper.results[variable].model.predict(matrix_to_pred)
      predictions[variable] = y_pred
    pred_df = matrix_to_pred.copy()
    for variable in responses:
      pred_df[variable] = predictions[variable]
    predicted_responses = pred_df[responses]
    
    return predicted_responses
  
  def _check_constant_levels(self,
                              design_matrix: pd.DataFrame,
                              x: str,
                              y: str,
                              z: Optional[str] = None,
                              constant_levels: Optional[Dict[str, float]] = None,
                              ) -> Dict[str, float]:
    """
    Validate and generate constant factor levels for visualization and optimization.
    
    Checks the validity of free variables (x, y, z) and constant level specifications,
    then generates default constant levels for all factors not being varied. Used
    primarily in contour plotting and response surface visualization.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        The coded design matrix for validation.
    x : str
        Name of the first free variable (x-axis).
    y : str
        Name of the second free variable (y-axis).
    z : str, optional
        Name of the third free variable (z-axis) for simplex plots, by default None.
    constant_levels : dict of {str: float}, optional
        User-specified constant levels for non-free factors, by default None.
    
    Returns
    -------
    dict of {str: float}
        Dictionary mapping factor names to their constant coded values.
        
    Raises
    ------
    ValueError
        If x, y, z are not distinct, not found in factors, have incompatible types,
        or if constant_levels has missing/extra keys or invalid values.
    TypeError
        If a continuous factor's constant level is not numeric.
        
    Notes
    -----
    When constant_levels is None, defaults are generated:
    - Continuous factors: 0 (center point)
    - Categorical factors: configured reference level
    - Mixture factors: simplex centroid considering lower bounds
    """

    factor_names = list(self._factors.keys())
    
    # basic checks on x and y and z (if provided)
    if x == y or x == z or y == z:
        raise ValueError("x, y, and z must be different variables.")
    if x not in factor_names or y not in factor_names or (z is not None and z not in factor_names):
        raise ValueError(f"x ({x!r}), y ({y!r}), and z ({z!r}) must be in {list(factor_names)}.")
    if z is None:
      if self._factors[x].type not in ["cont", "cat"] or self._factors[y].type not in ["cont", "cat"]:
          raise ValueError("For 2D designs, x and y must be continuous or categorical variables.")
    else:
      if self._factors[x].type != "mix" or self._factors[y].type != "mix" or self._factors[z].type != "mix":
          raise ValueError("For simplex designs, x, y, and z must be mixture variables.")
    if constant_levels is not None:
      for var in constant_levels.keys():
        if var not in factor_names:
            raise ValueError(f"Constant level variable {var!r} must be in {list(factor_names)}.")

    free_vars = {x, y, z} if z is not None else {x, y}
    mix_factors = [f for f in factor_names if self._factors[f].type == "mix"]
    
    if len(mix_factors) > 0:
        n_vars = len(mix_factors)
        lb = [self._factors[f].lower_bound for f in mix_factors]
        S = 1.0 - sum(lb)
        if S < -self.TOLERANCE:
            raise ValueError("Sum of lower_bounds must be ≤ 1.")
        S = max(S, 0.0)
        z = 1.0 / n_vars
    
    # If the user did not provide constant levels,
    # build a dictionary of center levels for each variable
    if constant_levels is None:
        # build defaults for *all other* variables
        out: dict[str, float] = {}
        for v in factor_names:
            if v in free_vars:
                continue
            if self._factors[v].type == "cont":
                out[v] = 0.0
            if self._factors[v].type == "cat":
                out[v] = self._factors[v].reference_code
            if self._factors[v].type == "mix":
                lb_v = self._factors[v].lower_bound
                out[v] = lb_v + S * z
        return out
      
    # Validate the provided constant levels
    provided = dict(constant_levels)

    # Ensure all required keys are present, and no unexpected keys
    expected_keys = [v for v in factor_names if v not in free_vars]
    missing = [v for v in expected_keys if v not in provided]
    extra   = [k for k in provided if k not in expected_keys]
    if missing or extra:
        parts = []
        if missing:
          parts.append(f"missing keys: {missing}")
        if extra:
          parts.append(f"unexpected keys: {extra}")
        raise ValueError(
            f"constant_levels must specify exactly {len(expected_keys)} entries "
            f"for variables {expected_keys}; " + "; ".join(parts)
        )

    # Validate each provided level
    for name, val in provided.items():
        if self._factors[name].type == "cont":
            # continuous: value must be in [-1, 1]
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name!r} level must be a number, got {type(val).__name__}.")
            lb = np.min(design_matrix[name])
            ub = np.max(design_matrix[name])
            if not (lb <= float(val) <= ub):
                raise ValueError(f"{name!r} level {val} not in [{lb}, {ub}].")
        if self._factors[name].type == "cat":
            # categorical: value must be among declared levels
            allowed = design_matrix[name].unique()
            if val not in allowed:
                raise ValueError(f"{name!r} level {val} not in {list(allowed)}.")
        elif self._factors[name].type == "mix":
            lb = np.min(design_matrix[name])
            ub = np.max(design_matrix[name])
            if not (lb <= float(val) <= ub):
                raise ValueError(f"{name!r} level {val} not in [{lb}, {ub}].")

    return provided
  
  # ==========================================================================
  #                             Public Methods
  # ==========================================================================
  
  # --------------------------- Setup Methods ------------------------------- #

  def set_model_terms(self,
                      terms: ModelTerms) -> None:
    """
    Specify the model terms for regression analysis.

    Compiles ``terms`` for the factors in this design and builds the model matrix
    used for fitting. Process-factor terms include main effects, interactions, and
    quadratics; mixture terms are specified through the Scheffe options on
    :class:`ModelTerms`.

    Args:
        terms (ModelTerms): Requested process and mixture model terms.

    Raises:
        ValueError: If the requested terms are invalid for the design or the
            compiled model has more terms than experimental runs.

    Notes:
        The model matrix is constructed from the coded design matrix. Calling this
        method again replaces the previous model specification and matrix.
    """
    
    pro_factors = [name for name, factor in self._factors.items() if factor.type in ["cont", "cat"]]
    mix_factors = [name for name, factor in self._factors.items() if factor.type == "mix"]

    self._model_spec = compile_model_spec(terms, pro_factors, mix_factors)
    
    if self._model_spec.model_terms > self._coded_design_matrix.shape[0]:
      raise ValueError(f"The number of model terms ({self._model_spec.model_terms}) exceeds the number of experiments ({self._coded_design_matrix.shape[0]}). Please reduce the model complexity.")

    # Build the model matrix with the specified terms
    self._model_matrix = self._build_model_matrix(self._coded_design_matrix, self._model_spec)
  
  def set_response_conditions(self,
                              lower_limits: List[Union[str, bool]],
                              upper_limits: List[Union[str, bool]],
                              maximize: List[bool]) -> None:
    """
    Define optimization criteria for response variables.

    The limits and objective direction are stored per response for Pareto analysis
    and plots that highlight response constraints. They are not required to fit an
    MLR model.

    Args:
        lower_limits (list[float | bool]): Lower limits in response-list order.
            Use ``False`` or ``None`` where no lower limit is required.
        upper_limits (list[float | bool]): Upper limits in response-list order.
            Use ``False`` or ``None`` where no upper limit is required.
        maximize (list[bool]): Whether to maximize (``True``) or minimize
            (``False``) each response.

    Raises:
        ValueError: If a supplied list does not have one entry for every response.

    Notes:
        Response names must already have been defined, normally by passing
        ``responses`` to :meth:`export_experiments`.
    """
    
    responses = self._response_list
    # Input checks
    if len(responses) != len(lower_limits) or len(responses) != len(upper_limits) or len(responses) != len(maximize):
      raise ValueError("The number of conditions must be the same")
    for i in range(len(responses)):
      # Initiate the dictionary
      self._response_conditions[responses[i]] = {}
      # Set lower limits
      self._response_conditions[responses[i]]["lower_limit"] = lower_limits[i]
      # Set upper limits
      self._response_conditions[responses[i]]["upper_limit"] = upper_limits[i]
      # Set "maximize" -> True / False
      self._response_conditions[responses[i]]["maximize"] = maximize[i]
  
  # ----------------------- Modify Design Methods ---------------------------- #
  
  def add_replicates(self, type: Literal["all", "leverage", "center", "manual"], n_replicates: int, indices: Optional[List[int]] = None) -> None:
    """
    Add replicate runs to the experimental design.
    
    Augments the design with additional experimental runs to improve precision
    of model estimates or pure error estimation. Supports multiple replication
    strategies.
    
    Parameters:
        type :
            Replication strategy:
                - 'all': Replicate all design points
                - 'leverage': Replicate high-leverage points only
                - 'center': Add center point replicates
                - 'manual': Replicate specific points by index
        n_replicates : 
            Number of replicates to add (interpretation depends on type).
        indices :
            Row indices of points to replicate (required when type='manual').
    Raises:
        ValueError: If n_replicates is negative or not an integer.
        ValueError: If type is not one of the allowed options.
        ValueError: If indices are not provided for manual replication.
        ValueError: If model matrix is not defined when using leverage replication.
        
    Notes:
        Replication updates both the coded and decoded design matrices. If a model
        matrix exists, it is automatically rebuilt to include the new runs.
        
        For type='leverage', points are selected based on their hat matrix diagonal
        values, with higher leverage points being more influential on model estimates.
    
    """
    
    if n_replicates < 0 or not isinstance(n_replicates, int):
      raise ValueError("Replicates must be a non-negative integer")
    if type not in ["all", "leverage", "center", "manual"]:
      raise ValueError("Type must be one of the following: all, leverage, center, manual")
    
    def append_coded_rows(extra: pd.DataFrame) -> None:
      
      self._coded_design_matrix = pd.concat(
          [self._coded_design_matrix, extra],
          ignore_index=True,
      )
      self._design_matrix = self._decode_matrix(self._coded_design_matrix)
      if self._model_matrix is not None:
          self._model_matrix = self._build_model_matrix(self._coded_design_matrix, self._model_spec)
          
    # Add replicates based on the specified type
    
    if type == "all":
        if n_replicates == 0:
            return
        extra = self._coded_design_matrix.loc[
            self._coded_design_matrix.index.repeat(n_replicates)
        ].reset_index(drop=True)
        append_coded_rows(extra)

    elif type == "leverage":
        if self._model_matrix is None:
            raise ValueError("No model matrix defined, please call set_model_terms first.")
        leverages = self._compute_leverage(self._coded_design_matrix)
        top_indices = np.argsort(leverages)[-n_replicates:]
        extra = self._coded_design_matrix.iloc[top_indices].copy()
        append_coded_rows(extra)

    elif type == "center":
        self._coded_design_matrix= self._add_center_points(self._coded_design_matrix, n_replicates)
        self._design_matrix = self._decode_matrix(self._coded_design_matrix)
        if self._model_matrix is not None:
            self._model_matrix = self._build_model_matrix(self._coded_design_matrix, self._model_spec)

    elif type == "manual":
        if indices is None or not isinstance(indices, list):
            raise ValueError("Please provide a list of indices for manual replicates")
        extra = self._coded_design_matrix.iloc[indices]
        extra = extra.loc[extra.index.repeat(n_replicates)].reset_index(drop=True)
        append_coded_rows(extra)
      
  # ------------------------ Export / Import Data ---------------------------- #

  def export_experiments(self,
                        responses: Optional[List[str]] = None,
                        randomize: bool = True,
                        destination: str | Path = "design_matrix.xlsx",
                        coded: bool = False) -> None:
    """
    Export the experimental design to an Excel file for laboratory execution.

    The execution order can be randomized to reduce systematic effects during
    laboratory work. Export is supported only for Excel workbooks; use an
    ``.xlsx`` path.

    Args:
        responses (list[str] | None): Names of response variables to add to the
            workbook. Provide these names when results will later be imported with
            :meth:`import_responses`; they are retained for validating the import.
            Defaults to ``None``.
        randomize (bool): Whether to randomize the execution order. Defaults to
            ``True``.
        destination (str | Path): Destination Excel workbook path. CSV export is not
            supported. Defaults to ``"design_matrix.xlsx"``.
        coded (bool): If ``True``, export coded factor values. If ``False``, export
            values in the original experimental units. Defaults to ``False``.

    Notes:
        The workbook contains the following columns:

        - ``Exp. Order`` is the planned execution sequence. Follow this column in
          the laboratory; it is randomized when ``randomize=True``.
        - ``Exp. Idx`` is the stable identifier of the corresponding internal design
          row. Do not edit it: :meth:`import_responses` uses it to restore the
          internal design order.
        - Factor columns contain the settings for each run.
        - Response columns contain the named measurements to complete after running
          the experiments.
    """
    
    self._response_list = responses
    # Define which matrix to export (design matrix or raw_design_matrix)
    if coded:
      d_matrix = self._coded_design_matrix.copy()
    else:
      d_matrix = self._design_matrix.copy()
    # Add Response columns to the design matrix to export
    if isinstance(responses, list):
      for response in responses:
        d_matrix[response] = np.zeros(d_matrix.shape[0])
    # Add the experimental order and randomize the experiments
    d_matrix.insert(0, self.EXP_IDX_COL, d_matrix.index)
    if randomize:
      random_idx = np.random.choice(d_matrix.index, size=len(d_matrix), replace=False)
      d_matrix.insert(0, self.EXP_ORDER_COL, random_idx)
      d_matrix = d_matrix.sort_values(by=self.EXP_ORDER_COL)
    else:
      d_matrix.insert(0, self.EXP_ORDER_COL, d_matrix.index)
    d_matrix.to_excel(destination, index=False)

  def import_responses(self, source: str | Path | pd.DataFrame) -> None:
    """
    Import experimental response data from a file.

    Response data can be read from an Excel or CSV file. For the standard export /
    import workflow, first call :meth:`export_experiments` with a defined
    ``responses`` list; those names determine which columns are imported.

    Args:
        source (str | Path | pd.DataFrame): Completed results as a DataFrame or
            an Excel/CSV file path.

    Raises:
        ValueError: If the number of rows does not match the design or expected
            response columns are missing.
        FileNotFoundError: If a supplied path does not exist.

    Notes:
        The results file must retain ``Exp. Idx`` and every response column supplied
        to :meth:`export_experiments`. ``Exp. Idx`` identifies the stable internal
        design row and is used to reorder results before they are stored. ``Exp.
        Order`` is the planned laboratory execution sequence; it is retained as
        metadata when present but does not determine response matching.
    """

    md_matrix = self.upload_file(source)
    # Sort values by the exp index -> reorder
    md_matrix = md_matrix.sort_values(by=self.EXP_IDX_COL).reset_index(drop=True)
            
    if md_matrix.shape[0] != self._coded_design_matrix.shape[0]:
      raise ValueError("Number of rows in the response source must match the number of experiments")
    
    # Validate response columns exist in the uploaded file
    missing_responses = [r for r in self._response_list if r not in md_matrix.columns]
    if missing_responses:
      raise ValueError(f"Response columns not found in source: {missing_responses}. Available columns: {list(md_matrix.columns)}")
    
    metadata_columns = [
      column
      for column in (self.EXP_ORDER_COL, self.EXP_IDX_COL)
      if column in md_matrix.columns
    ]
    self._experimental_metadata = (
      md_matrix[metadata_columns].copy()
      if metadata_columns
      else None
    )
    self._responses = md_matrix[self._response_list]


  # ---------------------------- MLR Model Computation ----------------------------- #

  def compute_mlr_model(self) -> None:
    
    """
    Fit multiple linear regression models to the response data.

    Fits one ordinary least-squares model for each imported response using the
    current model matrix. Replicate groups are identified automatically and used in
    the pure-error and lack-of-fit calculations when applicable.

    Raises:
        ValueError: If responses have not been imported or model terms have not
            been defined.

    Notes:
        Call :meth:`set_model_terms` and :meth:`import_responses` before fitting.
        The fitted models are then available to prediction and summary getters,
        and model-based plots.
    """
    
    if self._responses is None:
        raise ValueError("No responses defined, please import them first.")
    if self._model_matrix is None:
        raise ValueError("No model matrix defined, please define the model terms first.")
    
    replicate_groups = self._group_by_replicates()
    self._mlr_wrapper = self._analyzer.mlr_fit(self._model_matrix, self._responses, replicate_groups)

  # ---------------------------- MLR Model Prediction ----------------------------- #
  
  def _predict(self,
               matrix_to_pred: pd.DataFrame,
               responses) -> pd.DataFrame:
    
    """
    Predict response values at coded design points for internal consumers.

    Builds the configured model terms for each supplied point and returns point
    predictions from the fitted models.

    Args:
        matrix_to_pred (pd.DataFrame): Factor values for the points to predict.
            Process factors must use the coded scale of the design and the columns
            must cover the factors used by the fitted model.
        responses (list[str]): Names of the fitted responses to predict.

    Returns:
        pd.DataFrame: One row per prediction point and one column per requested
        response.

    Raises:
        ValueError: If no MLR model has been computed.

    Notes:
        This private helper returns point predictions only; user-facing predictions
        are provided by ``get_prediction_results``.
    """
    
    if self._mlr_wrapper is None:
        raise ValueError("No MLR model computed, please call compute_mlr_model first.")
    
    md_matrix_to_pred = self._build_model_matrix(matrix_to_pred, self._model_spec)
    
    predicted_responses = self._mlr_predict(md_matrix_to_pred, responses)
    
    return predicted_responses
