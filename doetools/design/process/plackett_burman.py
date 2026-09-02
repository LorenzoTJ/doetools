# Import necessary libraries
import pandas as pd
import numpy as np

# Import mixins and core abstract design class
from ...graphs import GraphsMixin
from ...utils import Design
from ...utils import CategoricalFactor
from ...utils import ModelSpec

# Class Definition
class PlackettBurmanDesign(Design, GraphsMixin):
  r"""
  Plackett-Burman Design Class.

  Plackett-Burman designs are highly efficient screening designs used to identify
  the most important factors among a large number of potential factors. 

  Parameters:
    factors (dict): Dictionary mapping factor names (str) to Factor objects. All factors must
        have exactly 2 levels. Supports 4 to 24 factors.
    center_points (int, optional): Number of center point replicates to add. Default is 0.
    replicates (int, optional): Number of replicate runs for each design point. Default is 0.

  Raises:
    ValueError: If number of factors is less than 4 or greater than 24.
    ValueError: If center_points is negative.
    ValueError: If replicates is negative.
    ValueError: If any factor does not have exactly 2 levels.

  Notes:
    Plackett-Burman designs provide estimates of main effects only, with all
    interactions assumed to be negligible. The number of runs is always a multiple
    of 4: n = 4, 8, 12, 16, 20, 24, etc.
    
    If the number of factors is less than (n-1), dummy factors are automatically
    added and labeled as 'd1', 'd2', etc.

    The design assumes a first-order model:
    
    .. math::
        y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \epsilon

  """

  @staticmethod
  def next_valid_pb_run_size(n_vars):
      """
      Calculate the minimum valid run size for a Plackett-Burman design.

      Parameters
      ----------
      n_vars : int
          Number of factors to screen.

      Returns
      -------
      int
          The minimum number of runs required (always a multiple of 4).

      Raises
      ------
      ValueError
          If n_vars is too large (>99).

      Notes
      -----
      Plackett-Burman designs require n runs where n is a multiple of 4,
      and n-1 ≥ number of factors.
      """
      for n in range(1, 100):
          if 4 * n > n_vars and (4 * n - 1) >= n_vars:
              return 4 * n
      raise ValueError("Too many factors")

  def __init__(self,
               factors : dict,
               center_points : int = 0,
               replicates : int = 0):

    super().__init__()
    
    vars = len(factors.keys())

    # -------------------------- INPUT CHECK --------------------------------- #
    if vars < 4 or vars > 24:
      raise ValueError("Plackett-Burman design requires at least 4 factors and at most 24 factors.")
    if center_points < 0:
      raise ValueError("Center points must be a non-negative integer")
    if replicates < 0:
      raise ValueError("Replicates must be a non-negative integer")
    for f in factors.values():
      if (f.type == "cat" and len(f.levels) != 2) or (f.type == "cont" and f.n_levels != 2):
        raise ValueError("All factors in a Plackett-Burman design must have 2 levels.") 

    # ---------------------- DESIGN INITIALIZATION --------------------------- #
    self._factors = factors
    self._design_type = "Plackett-Burman"
    self._coded_design_matrix = self.build_pb_design_matrix(factors = self._factors, center_points=center_points, replicates=replicates)
    self._design_matrix =  self._decode_matrix(self._coded_design_matrix)

  def build_pb_design_matrix(self, factors, center_points: int = 0, replicates: int = 0) -> pd.DataFrame:
    """
    Build the Plackett-Burman design matrix using predefined generator rows.

    Parameters
    ----------
    factors : dict
        Dictionary mapping factor names to Factor objects.
    center_points : int, optional
        Number of center point runs to add. Default is 0.
    replicates : int, optional
        Number of replicates for each design point. Default is 0.

    Returns
    -------
    pd.DataFrame
        Coded design matrix with columns for all factors (including dummy factors
        if needed), plus replicates and center points.

    Notes
    -----
    The design is constructed by circular permutation of a generator row specific
    to each run size (8, 12, 16, 20, 24). A final row of all -1 values completes
    the design. Dummy factors are added as needed to fill the design matrix.
    """

    var_names = list(factors.keys())
    
    k = len(var_names)

    # Numero di esperimenti da fare dato il numero di fattori

    n_runs = PlackettBurmanDesign.next_valid_pb_run_size(k)

    # Prima riga
    def generate_first_row(n):

        if n == 8:
            return [1,1,1,-1,1,-1,-1]
        if n == 12:
            return [1,1,-1,1,1,1,-1,-1,-1,1,-1]
        if n == 16:
            return [1,1,1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1]
        if n == 20:
            return [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1]
        if n == 24:
            return [1,1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]

    first_row = generate_first_row(n_runs)
    matrix = [first_row]

    # Costruzione delle righe successive per rotazione
    for i in range(1, n_runs - 1):
        matrix.append(first_row[-i:] + first_row[:-i])

    # Aggiunta della riga di -1
    matrix.append([-1] * (n_runs - 1))

    # Conversione a np
    pb_matrix = np.array(matrix)

    design_matrix = pd.DataFrame(pb_matrix)

    for i, var in enumerate(var_names):
      design_matrix.rename(columns={i: var}, inplace=True)

    for i in range(len(var_names), len(design_matrix.columns)):
      design_matrix.rename(columns={i: f"d{i+1}"}, inplace=True)
      self._factors[f"d{i+1}"] = CategoricalFactor(levels=[-1, 1])

    if replicates > 0:
      design_matrix = design_matrix.loc[design_matrix.index.repeat(replicates+1)].reset_index(drop=True)

    if center_points > 0:
      design_matrix = self._add_center_points(design_matrix, center_points)
      
    return design_matrix

  def set_model_terms(self,
                      intercept: bool = True):
    """
    Define the model terms for the Plackett-Burman design.

    Parameters
    ----------
    intercept : bool, optional
        Whether to include an intercept term in the model. Default is True.

    Notes
    -----
    Plackett-Burman designs are intended for main effects screening only.
    This method sets up a first-order model with main effects for all factors
    (including dummy factors). No interaction or quadratic terms are included.
    """

    main = list(self._factors.keys())
    
    self._model_spec = ModelSpec(
      intercept=intercept,
      main=main,
      interaction2= [],
      quadratic= [],
      interaction3= [])

    # Build the model matrix with the specified terms
    self._model_matrix = self._build_model_matrix(self._coded_design_matrix, self._model_spec)