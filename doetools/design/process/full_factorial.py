# Import necessary libraries
import itertools
import pandas as pd

# Import mixins and core abstract design class
from ...graphs import GraphsMixin
from ...utils import Design
from ...utils import ParetoMixin    

# Class Definition
class FullFactorialDesign(Design, GraphsMixin, ParetoMixin):
  r"""
  Full Factorial Design Class.

  A full factorial design systematically evaluates every combination of the selected
  factor levels, enabling the estimation of main effects and interactions between factors.
  It is best suited to experiments with relatively few factors and levels, as the number of
  required runs grows rapidly with design size—for example, a two-level design requires 
  \(2^k\) runs for \(k\) factors.

  Parameters:
    factors (dict): Dictionary mapping factor names (str) to Factor objects (ContinuousFactor or
        CategoricalFactor). Each factor defines the levels to be tested.
    center_points (int, optional): Number of center point replicates to add to the design. Center points are
        runs where all continuous factors are set to their middle level and categorical
        factors are set to their reference level. Default is 0.
    replicates (int, optional): Number of replicate runs for each design point (excluding center points).
        If replicates=1, each design point is run twice. Default is 0 (no replication).

  Raises:
    ValueError: If center_points is negative or not an integer.
    ValueError: If replicates is negative or not an integer.

  Notes
  -----
  The number of runs in a full factorial design is:
  
  .. math::
      N = \prod_{i=1}^{k} l_i \times (r + 1) + c
  
  where k is the number of factors, l_i is the number of levels for factor i,
  r is the number of replicates, and c is the number of center points.

  """

  def __init__(self,
               factors : dict,
               center_points : int = 0,
               replicates : int = 0):

    super().__init__()

    # ---------------------------- INPUT CHECK ------------------------------- #
    if center_points < 0 or not isinstance(center_points, int):
      raise ValueError("Center points must be a non-negative integer")
    if replicates < 0 or not isinstance(replicates, int):
      raise ValueError("Replicates must be a non-negative integer")

    # ----------------------- DESIGN INITIALIZATION -------------------------- #
    self._factors = factors
    self._design_type = "Full Factorial"
    self._coded_design_matrix = self.build_full_fact_design_matrix(factors = self._factors, center_points=center_points, replicates=replicates)
    self._design_matrix =  self._decode_matrix(self._coded_design_matrix)

  def build_full_fact_design_matrix(self, factors, center_points: int = 0, replicates: int = 0) -> pd.DataFrame:
    """
    Build the full factorial design matrix with all factor level combinations.

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
        Coded design matrix containing all factorial combinations, replicates,
        and center points.

    Notes
    -----
    The method generates all possible combinations using itertools.product,
    then adds replicates by duplicating rows, and finally appends center points.
    """
    # 1) Extract factor names
    factor_names = list(factors.keys())
    # 2) Build raw full‐factorial grid (all the combinations)
    grid = list(itertools.product(*(factors[f].coded_levels for f in factor_names)))
    design_matrix = pd.DataFrame(grid, columns=factor_names)
    # 3) Add replicates, duplicating the rows
    if replicates > 0:
      design_matrix = design_matrix.loc[design_matrix.index.repeat(replicates+1)].reset_index(drop=True)
    # 4) Add center points as -1 if "cat" or 0 if "cont"
    if center_points > 0:
      design_matrix = self._add_center_points(design_matrix, center_points)
    return design_matrix
  
  

  