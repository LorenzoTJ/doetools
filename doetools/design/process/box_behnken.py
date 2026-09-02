# Import standard libraries
import itertools
import pandas as pd
from typing import List

# Import mixins and core abstract design class
from ...graphs import GraphsMixin
from ...utils import Design
from ...utils import ParetoMixin

# Class Definition
class BoxBehnkenDesign(Design, GraphsMixin, ParetoMixin):
  r"""
  Box-Behnken Design Class.

  Box-Behnken designs are efficient response surface designs that do not contain
  runs at the vertices of the cubic factor space. This makes them particularly
  useful when extreme combinations of factor levels are prohibitively expensive
  or physically impossible. All design points (except center points) are located
  at the midpoints of the edges of the factor space or at the center.

  Parameters: 
    factors (dict): Dictionary mapping factor names (str) to ContinuousFactor objects.
        All factors must be continuous with exactly 3 levels each.
        Minimum of 3 factors required.
    replicates (int, optional): Number of replicate runs for each design point (excluding center points).
        Default is 0.
    center_points (int, optional): Number of center point replicates. Default is 3. Center points are
        recommended for estimating pure error and checking curvature.

  Raises:
    ValueError: If fewer than 3 factors are provided.
    ValueError: If any factor does not have exactly 3 levels.
    ValueError: If any factor is not continuous.

  Notes: 
    The number of runs in a Box-Behnken design is:
    
    .. math::
        N = 2k(k-1) + c_p
    
    where k is the number of factors and c_p is the number of center points.

    Box-Behnken designs support fitting second-order polynomial models:
    
    .. math::
        y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2 + 
            \sum_{i<j} \beta_{ij} x_i x_j + \epsilon

    The design points are constructed by taking all combinations of:
    
    * Two factors at ±1 (high and low levels)
    * Remaining factors at 0 (middle level)

    This ensures no runs are conducted at extreme combinations where all factors
    are simultaneously at their high or low levels.
  
  """

  def __init__(self,
               factors : dict,
               replicates : int = 0,
               center_points : int = 3):

    super().__init__()

    # -------------------------- INPUT CHECK --------------------------------- #
    if len(factors.keys()) < 3:
      raise ValueError("Box Behnken design requires at least 3 factors.")
    for f in factors.values():
      if len(f.levels) != 3:
        raise ValueError("Box Behnken design requires all factors to have exactly 3 levels.")
      if f.type != "cont":
        raise ValueError("Box Behnken design requires all factors to be continuous.")

    # ----------------------- DESIGN INITIALIZATION -------------------------- #
    self._factors = factors
    self._design_type = "Box Behnken"
    self._coded_design_matrix = self.build_bb_design_matrix(factors=factors, center_points=center_points, replicates=replicates)
    self._design_matrix =  self._decode_matrix(self._coded_design_matrix)

  def build_bb_design_matrix(self, factors, center_points: int = 3, replicates: int = 0) -> pd.DataFrame:
    """
    Build the Box-Behnken design matrix with edge midpoints.

    Parameters
    ----------
    factors : dict
        Dictionary mapping factor names to ContinuousFactor objects.
    center_points : int, optional
        Number of center point runs to add. Default is 3.
    replicates : int, optional
        Number of replicates for each design point. Default is 0.

    Returns
    -------
    pd.DataFrame
        Coded design matrix with levels at -1, 0, and 1.

    Notes
    -----
    The design is constructed by:
    1. For each pair of factors (i, j), create all combinations of (±1, ±1)
       while holding other factors at 0
    2. This generates the midpoints of all edges of the factor space hypercube
    3. Add replicates of all runs if specified
    4. Add center points where all factors are at 0

    For k factors, this creates 2k(k-1) unique design points before replication.
    """
    # 1) Extract the number of variables
    k = len(factors.keys())
    # 2) If k == 2 the design matrix is pre-built
    if k == 2:
      runs = [[1,0], [-1,0], [0,1], [0,-1]]
    # 3) If k != 2 -> build the design matrix
    else:
      # 4) Build the design_matrix rows
      runs: List[List[int]] = []
      # 5) for each pair of factors, vary them over ±1 and hold others at 0
      # es. k = 3, [(0,1), (0,2), (1,2)]
      # es. k = 4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
      for i, j in itertools.combinations(range(k), 2):
          # (-1,-1), (-1, 1), (1, -1), (1,1)
          for li, lj in itertools.product([-1, 1], repeat=2):
              row = [0] * k
              row[i], row[j] = li, lj
              runs.append(row)
    design_matrix = pd.DataFrame(runs, columns=factors.keys())
    if replicates > 0:
      design_matrix = design_matrix.loc[design_matrix.index.repeat(replicates+1)].reset_index(drop=True)
    # 6) Add Center Points
    if center_points > 0:
      design_matrix = self._add_center_points(design_matrix, center_points)

    return design_matrix