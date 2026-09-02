# Import standard libraries
import pandas as pd
import numpy as np
from itertools import product

# Import mixins and core abstract design class
from ...graphs import GraphsMixin
from ...utils import Design
from ...utils import ParetoMixin

# Class Definition
class CentralCompositeDesign(Design, GraphsMixin, ParetoMixin):
  r"""
  Central Composite Design (CCD) Class.

  Central Composite Designs are widely used for response surface modeling,
  allowing estimation of quadratic models with fewer runs than a full factorial
  at three levels. CCDs consist of three types of points: factorial points,
  axial (star) points, and center points.

  Parameters:
    factors (dict): Dictionary mapping factor names (str) to ContinuousFactor objects.
        All factors must be continuous. Minimum of 2 factors required.
    design ({'ccc', 'ccf', 'cci'}): Type of central composite design:
        
        - 'ccc' (default): Circumscribed CCD. Axial points are outside the factorial space at distance alpha = (2^k)^(1/4) from center, making it rotatable.
        - 'ccf': Face-centered CCD. Axial points are at the faces of the factorial space (alpha = 1). Requires fewer factor levels.
        - 'cci': Inscribed CCD. Factorial points are inside the space at distance 1/alpha from center, with axial points at ±1. Useful when factor limits cannot be exceeded.
    center_points (int, optional): Number of center point replicates. Default is 1. Center points estimate
        pure error and improve prediction at the center of the design space.
    replicates (int, optional): Number of replicate runs for each design point (excluding center points).
        Default is 0.

  Raises:
     ValueError: If center_points is negative.
     ValueError: If design is not one of 'ccc', 'ccf', or 'cci'.
     ValueError: If fewer than 2 factors are provided.
     ValueError: If any factor is not continuous.

  Notes:
    The number of runs in a CCD is:
    
    .. math::
        N = 2^k + 2k + c_p
    
    where k is the number of factors and c_p is the number of center points.

    Central Composite Designs support fitting second-order polynomial models:
    
    .. math::
        y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2 + 
            \sum_{i<j} \beta_{ij} x_i x_j + \epsilon
  
  """

  def __init__(self,
               factors : dict,
               design: str = "ccc",
               center_points : int = 1, 
               replicates : int = 0):

    super().__init__()

    # -------------------------- INPUT CHECK --------------------------------- #
    if center_points < 0:
        raise ValueError("Center points must be a non-negative integer")
    if design not in ["ccc", "ccf", "cci"]:
        raise ValueError("Design must be ccc, ccf, or cci")
    if len(factors.keys()) < 2:
        raise ValueError("Central Composite design requires at least 2 factors.")
    for f in factors.values():
        if f.type != "cont":
            raise ValueError("Central Composite design requires all factors to be continuous.")

    # ----------------------- DESIGN INITIALIZATION -------------------------- #
    self._factors = factors
    self._type = design
    self._design_type = "Central Composite"
    coded_design_matrix = self.build_ccd_design_matrix(factors=factors, center_points=center_points, replicates=replicates)
    design_matrix =  self._decode_matrix(coded_design_matrix)
    for col in design_matrix.columns:
      design_matrix[col] = np.round(design_matrix[col], decimals=self._factors[col].decimals)
    self._design_matrix = design_matrix
    self._coded_design_matrix = self._code_matrix(design_matrix)
    for col in self._design_matrix.columns:
        self._factors[col].levels = np.sort(self._design_matrix[col].unique()).tolist()
        self._factors[col].coded_levels = np.sort(self._coded_design_matrix[col].unique()).tolist()
        self._factors[col].n_levels = len(self._factors[col].levels)

  def build_ccd_design_matrix(self, factors, center_points: int = 1, replicates: int = 0) -> pd.DataFrame:
    """
    Build the central composite design matrix based on the specified design type.

    Parameters
    ----------
    factors : dict
        Dictionary mapping factor names to ContinuousFactor objects.
    center_points : int, optional
        Number of center point runs to add. Default is 1.
    replicates : int, optional
        Number of replicates for each design point. Default is 0.

    Returns
    -------
    pd.DataFrame
        Coded design matrix containing factorial points, axial points, replicates,
        and center points.

    Notes
    -----
    The method constructs three distinct design types:
    
    1. CCC (Circumscribed): Factorial at ±1, axial at ±alpha where alpha = (2^k)^(1/4)
    2. CCF (Face-centered): Factorial at ±1, axial at ±1
    3. CCI (Inscribed): Factorial at ±1/alpha, axial at ±1 where alpha = (2^k)^(1/4)
    
    Each design consists of:
    - 2^k factorial points (vertices of a hypercube)
    - 2k axial points (one on each axis)
    - Center point replicates
    """

    k = len(factors.keys())
    variables_name = list(factors.keys())
    # ------------------------------ ccc : Central Composite Circumscripted ----------------------------------#
    if self._type == "ccc":
        # Factorial Points
        # All the possible combinations of (-1, 1) in k spaces
        factorial_points = np.array(list(product([-1, 1], repeat=k)))
        # Axial Points
        # Two Axial Points each axe, one positive and one negative
        # (alpha, 0*k)
        alpha = (2**k)**(1/4)
        axial_points = []
        for i in range(k):
            pt_pos = np.zeros(k)
            pt_neg = np.zeros(k)
            pt_pos[i] = alpha
            pt_neg[i] = -alpha
            axial_points.extend([pt_pos, pt_neg])
        axial_points = np.array(axial_points)
        design_coded = pd.DataFrame(
            np.vstack([factorial_points, axial_points]),
            columns=variables_name
        )
        design_matrix = design_coded
        if replicates > 0:
            design_matrix = design_matrix.loc[design_matrix.index.repeat(replicates+1)].reset_index(drop=True)
        # Add Center Points
        if center_points > 0:
            design_matrix = self._add_center_points(design_matrix, center_points)

    # ------------------------------ ccf : Central Composite Face Centered ---------------------------------- #

    elif self._type == "ccf":
        # Factorial Points
        # All the possible combinations of (-1, 1) in k spaces
        factorial_points = np.array(list(product([-1, 1], repeat=k)))
        # Axial Points
        # Two Axial Points each axe, one positive and one negative
        # (1, 0*k)
        alpha = 1
        axial_points = []
        for i in range(k):
            pt_pos = np.zeros(k)
            pt_neg = np.zeros(k)
            pt_pos[i] = alpha
            pt_neg[i] = -alpha
            axial_points.extend([pt_pos, pt_neg])
        axial_points = np.array(axial_points)
        # Center Points #
        design_coded = pd.DataFrame(
            np.vstack([factorial_points, axial_points]),
            columns=variables_name
        )
        design_matrix = design_coded
        if replicates > 0:
            design_matrix = design_matrix.loc[design_matrix.index.repeat(replicates+1)].reset_index(drop=True)
        # Add Center Points
        if center_points > 0:
            design_matrix = self._add_center_points(design_matrix, center_points)

    # --------------------------------- cci : Central Composite incribed ----------------------------------- #

    elif self._type == "cci":
        # Factorial Points
        # All the possible combinations of (-alpha, alpha) in k spaces
        alpha = (2**k)**(1/4)
        factorial_points = np.array(list(product([-1/alpha, 1/alpha], repeat=k)))
        # Axial Points
        # Two Axial Points each axe, one positive and one negative
        # (1, 0*k)
        alpha = 1
        axial_points = []
        for i in range(k):
            pt_pos = np.zeros(k)
            pt_neg = np.zeros(k)
            pt_pos[i] = alpha
            pt_neg[i] = -alpha
            axial_points.extend([pt_pos, pt_neg])
        axial_points = np.array(axial_points)
        # Center Points #
        design_coded = pd.DataFrame(
            np.vstack([factorial_points, axial_points]),
            columns=variables_name
        )
        design_matrix = design_coded
        if replicates > 0:
            design_matrix = design_matrix.loc[design_matrix.index.repeat(replicates+1)].reset_index(drop=True)
        # Add Center Points
        if center_points > 0:
            design_matrix = self._add_center_points(design_matrix,  center_points)

    return design_matrix

