# Import necessary libraries
import itertools
import pandas as pd
import doe_box as dbox
import string

# Import mixins and core abstract design class
from ...graphs import GraphsMixin
from ...utils import Design

# Class Definition
class FractionalFactorialDesign(Design, GraphsMixin):
  r"""
  Fractional Factorial Design Class.

  Fractional factorial designs use a carefully chosen subset of the full factorial
  design to reduce the number of experimental runs while maintaining the ability
  to estimate important effects. These designs are specified by their resolution,
  which determines which effects are confounded (aliased) with each other.

  Parameters:
    factors (dict): Dictionary mapping factor names (str) to Factor objects. All factors must
        have exactly 2 levels.
    resolution (int): The resolution of the fractional factorial design. Common values:
    
        - Resolution III: Main effects are not aliased with each other but may be
          aliased with two-factor interactions.
        - Resolution IV: Main effects are clear of two-factor interactions, but
          two-factor interactions may be aliased with each other.
        - Resolution V: Main effects and two-factor interactions are clear of each
          other but may be aliased with three-factor interactions.
    center_points (int, optional):
        Number of center point replicates to add. Default is 0.
    replicates (int, optional):
        Number of replicate runs for each design point. Default is 0.

  Raises:
    ValueError: If center_points is negative or not an integer.
    ValueError: If replicates is negative or not an integer.
    ValueError: If any factor does not have exactly 2 levels.
    ValueError: If more than 26 factors are specified (due to single-letter mapping).

  Notes: 
    This implementation uses the doe_box library to generate the fractional factorial
    design based on the specified resolution.

  """
  def __init__(self,
               factors : dict,
               resolution : int,
               center_points : int = 0,
               replicates : int = 0):

    super().__init__()

    # ---------------------------- INPUT CHECK ------------------------------- #
    if center_points < 0 or not isinstance(center_points, int):
      raise ValueError("Center points must be a non-negative integer")
    if replicates < 0 or not isinstance(replicates, int):
      raise ValueError("Replicates must be a non-negative integer")
    levels = [factors[f].n_levels if factors[f].type == "cont" else len(factors[f].levels) for f in factors]
    if any(lv != 2 for lv in levels):
      raise ValueError("Fractional Factorial Design only supports 2-level factors")

    # ----------------------- DESIGN INITIALIZATION -------------------------- #
    self._factors = factors
    self._resolution = resolution
    self._design_type = "Fractional Factorial"
    self._coded_design_matrix = self.build_design_matrix(center_points=center_points, replicates=replicates)
    self._design_matrix =  self._decode_matrix(self._coded_design_matrix)
    self._aliases : dict = None

  def build_design_matrix(self, center_points: int = 0, replicates: int = 0) -> pd.DataFrame:
    """
    Build the fractional factorial design matrix using doe_box generators.

    Parameters
    ----------
    center_points : int, optional
        Number of center point runs to add. Default is 0.
    replicates : int, optional
        Number of replicates for each design point. Default is 0.

    Returns
    -------
    pd.DataFrame
        Coded design matrix containing the fractional factorial runs, replicates,
        and center points.

    Notes
    -----
    Factor names are temporarily mapped to single letters (a-z) for compatibility
    with doe_box, then remapped back to original names in the final DataFrame.
    """
    
    # 1) Map factor names to single-letter representations
    n = len(self._factors.keys())
    if n > 26:
        raise ValueError("Cannot handle more than 26 factors with single-letter mapping.")

    letters = list(string.ascii_lowercase[:n])
    factor_to_letter = dict(zip(self._factors.keys(), letters))
    letter_to_factor = {v: k for k, v in factor_to_letter.items()}

    # 2) Build term string for generator
    terms = " ".join(letters)

    # 3) Generate design in coded units via dbox
    gen = dbox.fracfactgen(terms=terms, resolution=self._resolution)
    df_letters = pd.DataFrame(dbox.fracfact(gen), columns=letters)
    # 4) Remap columns back to original factor names
    design_matrix = df_letters.rename(columns=letter_to_factor)
    if replicates > 0:
      design_matrix = design_matrix.loc[design_matrix.index.repeat(replicates+1)].reset_index(drop=True)
    if center_points > 0:
      design_matrix = self._add_center_points(design_matrix, center_points)

    return design_matrix

  def alias_check(self):
    """
    Check for aliased (confounded) terms in the model matrix.

    Raises
    ------
    ValueError
        If no model matrix has been defined.

    Notes
    -----
    This method compares all pairs of columns in the model matrix to identify
    terms that are perfectly correlated (either positively or negatively).
    Aliased relationships are stored in the _aliases attribute and printed
    to the console.

    Aliasing occurs when two or more effects cannot be estimated independently
    because they have identical (or opposite) patterns in the design matrix.
    """

    if self._model_matrix is None:
        raise ValueError("No model matrix defined, please compute the model matrix")
    model_matrix = self._model_matrix.copy()

    aliases = {}
    # Compare columns pairwise for exact equality
    for col1, col2 in itertools.combinations(model_matrix.columns, 2):
        if model_matrix[col1].equals(model_matrix[col2]) or model_matrix[col1].equals(-model_matrix[col2]):
            # Record alias relationship (map the later term to the first)
            aliases[col2] = col1

    self._aliases = aliases

    print(self._aliases)