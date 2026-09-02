# Import standard libraries
import itertools
import pandas as pd

# Import doetools base classes
from ...utils import Design
from ...graphs import GraphsMixin
from ...utils import ParetoMixin
from ...utils.model_spec import ModelTerms, compile_model_spec

# Initialize Simplex Lattice Design class
class SimplexLatticeDesign(Design, GraphsMixin, ParetoMixin):
    r"""
    Generate a simplex-lattice design for a mixture experiment.

    Simplex lattice designs create a uniformly spaced grid on the mixture simplex,
    with spacing determined by ``m``.

    Parameters:
        factors (dict[str, MixtureFactor]): Mapping of component names to mixture
            factors. Lower bounds are applied through a pseudocomponent
            transformation. Upper bounds are not applied by this design.
        m (int, optional): Lattice degree. In the unrestricted simplex, each
            component takes values ``0/m, 1/m, ..., m/m``. Larger values produce
            a finer grid and more runs. Defaults to 1.
        center_points (int, optional): Number of overall-centroid runs to append.
            Defaults to 0.
        replicates (int, optional): Number of additional copies of every lattice
            point. Defaults to 0.

    Raises:
        ValueError: If ``m`` is not a positive integer.
        ValueError: If the sum of lower bounds is greater than or equal to 1.
        ValueError: If ``center_points`` is not a non-negative integer.
        ValueError: If ``replicates`` is not a non-negative integer.

    Notes:
        For ``q`` mixture components and lattice degree ``m``, the base design
        contains

        .. math::
            N = \binom{q + m - 1}{m}

        points before additional center points or replication. With nonzero lower
        bounds, the lattice is mapped into a reduced simplex and its component
        values are transformed accordingly.

        Use ``ConstrainedMixtureDesign`` when upper bounds must be enforced.
    """

    def __init__(self,
                 factors,
                 m : int = 1,
                 center_points : int = 0,
                 replicates : int = 0):
      
        super().__init__()

        # Input validation
        if m < 1 or not isinstance(m, int):
            raise ValueError("m must be a positive integer equal or greater than 1.")
        if sum([factors[f].lower_bound for f in factors]) >= 1.0:
          raise ValueError("Sum of lower_bounds must be < 1.")
        if center_points < 0 or not isinstance(center_points, int):
            raise ValueError("Center points must be a non-negative integer")
        if replicates < 0 or not isinstance(replicates, int):
            raise ValueError("Replicates must be a non-negative integer")
        # Design matrix generation
        self._factors = factors
        self._coded_design_matrix = self.build_simplex_lattice_design_matrix(replicates=replicates,
                                                                       center_points=center_points,
                                                                       m = m)
        self._design_matrix = self._coded_design_matrix.copy()
        # Design type
        self._design_type = "Simplex Lattice"
        # Define mixture levels for each factor
        for f in self._factors:
            self._factors[f].levels = self._coded_design_matrix[f].unique().tolist()

    # Design matrix builder
    def build_simplex_lattice_design_matrix(self,
                            center_points : int = 0,
                            replicates : int = 0,
                            m : int = 1) -> pd.DataFrame:
        """
        Build the simplex lattice design matrix with uniform spacing.

        Parameters
        ----------
        center_points : int, optional
            Number of overall centroid replicates. Default is 0.
        replicates : int, optional
            Number of replicates for each design point. Default is 0.
        m : int, optional
            Lattice degree (spacing parameter). Default is 1.

        Returns
        -------
        pd.DataFrame
            Design matrix with mixture proportions uniformly spaced on the simplex.
            All rows sum to 1.0 (within numerical tolerance).

        Notes
        -----
        The algorithm generates all integer combinations (r1, r2, ..., rk)
        where each ri is in {0, 1, ..., m} and sum(ri) = m.
        These are then divided by m to create proportions.
        
        If lower bounds exist, points are transformed from the unrestricted
        simplex z-space to the constrained simplex x-space.
        """
            
        # Save: name of the variables / dimension / m / lower_bounds
        names = self._factors.keys()
        k = len(names)
        # Extract the lower bounds as a list
        lb = [self._factors[name].lower_bound for name in names]
        if any(lv > 0 for lv in lb):
            # Calculate the slack
            S = 1.0 - sum(lb)
            if S < -1e-12:
                raise ValueError("Sum of lower_bounds must be < 1.")
            S = max(S, 0.0)
            z = []
            # Build points as seen before
            for r in itertools.product(range(m+1), repeat=k):
                if sum(r) == m:
                    z.append([ri/m for ri in r])
            z = pd.DataFrame(z, columns=names)
            x = z.copy()
            # Map them back to the simplex
            for name in names:
                x[name] = self._factors[name].lower_bound + S * z[name]
            df = pd.DataFrame(x, columns=names)
        else:
            grid = []
            # Take all the combinations of points on the axes
            for r in itertools.product(range(m+1), repeat=k):
                # Filter the combinations that sum to m
                if sum(r) == m:
                    # Divide by m to obtain points on the simplex
                    grid.append([ri/m for ri in r])
            df =  pd.DataFrame(grid, columns=names)

        if replicates > 0:
            df = df.loc[df.index.repeat(replicates+1)].reset_index(drop=True)
        if center_points > 0:
            df = self._add_center_points(df, center_points)
        return df

    def set_model_terms(self,
                          terms = ModelTerms) -> None:
        """
        Define the mixture model terms to be estimated from the experimental design.

        Parameters
        ----------
        terms : ModelTerms
            ModelTerms object specifying which Scheffé mixture model terms to include.
            The lattice degree m determines which polynomial degree can be estimated.

        Raises
        ------
        ValueError
            If the number of model terms exceeds the number of experimental runs.

        Notes
        -----
        This method must be called before fitting models or performing analysis.
        
        The simplex lattice design of degree m supports fitting mixture models
        up to degree m:
        - m=1: Linear model (main effects only)
        - m=2: Quadratic model (main + binary blending)
        - m=3: Special cubic model (main + binary + ternary blending)

        The design provides the minimum number of points needed to estimate
        a Scheffé canonical polynomial of the specified degree.

        """
        
        pro_factors = [name for name, factor in self._factors.items() if factor.type in ["cont", "cat"]]
        mix_factors = [name for name, factor in self._factors.items() if factor.type == "mix"]

        self._model_spec = compile_model_spec(terms, pro_factors, mix_factors)
        
        if self._model_spec.model_terms > self._coded_design_matrix.shape[0]:
          raise ValueError(f"The number of model terms ({self._model_spec.model_terms}) exceeds the number of experiments ({self._coded_design_matrix.shape[0]}). Please reduce the model complexity.")

        # Build the model matrix with the specified terms
        self._model_matrix = self._build_model_matrix(self._coded_design_matrix, self._model_spec)
