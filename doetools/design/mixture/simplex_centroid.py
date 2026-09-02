# Import standard libraries
import itertools
import numpy as np
import pandas as pd

# Import doetools base classes
from ...utils import Design
from ...graphs import GraphsMixin
from ...utils import ParetoMixin
from ...utils import ModelTerms, compile_model_spec

# Initialize Simplex Centroid Design class
class SimplexCentroidDesign(Design, GraphsMixin, ParetoMixin):
    r"""
    Generate a simplex-centroid design for a mixture experiment.

    The design contains the centroids of every non-empty subset of components,
    including simplex vertices, edge midpoints, higher-dimensional face
    centroids, and the overall centroid.

    Parameters:
        factors (dict[str, MixtureFactor]): Mapping of component names to mixture
            factors. Lower bounds are applied through a pseudocomponent
            transformation. Upper bounds are not applied by this design.
        center_points (int, optional): Number of additional overall-centroid runs
            to append. The base design already contains one overall centroid.
            Defaults to 0.
        replicates (int, optional): Number of additional copies of every base
            design point. Defaults to 0.

    Raises:
        ValueError: If the sum of lower bounds exceeds 1 beyond numerical
            tolerance.
        ValueError: If ``center_points`` is not a non-negative integer.
        ValueError: If ``replicates`` is not a non-negative integer.

    Notes:
        For ``q`` mixture components, the base design contains

        .. math::
            N = 2^q - 1

        rows before additional center points or replication. With zero lower
        bounds, these rows consist of pure-component vertices and the centroids
        of every binary and higher-order component subset. Nonzero lower bounds
        transform the points into a reduced simplex, whose vertices no longer
        represent pure components.

        Use ``ConstrainedMixtureDesign`` when upper bounds must be enforced.
    """

    tol: float = 1e-8

    def __init__(self,
                 factors,
                 center_points : int = 0,
                 replicates : int = 0):
      
        super().__init__()

        if sum([factors[f].lower_bound for f in factors]) >= 1.0 + self.tol:
          raise ValueError("Sum of lower_bounds must be < 1.")
        if center_points < 0 or not isinstance(center_points, int):
            raise ValueError("Center points must be a non-negative integer")
        if replicates < 0 or not isinstance(replicates, int):
            raise ValueError("Replicates must be a non-negative integer")
          
        self._factors = factors
        self._coded_design_matrix = self.build_simplex_centroid_design_matrix(factors=factors,
                                                                        replicates=replicates,
                                                                        center_points=center_points)
        self._design_matrix = self._coded_design_matrix.copy()
        # Design type
        self._design_type = "Simplex Centroid"
        # Define mixture levels for each factor
        for f in self._factors:
            self._factors[f].levels = self._coded_design_matrix[f].unique().tolist()


    def build_simplex_centroid_design_matrix(self,
                            factors : dict[str, object],
                            replicates : int = 0,
                            center_points : int = 0) -> pd.DataFrame:
        """
        Build the simplex centroid design matrix for mixture experiments.

        Parameters
        ----------
        factors : dict
            Dictionary mapping factor names to MixtureFactor objects.
        replicates : int, optional
            Number of replicates for each design point. Default is 0.
        center_points : int, optional
            Number of overall centroid replicates. Default is 0.

        Returns
        -------
        pd.DataFrame
            Design matrix with mixture proportions. All rows sum to 1.0
            (within numerical tolerance).

        Notes
        -----
        The algorithm generates centroids for all possible subsets of components:
        1. For each subset size m (1 to k), generate all combinations of m components
        2. Set each selected component to 1/m in the unrestricted space
        3. Map to the constrained simplex if lower bounds are specified
        4. Clip any negative values to ensure feasibility
        """
      
        # Extract lower bounds and labels
        lower_bounds = [factors[f].lower_bound for f in factors]
        labels = factors.keys()
        # Extract dimensionality -> save in k
        k = len(lower_bounds)
        # Sum the lower bounds to define the slack
        L = np.asarray(lower_bounds, dtype=float)
        S = 1.0 - L.sum()
        rows = []
        # Build standard SCD in z-space, then map x = L + S*z
        # m can be 1, 2, ... k where k is the dimension
        for m in range(1, k + 1):
            # compute all the possible combinations of k objects (components of the mixture) in m spaces
            # m = 1, {0}, {1}, ..., {k} -> single component mixtures
            # m = 2, {0,1}, {0,2}, ... -> two component mixtures
            # m = k, {0,1,2,...,k} -> k component mixtures
            for T in itertools.combinations(range(k), m):
                z = np.zeros(k, dtype=float)
                # Update the components with 1/m
                z[list(T)] = 1.0 / m
                # Map each component back to the reduced simplex
                # This is a vectorized operation
                x = L + S * z
                rows.append(x)

        df = pd.DataFrame(np.vstack(rows), columns=labels)
        df = df.clip(lower=0.0)

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
            For mixture designs, typical terms include linear blending, quadratic
            blending, and special cubic terms.

        Raises
        ------
        ValueError
            If the number of model terms exceeds the number of experimental runs.

        Notes
        -----
        This method must be called before fitting models or performing analysis.
        
        For mixture experiments, the model uses Scheffé canonical polynomials:
        - Linear: βᵢxᵢ (main effects)
        - Quadratic: βᵢⱼxᵢxⱼ (binary blending)
        - Special cubic: βᵢⱼₖxᵢxⱼxₖ (ternary blending)

        The method automatically separates process factors (if any) from mixture
        factors to apply appropriate modeling strategies.
        """
        
        pro_factors = [name for name, factor in self._factors.items() if factor.type in ["cont", "cat"]]
        mix_factors = [name for name, factor in self._factors.items() if factor.type == "mix"]

        self._model_spec = compile_model_spec(terms, pro_factors, mix_factors)
        
        if self._model_spec.model_terms > self._coded_design_matrix.shape[0]:
          raise ValueError(f"The number of model terms ({self._model_spec.model_terms}) exceeds the number of experiments ({self._coded_design_matrix.shape[0]}). Please reduce the model complexity.")

        # Build the model matrix with the specified terms
        self._model_matrix = self._build_model_matrix(self._coded_design_matrix, self._model_spec)
