from typing import Collection

# Import standard libraries
import numpy as np
import pandas as pd

# Import candidate point builder
from .mixture_cp_generator import build_candidate_points

# Import doetools base classes
from ...graphs import GraphsMixin
from ...utils import Design, ModelTerms, ParetoMixin, compile_model_spec


# Initialize Constrained Mixture Design class
class ConstrainedMixtureDesign(Design, GraphsMixin, ParetoMixin):
    r"""
    Generate a design for a constrained mixture region.

    The design is generated from the geometry of the feasible mixture region
    defined by component lower and upper bounds and ``sum(x_i) = 1``. The
    default structure includes vertices, true edge midpoints, and true face
    centroids.

    Parameters:
        factors (dict[str, MixtureFactor]): Mapping of component names to mixture
            factors that define the feasible region.
        structure (str | Collection[str], optional): Geometric structures to
            include. Use ``"complete"`` for vertices, edge midpoints, and face
            centroids, or pass one name or a collection containing
            ``"vertices"``, ``"edge_midpoints"``, and ``"face_centroids"``.
            Singular aliases are accepted. Defaults to ``"complete"``.
        center_points (int, optional): Number of feasible-region centroid rows to
            append. In a three-component complete design, the face centroid is
            already the region centroid, so this adds replicates of that point.
            Defaults to 0.
        replicates (int, optional): Number of additional copies of each generated
            geometric point. Center points are appended after replication and are
            not affected by this parameter. Defaults to 0.

    Raises:
        NotImplementedError: If the mixture does not contain three or four
            components.
        ValueError: If a factor is not a ``MixtureFactor``.
        ValueError: If the component bounds do not define a feasible region.
        ValueError: If ``structure`` contains an unsupported value.
        ValueError: If ``center_points`` or ``replicates`` is not a non-negative
            integer.

    Examples:
        >>> from doetools import ConstrainedMixtureDesign, MixtureFactor
        >>> factors = {
        ...     "A": MixtureFactor(lower_bound=0.0, upper_bound=0.7),
        ...     "B": MixtureFactor(lower_bound=0.1, upper_bound=0.8),
        ...     "C": MixtureFactor(lower_bound=0.1, upper_bound=0.7),
        ... }
        >>> design = ConstrainedMixtureDesign(factors, structure="vertices")
    """

    tol: float = 1e-8

    def __init__(
        self,
        factors,
        structure: str | Collection[str] = "complete",
        center_points: int = 0,
        replicates: int = 0,
    ):
        super().__init__()

        include = self._normalize_structure(structure)
        self._validate_inputs(factors, center_points, replicates)

        self._factors = factors
        self._coded_design_matrix = self.build_constrained_mixture_design_matrix(
            factors=factors,
            include=include,
            center_points=center_points,
            replicates=replicates,
        )
        self._design_matrix = self._coded_design_matrix.copy()
        self._design_type = "Constrained Mixture"

        for name in self._factors:
            self._factors[name].levels = self._coded_design_matrix[name].unique().tolist()

    def _normalize_structure(self, structure: str | Collection[str]) -> tuple[str, ...]:
        if isinstance(structure, str):
            if structure == "complete":
                return ("vertices", "edge_midpoints", "face_centroids")
            requested = (structure,)
        else:
            requested = tuple(structure)

        aliases = {
            "vertex": "vertices",
            "vertices": "vertices",
            "edge_midpoint": "edge_midpoints",
            "edge_midpoints": "edge_midpoints",
            "face_centroid": "face_centroids",
            "face_centroids": "face_centroids",
        }
        unknown = sorted(set(requested) - set(aliases))
        if unknown:
            raise ValueError(
                "Unsupported constrained mixture structure: "
                + ", ".join(unknown)
                + ". Use 'complete', 'vertices', 'edge_midpoints', or 'face_centroid'."
            )

        include = []
        for item in requested:
            category = aliases[item]
            if category not in include:
                include.append(category)
        return tuple(include)

    def _validate_inputs(
        self,
        factors,
        center_points: int,
        replicates: int,
    ) -> None:
        if len(factors) not in {3, 4}:
            raise NotImplementedError(
                "ConstrainedMixtureDesign currently supports only three- or "
                "four-component mixtures."
            )
        if any(factor.type != "mix" for factor in factors.values()):
            raise ValueError("ConstrainedMixtureDesign supports only mixture factors.")
        if sum(factors[name].lower_bound for name in factors) > 1.0 + self.tol:
            raise ValueError("Sum of lower_bounds must be <= 1.")
        if sum(factors[name].upper_bound for name in factors) < 1.0 - self.tol:
            raise ValueError("Sum of upper_bounds must be >= 1.")
        if any(factors[name].lower_bound > factors[name].upper_bound for name in factors):
            raise ValueError("Each lower_bound must be less than or equal to upper_bound.")
        if center_points < 0 or not isinstance(center_points, int):
            raise ValueError("Center points must be a non-negative integer.")
        if replicates < 0 or not isinstance(replicates, int):
            raise ValueError("Replicates must be a non-negative integer.")

    def build_constrained_mixture_design_matrix(
        self,
        factors: dict[str, object],
        include: Collection[str] | None = None,
        center_points: int = 0,
        replicates: int = 0,
    ) -> pd.DataFrame:
        """
        Build the constrained mixture design matrix.

        The base design contains only geometric boundary candidates:
        vertices, true edge midpoints, and true face centroids. It excludes
        grid points and excludes the global centroid unless ``center_points``
        is requested.
        """

        names = list(factors.keys())
        lower_bounds = [factors[name].lower_bound for name in names]
        upper_bounds = [factors[name].upper_bound for name in names]

        candidates = build_candidate_points(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            component_names=names,
            include=include or {"vertices", "edge_midpoints", "face_centroids"},
            grid=None,
            tolerance=self.tol,
        )
        df = candidates[names].reset_index(drop=True)

        if replicates > 0:
            df = df.loc[df.index.repeat(replicates + 1)].reset_index(drop=True)
        if center_points > 0:
            df = self._add_center_points(df, center_points)
        self._validate_design_matrix_bounds(df, factors)
        return df

    def _add_center_points(
        self,
        design_matrix: pd.DataFrame,
        center_points: int,
    ) -> pd.DataFrame:
        names = list(design_matrix.columns)
        lower_bounds = [self._factors[name].lower_bound for name in names]
        upper_bounds = [self._factors[name].upper_bound for name in names]
        centroid = build_candidate_points(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            component_names=names,
            include={"global_centroid"},
            grid=None,
            tolerance=self.tol,
        )[names]
        center_df = centroid.loc[centroid.index.repeat(center_points)].reset_index(drop=True)
        return pd.concat(
            [design_matrix.reset_index(drop=True), center_df],
            ignore_index=True,
        )

    def _validate_design_matrix_bounds(
        self,
        design_matrix: pd.DataFrame,
        factors: dict[str, object],
    ) -> None:
        names = list(factors.keys())
        points = design_matrix[names].to_numpy(dtype=float)
        lower_bounds = [factors[name].lower_bound for name in names]
        upper_bounds = [factors[name].upper_bound for name in names]

        if not np.allclose(points.sum(axis=1), 1.0, atol=self.tol, rtol=0.0):
            raise RuntimeError("Internal error: constrained mixture points do not sum to 1.")
        if (points < np.asarray(lower_bounds, dtype=float) - self.tol).any():
            raise RuntimeError("Internal error: constrained mixture points violate lower bounds.")
        if (points > np.asarray(upper_bounds, dtype=float) + self.tol).any():
            raise RuntimeError("Internal error: constrained mixture points violate upper bounds.")

    def set_model_terms(self, terms=ModelTerms) -> None:
        pro_factors = [
            name for name, factor in self._factors.items()
            if factor.type in ["cont", "cat"]
        ]
        mix_factors = [
            name for name, factor in self._factors.items()
            if factor.type == "mix"
        ]

        self._model_spec = compile_model_spec(terms, pro_factors, mix_factors)

        if self._model_spec.model_terms > self._coded_design_matrix.shape[0]:
            raise ValueError(
                f"The number of model terms ({self._model_spec.model_terms}) "
                f"exceeds the number of experiments ({self._coded_design_matrix.shape[0]}). "
                "Please reduce the model complexity."
            )

        self._model_matrix = self._build_model_matrix(
            self._coded_design_matrix,
            self._model_spec,
        )
