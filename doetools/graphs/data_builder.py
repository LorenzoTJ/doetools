import numpy as np
import pandas as pd
from scipy import stats
from doetools.utils.grid_builder import mixture_grid, rectangular_grid

class DataBuilder:
    
    def barycentric_to_cartesian_3D(self, W):
        
        """
        W: (N, 4) array, rows sum to 1
        returns: (N, 3) Cartesian coordinates
        """
        
        V_3d = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
            [0.5, np.sqrt(3)/6, np.sqrt(6)/3]])
        
        cart_coord = W @ V_3d
        complete_coord = np.column_stack((cart_coord, W[:]))
        
        return complete_coord
    
    def barycentric_to_cartesian_2D(self, W):
        
        """
        W: (N, 3) array, rows sum to 1
        returns: (N, 2) Cartesian coordinates
        """
        
        V_2d = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]])
            
        cart_coord = W @ V_2d
        complete_coord = np.column_stack((cart_coord, W[:]))
        
        return complete_coord
    
    def _create_grid(self,
                    ax1: str,
                    ax2: str,
                    ax3: str | None,
                    constant_levels: dict[str, float],
                    resolution : int = 100,
                    x_min : float = -1.0,
                    x_max : float = 1.0,
                    y_min : float = -1.0,
                    y_max : float = 1.0
                    ) -> pd.DataFrame:
        
        # Extract constant levels (coded)
        coded_constant_levels = self._code_dict(constant_levels)
        design_matrix = self._coded_design_matrix
        coded_constant_levels = self._check_constant_levels(design_matrix, ax1, ax2, ax3, coded_constant_levels)
        
        # Generate the grid of points
        if ax3 is None:
            grid_points = rectangular_grid(
                design_matrix=design_matrix,
                x=ax1,
                y=ax2,
                constants=coded_constant_levels,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                resolution=resolution,
            )
        else:
            mix_factors = [ax1, ax2, ax3]
            grid_points = mixture_grid(
                design_matrix=design_matrix,
                factors=self._factors,
                mix_factors=mix_factors,
                m=resolution,
                constant_levels=coded_constant_levels
            )
        
        return grid_points, coded_constant_levels

    def build_hovertemplate(self, labels, responses, precision=3):
        lines = []
        factors = getattr(self, "_factors", {}) or {}
        for i, lbl in enumerate(labels):
            factor = factors.get(lbl)
            if getattr(factor, "type", None) == "cat":
                lines.append(f"{lbl}: %{{customdata[{i}]}}")
            else:
                lines.append(f"{lbl}: %{{customdata[{i}]:.{precision}f}}")
        for i, response in enumerate(responses):
            lines.append(f"{response}: %{{customdata[{len(labels) + i}]:.{precision}f}}")  
        return "<br>".join(lines) + "<extra></extra>"

    def build_customdata(self, grid_df : pd.DataFrame, responses : pd.DataFrame) -> np.ndarray:
        combined_df = pd.concat([grid_df.reset_index(drop=True), responses.reset_index(drop=True)], axis=1)
        return combined_df.to_numpy()
    
    def mixture_scaler(self, design_matrix : pd.DataFrame, constant_levels : dict[str, float]) -> pd.DataFrame:
        
        mix_factors = [k for k, v in self._factors.items() if v.type == "mix"]
        
        if len(mix_factors) == 3:
            return design_matrix
        elif len(mix_factors) == 4:
            mix_constants = {k : v for k, v in constant_levels.items() if self._factors[k].type == "mix"}
            k = sum(mix_constants.values())
            scaled_matrix = design_matrix.copy()
            cols = design_matrix.columns.difference(constant_levels)
            scaled_matrix[cols] = design_matrix[cols] / (1 - k)
            return scaled_matrix    
        else:
            raise ValueError("Mixture scaler only supports 3 or 4 component mixtures.")
    
    def calculate_confidence_interval(self, grid_of_points : pd.DataFrame, response: str, type_of_correction : str, alpha: float = 0.05) -> np.ndarray:
       
        # Calculate the leverage
        leverages = self._compute_leverage(grid_of_points)
        # Extract mse and d.o.f from the model[response]
        anova = self._mlr_wrapper.results[response].anova
        if type_of_correction == "replicates":
            mse1, dof1 = anova["MS_pe"], anova["df_pe"]
        elif type_of_correction == "residuals":
            mse1, dof1 = anova["MS_res"], anova["df_res"]
        # Calculate the confidence interval
        std_of_pred = np.sqrt(mse1 * leverages)
        conf_int = std_of_pred * stats.t.ppf(1-(alpha/2), dof1)
        return conf_int
    
    def add_confidence_interval(self, predicted_response : pd.Series, response: str, conf_int : np.ndarray) -> pd.Series:
        
        if self._response_conditions[response]["maximize"]:
            corrected_response = predicted_response - conf_int
        else:
            corrected_response = predicted_response + conf_int
            
        return corrected_response
    
    def compute_feasible_region(
        self,
        resp1: str,
        response1: np.ndarray,
        resp2: str = None,
        response2: np.ndarray = None
    ) -> np.ndarray:

        # Condition 1 - feasible region for response 1
        Z1_lb = self._response_conditions[resp1].get("lower_limit", None)
        Z1_ub = self._response_conditions[resp1].get("upper_limit", None)

        cond1 = np.ones_like(response1, dtype=bool)
        cond1 &= np.isfinite(response1)  

        if Z1_lb is not None:
            cond1 &= (response1 >= Z1_lb)
        if Z1_ub is not None:
            cond1 &= (response1 <= Z1_ub)

        # Condition 2 - feasible region for response 2
        cond2 = True
        if resp2 is not None and response2 is not None:
            Z2_lb = self._response_conditions[resp2].get("lower_limit", None)
            Z2_ub = self._response_conditions[resp2].get("upper_limit", None)

            cond2 = np.ones_like(response2, dtype=bool)
            cond2 &= np.isfinite(response2) 

            if Z2_lb is not None:
                cond2 &= (response2 >= Z2_lb)
            if Z2_ub is not None:
                cond2 &= (response2 <= Z2_ub)

        mask = cond1 & cond2

        # Feasible = 1, non-feasible (including NaNs) = NaN
        Z = np.where(mask, 1.0, np.nan)

        return Z
        
    def interpolate_rect_grid_on_symplex(self,
                                         a_title : str,
                                         b_title : str,
                                         c_title : str,
                                         grid_df_scaled : pd.DataFrame,
                                         response_values : np.ndarray,
                                         resolution : int = 250,
                                         A : np.array = np.array([0,0]),
                                         B : np.array = np.array([1,0]),
                                         C : np.array = np.array([0.5, np.sqrt(3)/2])
                                         ) -> np.ndarray:
        
                response = np.asarray(response_values).ravel()
                from scipy.interpolate import griddata

                # Convert barycentric (normalized) to 2D
                grid_scaled_array = grid_df_scaled[[a_title, b_title, c_title]].to_numpy()
                grid_2d = self.barycentric_to_cartesian_2D(grid_scaled_array)
                X = grid_2d[:, 0]
                Y = grid_2d[:, 1]

                # Interpolation grid
                xi = np.linspace(X.min(), X.max(), resolution)
                yi = np.linspace(Y.min(), Y.max(), resolution)
                XI, YI = np.meshgrid(xi, yi)
                ZI = griddata((X, Y), response, (XI, YI), method="linear")

                # Mask outside triangle
                def inside_triangle(X, Y):
                    P = np.stack([X, Y], axis=-1)
                    v0 = C - A
                    v1 = B - A
                    v2 = P - A
                    den = v0[0]*v1[1] - v1[0]*v0[1]
                    a = (v2[...,0]*v1[1] - v1[0]*v2[...,1]) / den
                    b = (v0[0]*v2[...,1] - v2[...,0]*v0[1]) / den
                    c = 1 - a - b
                    return (a >= -1e-12) & (b >= -1e-12) & (c >= -1e-12)

                ZI = np.where(inside_triangle(XI, YI), ZI, np.nan)
                
                return ZI
