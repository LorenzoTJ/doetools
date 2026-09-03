import numpy as np
import pandas as pd
from dataclasses import dataclass
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay, QhullError
from itertools import combinations
from typing import Literal


@dataclass(frozen=True)
class _MixtureSurfaceDomain:
    """Cartesian representation of a ternary mixture plotting domain."""

    mode: Literal["full", "allowed"]
    x: np.ndarray
    y: np.ndarray
    xi: np.ndarray
    yi: np.ndarray
    XI: np.ndarray
    YI: np.ndarray
    mask: np.ndarray
    boundary: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    has_area: bool


@dataclass(frozen=True)
class _ProcessSurfaceDomain:
    """Cartesian grid, mask, and boundary for a process surface."""

    mode: Literal["full", "allowed"]
    X: np.ndarray
    Y: np.ndarray
    mask: np.ndarray
    boundary: np.ndarray
    x_range: tuple[float, float] | None
    y_range: tuple[float, float] | None
    has_area: bool


class _RendererMixin:
    
    # USEFUL COMMON METHODS
    
    def _barycentric_to_cartesian_2d(self, W):
        
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
    
    def _bary_to_xy(self, a, b, c, A, B, C):
        P = a[..., None]*A + b[..., None]*B + c[..., None]*C
        return P[..., 0], P[..., 1]

    def _build_mixture_surface_domain(
        self,
        grid_df_scaled: pd.DataFrame,
        a_title: str,
        b_title: str,
        c_title: str,
        resolution: int = 250,
        mode: Literal["full", "allowed"] = "full",
    ) -> _MixtureSurfaceDomain:
        """Build one shared Cartesian grid and domain mask for mixture plots."""
        if mode not in {"full", "allowed"}:
            raise ValueError("domain must be either 'full' or 'allowed'.")
        if resolution < 2:
            raise ValueError("resolution must be at least 2.")

        mixture = grid_df_scaled[[a_title, b_title, c_title]].to_numpy(dtype=float)
        xy = self._barycentric_to_cartesian_2d(mixture)[:, :2]
        finite_points = xy[np.isfinite(xy).all(axis=1)]
        if len(finite_points) == 0:
            raise ValueError("The mixture plotting grid contains no finite points.")

        x = xy[:, 0]
        y = xy[:, 1]
        x_bounds = self._padded_range(finite_points[:, 0], padding=0.08)
        y_bounds = self._padded_range(finite_points[:, 1], padding=0.08)

        xi = np.linspace(finite_points[:, 0].min(), finite_points[:, 0].max(), resolution)
        yi = np.linspace(finite_points[:, 1].min(), finite_points[:, 1].max(), resolution)
        if np.isclose(xi[0], xi[-1]):
            xi = np.linspace(*x_bounds, resolution)
        if np.isclose(yi[0], yi[-1]):
            yi = np.linspace(*y_bounds, resolution)
        XI, YI = np.meshgrid(xi, yi)

        simplex_mask = self._inside_triangle(XI, YI)

        unique_points = np.unique(finite_points, axis=0)
        hull = None
        if len(unique_points) >= 3 and np.linalg.matrix_rank(
            unique_points - unique_points[0]
        ) >= 2:
            try:
                hull = ConvexHull(unique_points)
            except QhullError:
                hull = None

        if hull is not None:
            hull_points = unique_points[hull.vertices]
            experimental_boundary = np.vstack([hull_points, hull_points[0]])
            grid_points = np.column_stack([XI.ravel(), YI.ravel()])
            tolerance = 1e-10
            hull_mask = np.all(
                grid_points @ hull.equations[:, :-1].T
                + hull.equations[:, -1]
                <= tolerance,
                axis=1,
            ).reshape(XI.shape)
        else:
            experimental_boundary = self._degenerate_boundary(unique_points)
            hull_mask = np.zeros_like(XI, dtype=bool)

        if mode == "allowed":
            mask = hull_mask
            has_area = hull is not None
        else:
            mask = simplex_mask
            has_area = True

        return _MixtureSurfaceDomain(
            mode=mode,
            x=x,
            y=y,
            xi=xi,
            yi=yi,
            XI=XI,
            YI=YI,
            mask=mask,
            boundary=experimental_boundary,
            x_range=x_bounds,
            y_range=y_bounds,
            has_area=has_area,
        )

    @staticmethod
    def _interpolate_mixture_surface(
        domain: _MixtureSurfaceDomain,
        response_values: np.ndarray,
    ) -> np.ndarray:
        """Interpolate response values on a mixture domain and apply its mask."""
        response = np.asarray(response_values, dtype=float).ravel()
        if response.size != domain.x.size:
            raise ValueError(
                "response_values must contain one value for each mixture grid point."
            )

        valid = np.isfinite(domain.x) & np.isfinite(domain.y) & np.isfinite(response)
        if not domain.has_area or valid.sum() < 3:
            return np.full(domain.XI.shape, np.nan)

        points = np.column_stack([domain.x[valid], domain.y[valid]])
        try:
            interpolated = griddata(
                points,
                response[valid],
                (domain.XI, domain.YI),
                method="linear",
            )
        except (QhullError, ValueError):
            try:
                interpolated = griddata(
                    points,
                    response[valid],
                    (domain.XI, domain.YI),
                    method="nearest",
                )
            except (QhullError, ValueError):
                return np.full(domain.XI.shape, np.nan)

        return np.where(domain.mask, interpolated, np.nan)

    def _build_process_surface_domain(
        self,
        grid: pd.DataFrame,
        x_title: str,
        y_title: str,
        mode: Literal["full", "allowed"] = "full",
        filters: list | None = None,
        filter_grid: pd.DataFrame | None = None,
        fallback_points: pd.DataFrame | None = None,
    ) -> _ProcessSurfaceDomain:
        """Build a process-grid mask and its continuous convex-hull boundary."""
        if mode not in {"full", "allowed"}:
            raise ValueError("domain must be either 'full' or 'allowed'.")

        resolution_x = grid[x_title].nunique()
        resolution_y = grid[y_title].nunique()
        if resolution_x * resolution_y != len(grid):
            raise ValueError("The process plotting grid must be rectangular.")

        shape = (resolution_y, resolution_x)
        X = grid[x_title].to_numpy(dtype=float).reshape(shape)
        Y = grid[y_title].to_numpy(dtype=float).reshape(shape)
        full_mask = np.ones(shape, dtype=bool)
        if mode == "full":
            return _ProcessSurfaceDomain(
                mode=mode,
                X=X,
                Y=Y,
                mask=full_mask,
                boundary=np.empty((0, 2), dtype=float),
                x_range=None,
                y_range=None,
                has_area=True,
            )

        domain_filters = list(filters or [])
        if domain_filters:
            evaluation_grid = filter_grid if filter_grid is not None else grid
            if len(evaluation_grid) != len(grid):
                raise ValueError("filter_grid must contain one row per plotting point.")
            mask_flat = np.ones(len(grid), dtype=bool)
            for domain_filter in domain_filters:
                result = domain_filter(evaluation_grid)
                if np.isscalar(result):
                    filter_mask = np.full(len(grid), bool(result), dtype=bool)
                else:
                    if len(result) != len(grid):
                        raise ValueError(
                            "Each domain filter must return one value per plotting point."
                        )
                    if isinstance(result, pd.Series):
                        filter_mask = result.reindex(evaluation_grid.index)
                    else:
                        filter_mask = pd.Series(
                            result, index=evaluation_grid.index
                        )
                    filter_mask = filter_mask.fillna(False).to_numpy(dtype=bool)
                    if filter_mask.ndim != 1:
                        raise ValueError(
                            "Each domain filter must return a one-dimensional mask."
                        )
                mask_flat &= filter_mask
            mask = mask_flat.reshape(shape)
            boundary_source = np.column_stack([X[mask], Y[mask]])
        else:
            mask = full_mask.copy()
            if (
                fallback_points is not None
                and x_title in fallback_points
                and y_title in fallback_points
            ):
                boundary_source = fallback_points[
                    [x_title, y_title]
                ].to_numpy(dtype=float)
            else:
                boundary_source = np.empty((0, 2), dtype=float)

        finite = boundary_source[np.isfinite(boundary_source).all(axis=1)]
        unique_points = np.unique(finite, axis=0)
        hull = None
        if len(unique_points) >= 3 and np.linalg.matrix_rank(
            unique_points - unique_points[0]
        ) >= 2:
            try:
                hull = ConvexHull(unique_points)
            except QhullError:
                hull = None

        if hull is not None:
            hull_points = unique_points[hull.vertices]
            boundary = np.vstack([hull_points, hull_points[0]])
            if not domain_filters:
                grid_points = np.column_stack([X.ravel(), Y.ravel()])
                mask = np.all(
                    grid_points @ hull.equations[:, :-1].T
                    + hull.equations[:, -1]
                    <= 1e-10,
                    axis=1,
                ).reshape(shape)
        else:
            boundary = self._degenerate_boundary(unique_points)
            if not domain_filters:
                mask = full_mask

        if len(boundary) >= 2:
            x_range = self._exact_range(boundary[:, 0])
            y_range = self._exact_range(boundary[:, 1])
        elif mask.any():
            x_range = self._exact_range(X[mask])
            y_range = self._exact_range(Y[mask])
        else:
            x_range = self._exact_range(X)
            y_range = self._exact_range(Y)

        return _ProcessSurfaceDomain(
            mode=mode,
            X=X,
            Y=Y,
            mask=mask,
            boundary=boundary,
            x_range=x_range,
            y_range=y_range,
            has_area=hull is not None,
        )

    @staticmethod
    def _coerce_process_surface_values(
        values: np.ndarray,
        domain: _ProcessSurfaceDomain,
    ) -> np.ndarray:
        """Reshape process values and hide points outside the domain."""
        array = np.asarray(values, dtype=float)
        if array.shape != domain.X.shape:
            if array.size != domain.X.size:
                raise ValueError(
                    "Process surface values must contain one value per grid point."
                )
            array = array.reshape(domain.X.shape)
        return np.where(domain.mask, array, np.nan)

    def _coerce_mixture_surface_values(
        self,
        values: np.ndarray,
        domain: _MixtureSurfaceDomain,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape == domain.XI.shape:
            return np.where(domain.mask, array, np.nan)
        if array.size == domain.x.size:
            return self._interpolate_mixture_surface(domain, array)
        raise ValueError(
            "Mixture contour values must match either the source points or "
            "the interpolation grid."
        )

    @staticmethod
    def _inside_triangle(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        height = np.sqrt(3) / 2.0
        return (
            (Y >= -1e-12)
            & (Y <= height + 1e-12)
            & (Y <= np.sqrt(3) * X + 1e-12)
            & (Y <= np.sqrt(3) * (1.0 - X) + 1e-12)
        )

    @staticmethod
    def _padded_range(
        values: np.ndarray,
        padding: float = 0.08,
    ) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (-0.5, 0.5)
        lower = float(finite.min())
        upper = float(finite.max())
        span = upper - lower
        if np.isclose(span, 0.0):
            span = max(abs(lower), 1.0) * 0.1
        pad = span * padding
        return lower - pad, upper + pad

    @staticmethod
    def _exact_range(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (-0.5, 0.5)
        lower = float(finite.min())
        upper = float(finite.max())
        if np.isclose(lower, upper):
            span = max(abs(lower), 1.0) * 0.1
            return lower - span, upper + span
        return lower, upper

    @staticmethod
    def _degenerate_boundary(points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return points.copy()
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        first, second = np.unravel_index(np.argmax(distances), distances.shape)
        return points[[first, second]]

    @staticmethod
    def _finite_range(
        *values: np.ndarray,
        padding: float = 0.08,
    ) -> tuple[float, float]:
        arrays = [np.asarray(value, dtype=float).ravel() for value in values if value is not None]
        finite = np.concatenate(arrays) if arrays else np.array([], dtype=float)
        finite = finite[np.isfinite(finite)]
        return _RendererMixin._padded_range(finite, padding=padding)

    @staticmethod
    def _finite_extrema(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return np.nan, np.nan
        return float(finite.min()), float(finite.max())

    def _add_vertex_annotations(self, fig, A, B, C, texts, font_size=12):
        
        pts = [(A, texts[0], -25, 0), (B, texts[1], 25, 0), (C, texts[2], 0, 15)]
        for (P, t, xshift, yshift) in pts:
            fig.add_annotation(
                x=P[0], y=P[1],
                text=t,
                showarrow=False,
                font=dict(size=font_size, family="Arial"),
                xshift= xshift,
                yshift=yshift
            )
    
    def _add_axis_titles(self, fig, A, B, C, a_title, b_title, c_title, font_size=13):
        
        mid_BC = 0.5*(B + C)
        mid_AC = 0.5*(A + C)
        mid_AB = 0.5*(A + B)

        fig.add_annotation(x=mid_BC[0] + 0.06, y=mid_BC[1] + 0.04, text=f"<b>{a_title}</b>", showarrow=False,
                        font=dict(size=font_size, family="Arial"))
        fig.add_annotation(x=mid_AC[0] - 0.06, y=mid_AC[1] + 0.04, text=f"<b>{b_title}</b>", showarrow=False,
                        font=dict(size=font_size, family="Arial"))
        fig.add_annotation(x=mid_AB[0], y=mid_AB[1] - 0.06, text=f"<b>{c_title}</b>", showarrow=False,
                        font=dict(size=font_size, family="Arial"))
        
    def _add_axis_titles_3d(
            self,
            fig,
            A, B, C,
            a_title, b_title, c_title,
            z0: float = 0.0,
            font_size: int = 13,
            color: str = "black"
        ) -> None:

            A = np.asarray(A, dtype=float)
            B = np.asarray(B, dtype=float)
            C = np.asarray(C, dtype=float)

            mid_BC = 0.5 * (B + C)
            mid_AC = 0.5 * (A + C)
            mid_AB = 0.5 * (A + B)

            # Same offsets as your 2D version (tweak if needed)
            pts = np.array([
                [mid_BC[0] + 0.1, mid_BC[1] + 0.08, z0],
                [mid_AC[0] - 0.1, mid_AC[1] + 0.08, z0],
                [mid_AB[0],        mid_AB[1] - 0.1, z0],
            ])

            fig.add_trace(go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="text",
                text=[f"<b>{a_title}</b>", f"<b>{b_title}</b>", f"<b>{c_title}</b>"],
                textfont=dict(size=font_size, family="Arial", color=color),
                showlegend=False,
                hoverinfo="skip"
            ))
    
    # COUNTOUR AND SURFACE PLOTTING METHODS
    
    def _render_contour_process(self,
                               grid : pd.DataFrame,
                               x_title : str,
                               y_title : str,
                               z_title : str,
                               response : np.ndarray,
                               constant_levels : list = None,
                               hovertemplate : str = None,
                               customdata : np.ndarray = None,
                               z2_title : str = None,
                               second_response : np.ndarray = None,
                               feasible_region : np.ndarray = None,
                               process_surface_domain: _ProcessSurfaceDomain = None,
                               ) -> go.Figure:
        
        # 1) Create the plotly figure
        fig = go.Figure()
        
        # 2) Prepare data for contour
        if process_surface_domain is None:
            X = grid[x_title].values
            Y = grid[y_title].values
            response_plot = response
            second_response_plot = second_response
            customdata_plot = customdata
        else:
            X = process_surface_domain.X[0]
            Y = process_surface_domain.Y[:, 0]
            response_plot = self._coerce_process_surface_values(
                response, process_surface_domain
            )
            second_response_plot = (
                self._coerce_process_surface_values(
                    second_response, process_surface_domain
                )
                if second_response is not None
                else None
            )
            if customdata is not None and customdata.ndim > 1:
                customdata_plot = customdata.reshape(
                    (*process_surface_domain.X.shape, customdata.shape[1])
                )
            elif customdata is not None:
                customdata_plot = customdata.reshape(process_surface_domain.X.shape)
            else:
                customdata_plot = None
        
        # 3) Create the contour trace
        cp_trace = go.Contour(
            z=response_plot,
            x=X,
            y=Y,
            name = z_title,
            showscale = False,
            colorscale=None,
            line=dict(color= "#003153", width = 2),
            contours=dict(
                coloring="none",
                showlabels=True,
                labelfont=dict(size=12, family="Arial", color="#1B9E9E"),
            ),
            showlegend=True,
            customdata=customdata_plot,
            hovertemplate= hovertemplate)
        
        fig.add_trace(cp_trace)
        
        # 3.b) Add second contour if provided
        if second_response_plot is not None:
            
            cp2_trace = go.Contour(
                z=second_response_plot,
                x=X,
                y=Y,
                name = z2_title,
                showscale = False,
                colorscale=None,
                line=dict(color= "#FF5733", width = 1, dash="solid"),
                contours=dict(
                    coloring="none",
                    showlabels=True,
                    labelfont=dict(size=12, family="Arial", color="#FF5733"),
                ),
                showlegend=True,
                customdata=customdata_plot,
                hovertemplate= hovertemplate)
            
            fig.add_trace(cp2_trace)
        
        # 3.c) Add feasible region if provided
        if feasible_region is not None:
            if process_surface_domain is not None:
                feasible_region = self._coerce_process_surface_values(
                    feasible_region, process_surface_domain
                )
                Xr = process_surface_domain.X
                Yr = process_surface_domain.Y
            else:
                Xr = X.reshape(feasible_region.shape)
                Yr = Y.reshape(feasible_region.shape)
                        
            fig.add_trace(go.Contour(
                x=Xr[0], y=Yr[:, 0], z=feasible_region,
                contours=dict(coloring="heatmap", showlines=False),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(50,205,50,0.25)"]],
                zmin=0, zmax=1,
                showscale=False,
                hoverinfo="skip",
                name="feasible region"
                ))

        if (
            process_surface_domain is not None
            and len(process_surface_domain.boundary) >= 2
        ):
            fig.add_trace(
                go.Scatter(
                    x=process_surface_domain.boundary[:, 0],
                    y=process_surface_domain.boundary[:, 1],
                    mode="lines",
                    line=dict(color="black", width=3),
                    name="Allowed domain",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        
        # Add Title Annotation
        fig.add_annotation(
            text=f"<b>Contour Plot - {z_title}</b>",
            xref="paper", yref="paper",
            x=0.5, y=1.09,
            showarrow=False,
            font=dict(size=20, family="Arial", color="black")
        )
        
        # Add constant levels annotation
        if constant_levels is not None and len(constant_levels) > 0:
            const_text = "<b>Constant Levels</b><br>" + "<br>".join([f"{k}: {v}" for k, v in constant_levels.items()])
        else:
            const_text = ""
        z_min, z_max = self._finite_extrema(response_plot)
        fig.add_annotation(
            text=f"{const_text}<br>Max. {z_title}: {z_max:.3f}<br>Min. {z_title}: {z_min:.3f}",
            xref="paper", yref="paper",
            x=0.8, y=0.96,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, family="Arial", color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=8
        )
               
        process_x_range = (
            process_surface_domain.x_range
            if process_surface_domain is not None
            else None
        )
        process_y_range = (
            process_surface_domain.y_range
            if process_surface_domain is not None
            else None
        )

        # 4) Add External contour
        fig.update_xaxes(
            title_text=f"<b>{x_title}</b>",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            domain = [0.25, 0.75],
            constrain="domain",
            range=process_x_range,
          )
        fig.update_yaxes(
            title_text=f"<b>{y_title}</b>",
            showline=True,
            linewidth=2,
            linecolor="black",
            domain = [0.05, 0.95],
            mirror=True,
            range=process_y_range,
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=40, r=20, t=60, b=50),
            legend=dict(
                x=0.80,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            ),
            meta=dict(
                domain=(
                    process_surface_domain.mode
                    if process_surface_domain is not None
                    else "full"
                )
            ),
            width=800,
            height=500)
        
        return fig
    
    def _render_surface_process(self,
                               grid : pd.DataFrame,
                               x_title : str,
                               y_title : str,
                               z_title : str,
                               response : np.ndarray,
                               constant_levels : list = None,
                               hovertemplate : str = None,
                               customdata : np.ndarray = None,
                               feasible_region : bool = False,
                               z2_title : str = None,
                               second_response : np.ndarray = None,
                               process_surface_domain: _ProcessSurfaceDomain = None,
                               ) -> go.Figure:
        
        # 1) Create the plotly figure
        fig = go.Figure()
        # 2) Prepare data for 3D surface
        if process_surface_domain is None:
            resolution_x = len(grid[x_title].unique())
            resolution_y = len(grid[y_title].unique())
            surface_shape = (resolution_x, resolution_y)
            X = grid[x_title].values.reshape(surface_shape)
            Y = grid[y_title].values.reshape(surface_shape)
            Z = np.asarray(response, dtype=float).reshape(surface_shape)
        else:
            surface_shape = process_surface_domain.X.shape
            X = process_surface_domain.X
            Y = process_surface_domain.Y
            Z = self._coerce_process_surface_values(
                response, process_surface_domain
            )
        # Reshape customdata preserving all columns: (N, cols) -> (res_x, res_y, cols)
        if customdata is not None and customdata.ndim > 1:
            n_cols = customdata.shape[1]
            customdata_r = customdata.reshape((*surface_shape, n_cols))
        elif customdata is not None:
            customdata_r = customdata.reshape(surface_shape)
        else:
            customdata_r = None
        
        # 3) Create the 3D surface trace
        z1_l, z1_u = self._finite_extrema(Z)
        show_z1_contours = bool(
            np.isfinite(z1_l) and np.isfinite(z1_u) and z1_u > z1_l
        )
        contour_size = (
            (z1_u - z1_l) / 30.0
            if show_z1_contours
            else 1.0
        )
        surface_trace = go.Surface(
            z=Z,
            x=X,
            y=Y,
            name = z_title,
            showscale = False,
            showlegend=True,
            colorscale="Teal",
            customdata=customdata_r,
            hovertemplate= hovertemplate,
                contours=dict(
                    z=dict(
                        show=show_z1_contours,
                        color="lightgrey",
                        start=z1_l if show_z1_contours else 0.0,
                        end=z1_u if show_z1_contours else 1.0,
                        size=contour_size,
                        width=1,
                    )
        ))
        fig.add_trace(surface_trace)
        
        # 4) Add second surface if provided
        Z2 = None
        if second_response is not None:
            if process_surface_domain is None:
                Z2 = np.asarray(second_response, dtype=float).reshape(surface_shape)
            else:
                Z2 = self._coerce_process_surface_values(
                    second_response, process_surface_domain
                )
            z2_l, z2_u = self._finite_extrema(Z2)
            show_z2_contours = bool(
                np.isfinite(z2_l) and np.isfinite(z2_u) and z2_u > z2_l
            )
            contour2_size = (
                (z2_u - z2_l) / 30.0
                if show_z2_contours
                else 1.0
            )
            surface2_trace = go.Surface(
                z=Z2,
                x=X,
                y=Y,
                name = z2_title,
                showscale = False,
                showlegend=True,
                colorscale="Hot",
                customdata=customdata_r,
                hovertemplate= hovertemplate,
                contours=dict(
                        z=dict(
                            show=show_z2_contours,
                            color="lightgrey",
                            start=z2_l if show_z2_contours else 0.0,
                            end=z2_u if show_z2_contours else 1.0,
                            size=contour2_size,
                            width=1,
                        )
            ))
            fig.add_trace(surface2_trace)
        
        if feasible_region:
            
            Z_ub = self._response_conditions[z_title].get("upper_limit", None)
            Z_lb = self._response_conditions[z_title].get("lower_limit", None)
            
            if Z_ub is not None:
                upper_plane = np.full_like(Z, Z_ub)
                if process_surface_domain is not None:
                    upper_plane = np.where(
                        process_surface_domain.mask, upper_plane, np.nan
                    )
                fig.add_trace(go.Surface(
                    z=upper_plane,
                    x=X,
                    y=Y,
                    name="Upper Bound",
                    showscale=False,
                    colorscale=[[0, "rgba(255,0,0,0.0)"], [1, "rgba(255,0,0,0.5)"]],
                    hoverinfo="skip",
                    showlegend=True
                ))
            if Z_lb is not None:
                lower_plane = np.full_like(Z, Z_lb)
                if process_surface_domain is not None:
                    lower_plane = np.where(
                        process_surface_domain.mask, lower_plane, np.nan
                    )
                fig.add_trace(go.Surface(
                    z=lower_plane,
                    x=X,
                    y=Y,
                    name="Lower Bound",
                    showscale=False,
                    colorscale=[[0, "rgba(0,0,255,0.0)"], [1, "rgba(0,0,255,0.5)"]],
                    hoverinfo="skip",
                    showlegend=True
                ))

        if (
            process_surface_domain is not None
            and len(process_surface_domain.boundary) >= 2
        ):
            boundary_z = self._finite_range(Z, Z2, padding=0.0)[0]
            fig.add_trace(
                go.Scatter3d(
                    x=process_surface_domain.boundary[:, 0],
                    y=process_surface_domain.boundary[:, 1],
                    z=np.full(len(process_surface_domain.boundary), boundary_z),
                    mode="lines",
                    line=dict(color="black", width=6),
                    name="Allowed domain",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        
        # Add Title Annotation
        fig.add_annotation(
            text=f"<b>3D Surface Plot - {z_title}</b>",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=20, family="Arial", color="black")
        )
        
        # Add constant levels annotation
        if constant_levels is not None and len(constant_levels) > 0:
            const_text = "<b>Constant Levels</b><br>" + "<br>".join([f"{k}: {v}" for k, v in constant_levels.items()])
        else:
            const_text = ""
            
        fig.add_annotation(
            text=f"{const_text}<br>Max. {z_title}: {z1_u:.3f}<br>Min. {z_title}: {z1_l:.3f}",
            xref="paper", yref="paper",
            x=0.80, y=0.96,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, family="Arial", color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=8
        )
               
        process_x_range = (
            process_surface_domain.x_range
            if process_surface_domain is not None
            else None
        )
        process_y_range = (
            process_surface_domain.y_range
            if process_surface_domain is not None
            else None
        )

        # Update Layout
        fig.update_layout(
          scene=dict(
              xaxis=dict(
                  title=x_title,
                  showbackground=False,
                  showgrid=True,
                  gridcolor="lightgrey",
                  zeroline=False,
                  range=process_x_range,
              ),
              yaxis=dict(
                  title=y_title,
                  showbackground=False,
                  showgrid=True,
                  gridcolor="lightgrey",
                  zeroline=False,
                  range=process_y_range,
              ),
              zaxis=dict(
                  title=z_title,
                  showbackground=False,
                  showgrid=True,
                  gridcolor="lightgrey",
                  zeroline=False,
              ),
              aspectmode="cube",
          ),
          paper_bgcolor="white",
          plot_bgcolor="white",
          margin=dict(l=40, r=40, t=50, b=50),
          legend=dict(
                x=0.80,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            ),
            meta=dict(
                domain=(
                    process_surface_domain.mode
                    if process_surface_domain is not None
                    else "full"
                )
            ),
            width=800,
            height=500
            )
        
        return fig   
    
    def _add_ternary_grid_2d(
        self,
        fig,
        A, B, C,
        min: float = 0.0,
        max: float = 1.0,
        step: float = 0.1,
        width: float = 1,
        color: str = "rgba(0,0,0,0.15)",
        label_font_size: int = 9,
        label_color: str = "rgba(0,0,0,0.55)",
    ) -> None:

        import numpy as np
        import plotly.graph_objects as go

        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)

        ts = np.linspace(0.0, 1.0, 80)
        vals = np.arange(step, 1.0, step)
        pct = np.round(np.linspace(min, max, 11), 3)[1:-1]

        # Collect labels (one trace at the end)
        lx, ly, lt = [], [], []

        # ---- a = v lines ----
        for v, p in zip(vals, pct):
            
            point = int(round(p * 100))

            b = (1 - v) * ts
            c = (1 - v) * (1 - ts)
            a = np.full_like(ts, v)

            x, y = self._bary_to_xy(a, b, c, A, B, C)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(width=width, color=color),
                showlegend=False,
                hoverinfo="skip"
            ))

            # label point (one side)
            i = np.argmax(y)  # bottom-most point on that segment
            lx.append(x[i] - 0.015)
            ly.append(y[i] + 0.015)
            lt.append(str(point))

        # ---- b = v lines ----
        for v, p in zip(vals, pct):
            
            point = int(round(p * 100))
            
            a = (1 - v) * ts
            c = (1 - v) * (1 - ts)
            b = np.full_like(ts, v)

            x, y = self._bary_to_xy(a, b, c, A, B, C)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(width=width, color=color),
                showlegend=False,
                hoverinfo="skip"
            ))

            i = np.argmin(y)
            lx.append(x[i])
            ly.append(y[i] - 0.019)
            lt.append(str(point))

        # ---- c = v lines ----
        for v, p in zip(vals, pct):
            
            point = int(round(p * 100))

            a = (1 - v) * ts
            b = (1 - v) * (1 - ts)
            c = np.full_like(ts, v)

            x, y = self._bary_to_xy(a, b, c, A, B, C)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(width=width, color=color),
                showlegend=False,
                hoverinfo="skip"
            ))

            i = np.argmin(y)
            lx.append(x[i] + 0.015)
            ly.append(y[i] + 0.015)
            lt.append(str(point))

        # Add all labels once
        fig.add_trace(go.Scatter(
            x=lx, y=ly,
            mode="text",
            text=lt,
            textfont=dict(size=label_font_size, color=label_color, family="Arial"),
            showlegend=False,
            hoverinfo="skip"
        ))
        
    def _add_ternary_grid_3d(
        self,
        fig,
        A, B, C,
        min: float = 0.0,
        max: float = 1.0,
        step: float = 0.1,
        width: float = 1,
        color: str = "rgba(0,0,0,0.15)",
        label_font_size: int = 9,
        label_color: str = "rgba(0,0,0,0.55)",
        z0: float = 0.0,
    ) -> None:

        import numpy as np
        import plotly.graph_objects as go

        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)

        ts = np.linspace(0.0, 1.0, 80)
        vals = np.arange(step, 1.0, step)
        pct = np.round(np.linspace(min, max, 11), 3)[1:-1]

        # Collect labels (single trace)
        lx, ly, lz, lt = [], [], [], []

        # ---- a = v lines ----
        for v, p in zip(vals, pct):
            point = int(round(p * 100))

            b = (1 - v) * ts
            c = (1 - v) * (1 - ts)
            a = np.full_like(ts, v)

            x, y = self._bary_to_xy(a, b, c, A, B, C)
            z = np.full_like(x, z0)

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=width, color=color),
                showlegend=False,
                hoverinfo="skip"
            ))

            i = 0
            lx.append(x[i] - 0.015)
            ly.append(y[i] + 0.015)
            lz.append(z0)
            lt.append(str(point))

        # ---- b = v lines ----
        for v, p in zip(vals, pct):
            point = int(round(p * 100))

            a = (1 - v) * ts
            c = (1 - v) * (1 - ts)
            b = np.full_like(ts, v)

            x, y = self._bary_to_xy(a, b, c, A, B, C)
            z = np.full_like(x, z0)

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=width, color=color),
                showlegend=False,
                hoverinfo="skip"
            ))

            i = -1
            lx.append(x[i])
            ly.append(y[i] - 0.019)
            lz.append(z0)
            lt.append(str(point))

        # ---- c = v lines ----
        for v, p in zip(vals, pct):
            point = int(round(p * 100))

            a = (1 - v) * ts
            b = (1 - v) * (1 - ts)
            c = np.full_like(ts, v)

            x, y = self._bary_to_xy(a, b, c, A, B, C)
            z = np.full_like(x, z0)

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=width, color=color),
                showlegend=False,
                hoverinfo="skip"
            ))

            i = 0
            lx.append(x[i] + 0.015)
            ly.append(y[i] + 0.015)
            lz.append(z0)
            lt.append(str(point))

        # ---- labels (single text trace) ----
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="text",
            text=lt,
            textfont=dict(size=label_font_size, color=label_color, family="Arial"),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    def _render_contour_mixture(
        self,
        grid_df_scaled : pd.DataFrame,
        grid_df: pd.DataFrame,
        response : np.ndarray,
        a_title: str,
        b_title: str,
        c_title: str,
        z_title: str,
        constant_levels: dict = None,
        customdata: np.ndarray = None,
        hovertemplate=None,
        grid_step=0.1,
        vertex_texts=None,
        show_grid=True,
        show_border=True,
        min : float = 0.0,
        max : float = 1.0,
        feasible_region: np.ndarray = None,
        second_response: np.ndarray = None,
        z2_title: str = None,
        resolution: int = 250,
        domain: Literal["full", "allowed"] = "full",
        surface_domain: _MixtureSurfaceDomain = None,
        ) -> go.Figure:
        if domain not in {"full", "allowed"}:
            raise ValueError("domain must be either 'full' or 'allowed'.")

        rt3 = np.sqrt(3)
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 0.0])
        C = np.array([0.5, rt3/2.0])

        domain = surface_domain or self._build_mixture_surface_domain(
            grid_df_scaled=grid_df_scaled,
            a_title=a_title,
            b_title=b_title,
            c_title=c_title,
            resolution=resolution,
            mode=domain,
        )
        response_grid = self._coerce_mixture_surface_values(response, domain)
        second_response_grid = (
            self._coerce_mixture_surface_values(second_response, domain)
            if second_response is not None
            else None
        )
        feasible_grid = (
            self._coerce_mixture_surface_values(feasible_region, domain)
            if feasible_region is not None
            else None
        )

        fig = go.Figure()

        if domain.mode == "full" and show_grid:
            self._add_ternary_grid_2d(fig, A, B, C, step=grid_step, min=min, max=max)

        fig.add_trace(go.Contour(
            z=response_grid, x=domain.xi, y=domain.yi,
            showscale=False,
            ncontours=10,
            line=dict(color="#003153", width=1),
            contours=dict(coloring="none", showlabels=True,
                        labelfont=dict(size=12, family="Arial", color="#1B9E9E")),
            showlegend=True,
            name = z_title,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=domain.x, y=domain.y,
            mode="markers",
            marker=dict(size=8, opacity=0.0),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False
        ))
        
        if second_response_grid is not None:
            fig.add_trace(go.Contour(
                z=second_response_grid, x=domain.xi, y=domain.yi,
                showscale=False,
                ncontours=10,
                line=dict(color="#FF5733", width=1, dash="solid"),
                contours=dict(coloring="none", showlabels=True,
                            labelfont=dict(size=12, family="Arial", color="#FF5733")),
                showlegend=True,
                name = z2_title,
                hoverinfo="skip"
            ))
            
        if feasible_grid is not None:
            fig.add_trace(go.Contour(
                x=domain.xi, y=domain.yi, z=feasible_grid,
                contours=dict(coloring="heatmap", showlines=False),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(50,205,50,0.25)"]],
                zmin=0, zmax=1,
                showscale=False,
                hoverinfo="skip",
                name="feasible region"
                ))

        if show_border and domain.mode == "full":
            fig.add_trace(go.Scatter(
                x=[A[0], B[0], C[0], A[0]],
                y=[A[1], B[1], C[1], A[1]],
                mode="lines",
                line=dict(width=2, color="black"),
                showlegend=False,
                hoverinfo="skip"
            ))

        if domain.boundary.size:
            fig.add_trace(go.Scatter(
                x=domain.boundary[:, 0],
                y=domain.boundary[:, 1],
                mode="lines",
                line=dict(
                    width=2 if domain.mode == "allowed" else 1,
                    color="black",
                    dash="solid" if domain.mode == "allowed" else "dash",
                ),
                name="Allowed domain",
                showlegend=False,
                hoverinfo="skip",
            ))

        if domain.mode == "full":
            self._add_axis_titles(fig, A, B, C, a_title, b_title, c_title)
            if vertex_texts is not None:
                self._add_vertex_annotations(fig, A, B, C, vertex_texts)
            
        fig.add_annotation(
            text=f"<b>Contour Plot - {z_title}</b>",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=20, family="Arial", color="black")
        )
        
        if constant_levels is not None and len(constant_levels) > 0:
            const_text = "<b>Constant Levels</b><br>" + "<br>".join([f"{k}: {v}" for k, v in constant_levels.items()])
        else:
            const_text = ""
        response_min, response_max = self._finite_extrema(response)
        fig.add_annotation(
            text=f"{const_text}<br>Max. {z_title}: {response_max:.3f}<br>Min. {z_title}: {response_min:.3f}",
            xref="paper", yref="paper",
            x=0.8, y=0.96,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, family="Arial", color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=8
        )

        xaxis = dict(visible=False)
        yaxis = dict(visible=False, scaleanchor="x", scaleratio=1)
        if domain.mode == "allowed":
            xaxis["range"] = list(domain.x_range)
            yaxis["range"] = list(domain.y_range)

        fig.update_layout(
            showlegend=True,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=xaxis,
            yaxis=yaxis,
            margin=dict(l=40, r=20, t=60, b=40),
            meta={"domain": domain.mode},
        )
        
        fig.update_layout(
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#003153",
                font=dict(size=12, family="Arial", color="black")
            ),
            legend=dict(
                x=0.80,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            ),
            width=800,
            height=500
        )
        return fig
            
    def _render_surface_mixture(
        self,
        grid_df_scaled: pd.DataFrame,
        grid_df: pd.DataFrame,
        response: np.ndarray,
        a_title: str,
        b_title: str,
        c_title: str,
        z_title: str,
        constant_levels: dict = None,
        hovertemplate: str = None,
        customdata: np.ndarray = None,
        min : float = 0.0,
        max : float = 1.0,
        second_response: np.ndarray = None,
        z2_title: str = None,
        feasible_region: bool = False,
        show_border: bool = True,
        center_xy: bool = True,
        domain: Literal["full", "allowed"] = "full",
        surface_domain: _MixtureSurfaceDomain = None,
    ):
        if domain not in {"full", "allowed"}:
            raise ValueError("domain must be either 'full' or 'allowed'.")

        rt3 = np.sqrt(3)
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 0.0])
        C = np.array([0.5, rt3 / 2.0])

        domain = surface_domain or self._build_mixture_surface_domain(
            grid_df_scaled=grid_df_scaled,
            a_title=a_title,
            b_title=b_title,
            c_title=c_title,
            mode=domain,
        )
        Z1 = np.asarray(response, dtype=float).ravel()
        Z2 = (
            np.asarray(second_response, dtype=float).ravel()
            if second_response is not None
            else None
        )
        if Z1.size != domain.x.size:
            raise ValueError("response must contain one value for each mixture grid point.")
        if Z2 is not None and Z2.size != domain.x.size:
            raise ValueError(
                "second_response must contain one value for each mixture grid point."
            )

        X = domain.x.copy()
        Y = domain.y.copy()
        if center_xy:
            centroid = (A + B + C) / 3.0
            X = X - centroid[0]
            Y = Y - centroid[1]
            A2, B2, C2 = A - centroid, B - centroid, C - centroid
        else:
            A2, B2, C2 = A, B, C

        fig = go.Figure()
        finite_xy = np.isfinite(X) & np.isfinite(Y)
        _, unique_indices = np.unique(
            np.column_stack([X[finite_xy], Y[finite_xy]]),
            axis=0,
            return_index=True,
        )
        source_indices = np.flatnonzero(finite_xy)[np.sort(unique_indices)]
        mesh_x = X[source_indices]
        mesh_y = Y[source_indices]
        mesh_z1 = Z1[source_indices]
        mesh_z2 = Z2[source_indices] if Z2 is not None else None
        mesh_customdata = (
            np.asarray(customdata)[source_indices]
            if customdata is not None
            else grid_df.iloc[source_indices].assign(response=mesh_z1).to_numpy()
        )

        simplices = np.empty((0, 3), dtype=int)
        if domain.has_area and len(source_indices) >= 3:
            try:
                triangulation = Delaunay(np.column_stack([mesh_x, mesh_y]))
                simplices = triangulation.simplices
                if domain.mode == "allowed":
                    centroids = np.column_stack([mesh_x, mesh_y])[simplices].mean(axis=1)
                    boundary = domain.boundary.copy()
                    if center_xy:
                        boundary[:, 0] -= centroid[0]
                        boundary[:, 1] -= centroid[1]
                    if len(boundary) >= 4:
                        hull = ConvexHull(boundary[:-1])
                        inside = np.all(
                            centroids @ hull.equations[:, :-1].T
                            + hull.equations[:, -1]
                            <= 1e-10,
                            axis=1,
                        )
                        simplices = simplices[inside]
            except (QhullError, ValueError):
                simplices = np.empty((0, 3), dtype=int)

        if len(simplices):
            simplex_i, simplex_j, simplex_k = simplices[:, 0], simplices[:, 1], simplices[:, 2]
            fig.add_trace(go.Mesh3d(
                x=mesh_x, y=mesh_y, z=mesh_z1,
                i=simplex_i, j=simplex_j, k=simplex_k,
                intensity=mesh_z1,
                colorscale="Teal",
                showscale=False,
                opacity=1.0,
                customdata=mesh_customdata,
                hovertemplate=hovertemplate,
                name=z_title,
                showlegend=True,
            ))
            if mesh_z2 is not None:
                fig.add_trace(go.Mesh3d(
                    x=mesh_x, y=mesh_y, z=mesh_z2,
                    i=simplex_i, j=simplex_j, k=simplex_k,
                    intensity=mesh_z2,
                    colorscale="Hot",
                    showscale=False,
                    opacity=1,
                    customdata=mesh_customdata,
                    hovertemplate=hovertemplate,
                    name=z2_title,
                    showlegend=True,
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=mesh_x,
                y=mesh_y,
                z=mesh_z1,
                mode="markers",
                marker=dict(size=4, color=mesh_z1, colorscale="Teal"),
                customdata=mesh_customdata,
                hovertemplate=hovertemplate,
                name=z_title,
                showlegend=True,
            ))

        if feasible_region:
            Z_ub = self._response_conditions[z_title].get("upper_limit", None)
            Z_lb = self._response_conditions[z_title].get("lower_limit", None)
            
            if Z_ub is not None and len(simplices):
                Z_plane = np.full(len(mesh_x), Z_ub)
                fig.add_trace(go.Mesh3d(
                    x=mesh_x, y=mesh_y, z=Z_plane,
                    i=simplex_i, j=simplex_j, k=simplex_k,
                    color="rgba(255,0,0,0.3)",
                    name="Upper Bound",
                    showscale=False,
                    hoverinfo="skip",
                    showlegend=True,
                    opacity=0.7
                ))
                
            if Z_lb is not None and len(simplices):
                Z_plane = np.full(len(mesh_x), Z_lb)
                fig.add_trace(go.Mesh3d(
                    x=mesh_x, y=mesh_y, z=Z_plane,
                    i=simplex_i, j=simplex_j, k=simplex_k,
                    color="rgba(0,0,255,0.3)",
                    name="Lower Bound",
                    showscale=False,
                    hoverinfo="skip",
                    showlegend=True,
                    opacity=0.7
                ))
        
        z_range = self._finite_range(Z1, Z2)
        combined_z = np.concatenate(
            [values[np.isfinite(values)] for values in (Z1, Z2) if values is not None]
        )
        z_min = float(combined_z.min()) if combined_z.size else z_range[0]
        if domain.mode == "full":
            self._add_ternary_grid_3d(
                fig, A2, B2, C2, z0=z_min, min=min, max=max, step=0.1
            )

        if show_border and domain.mode == "full":
            z0 = z_min
            def edge(p, q):
                fig.add_trace(go.Scatter3d(
                    x=[p[0], q[0]],
                    y=[p[1], q[1]],
                    z=[z0, z0],
                    mode="lines",
                    line=dict(width=4, color="black"),
                    hoverinfo="skip",
                    showlegend=False
                ))
            edge(A2, B2)
            edge(B2, C2)
            edge(C2, A2)

        if domain.boundary.size:
            boundary = domain.boundary.copy()
            if center_xy:
                boundary[:, 0] -= centroid[0]
                boundary[:, 1] -= centroid[1]
            fig.add_trace(go.Scatter3d(
                x=boundary[:, 0],
                y=boundary[:, 1],
                z=[z_min] * len(boundary),
                mode="lines",
                line=dict(
                    width=4 if domain.mode == "allowed" else 2,
                    color="black",
                    dash="solid" if domain.mode == "allowed" else "dash",
                ),
                name="Allowed domain",
                hoverinfo="skip",
                showlegend=False,
            ))

        if domain.mode == "full":
            self._add_axis_titles_3d(
                fig, A2, B2, C2, a_title, b_title, c_title, z0=z_min
            )

        if constant_levels is not None and len(constant_levels) > 0:
            const_text = "<b>Constant Levels</b><br>" + "<br>".join([f"{k}: {v}" for k, v in constant_levels.items()])
        else:
            const_text = ""
        response_min, response_max = self._finite_extrema(response)
        fig.add_annotation(
            text=f"{const_text}<br>Max. {z_title}: {response_max:.3f}<br>Min. {z_title}: {response_min:.3f}",
            xref="paper", yref="paper",
            x=0.8, y=0.96,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, family="Arial", color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=8
        )

        scene = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title=z_title, showgrid=True, gridcolor="lightgrey"),
            aspectmode="cube",
        )
        if domain.mode == "allowed":
            x_range = np.asarray(domain.x_range)
            y_range = np.asarray(domain.y_range)
            if center_xy:
                x_range = x_range - centroid[0]
                y_range = y_range - centroid[1]
            scene["xaxis"]["range"] = x_range.tolist()
            scene["yaxis"]["range"] = y_range.tolist()
            scene["zaxis"]["range"] = list(z_range)

        fig.update_layout(
            title=dict(
                text=f"<b>3D Surface Plot - {z_title}</b>",
                x=0.47,
                y = 0.92,
                font=dict(
                    size=20,
                    family="Arial",
                    color="black"
                )
            ),
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20),
            scene=scene,
            legend=dict(
                x=0.80,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            ),
            width=800,
            height=500,
            meta={"domain": domain.mode},
        )
        
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=1.21, y=1.26 , z=0.63)  # closer than default
            )
        )

        return fig
    

    def _regression_coefficients(self) -> go.Figure:
        """
        Create a bar chart of regression coefficients with significance markers.
        
        Significance levels:
        * p < 0.05
        ** p < 0.01
        *** p < 0.001
        """
        # Safety check
        if self._mlr_wrapper is None:
            raise ValueError("No OLS model has been computed, please call 'mlr_model_computation' first")

        # Helper function to get significance asterisks
        def get_significance(p_value):
            if p_value < 0.001:
                return "***"
            elif p_value < 0.01:
                return "**"
            elif p_value < 0.05:
                return "*"
            else:
                return ""

        # Organize coefficient data
        coef_dict = {}
        linear = self._model_spec.main
        int2 = [f"{tup[0]}:{tup[1]}" for tup in self._model_spec.interaction2] if self._model_spec.interaction2 is not None else []
        int3 = [f"{tup[0]}:{tup[1]}:{tup[2]}" for tup in self._model_spec.interaction3] if self._model_spec.interaction3 is not None else []
        sq = [f"{var}^2" for var in self._model_spec.quadratic] if self._model_spec.quadratic is not None else []

        # Map variable to term type
        def get_type(var):
            if var in linear:
                return "Linear"
            if len(sq) > 0 and var in sq:
                return "Quadratic"
            if len(int2) > 0 and var in int2:
                return "2-Terms"
            if len(int3) > 0 and var in int3:
                return "3-Terms"

        # Color scheme matching your preferred palette
        colour_map = {
            "Linear": "#003153",
            "2-Terms": "#708090",
            "Quadratic": "#1B9E9E",
            "3-Terms": "#4682B4",
        }

        # Prepare coefficient data for each response
        legend_positions = {}  # Store legend position for each response
        y_ranges = {}  # Store symmetric y-axis ranges for each response
        
        for resp in self._response_list:
            coef_df = self._mlr_wrapper.results[resp].coef.copy()
            coef_df = coef_df.loc[coef_df["Variable"] != "Int"].copy()
            coef_df["Type"] = coef_df["Variable"].map(get_type)
            coef_df["Colour"] = coef_df["Type"].map(colour_map)
            coef_df["Significance"] = coef_df["p_value"].apply(get_significance)
            coef_dict[resp] = coef_df
            
            # Calculate symmetric y-axis range based on maximum absolute coefficient value
            # Include error bars in the calculation
            max_upper = (coef_df["Upper_CI"]).max()
            min_lower = (coef_df["Lower_CI"]).min()
            max_abs_value = max(abs(max_upper), abs(min_lower))
            y_ranges[resp] = [-max_abs_value * 1.1, max_abs_value * 1.1]  # Add 10% padding
            
            # Determine legend position based on last coefficient value
            last_coef_value = coef_df.iloc[-1]["Coefficient"]
            if last_coef_value < 0:
                # Last coefficient is negative, place legend in positive/upper area
                legend_positions[resp] = {"x": 0.94, "y": 0.93, "xanchor": "right", "yanchor": "top"}
            else:
                # Last coefficient is positive, place legend in negative/lower area
                legend_positions[resp] = {"x": 0.94, "y": 0.12, "xanchor": "right", "yanchor": "bottom"}

        # Build figure
        fig = go.Figure()
        buttons = []
        
        # Store annotations for each response
        annotations_dict = {}

        # Add bar traces for each response
        for idx, var in enumerate(self._response_list):
            coef_data = coef_dict[var]
            
            # Add bar trace
            fig.add_trace(go.Bar(
                x=coef_data["Variable"],
                y=coef_data["Coefficient"],
                error_y=dict(
                    type='data',
                    array=(coef_data["Upper_CI"] - coef_data["Lower_CI"]) / 2,
                    visible=True,
                    thickness=2,
                    width=8,
                    color="black"
                ),
                marker_color=coef_data["Colour"],
                customdata=coef_data[["Upper_CI", "Lower_CI", "p_value"]],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Coefficient: %{y:.3f}<br>"
                    "Upper CI: %{customdata[0]:.3f}<br>"
                    "Lower CI: %{customdata[1]:.3f}<br>"
                    "p-value: %{customdata[2]:.4f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
                visible=(idx == 0)
            ))
            
            # Create annotations for significance markers
            sig_annotations = []
            for i, row in coef_data.iterrows():
                if row["Significance"]:
                    # Position asterisks above bars (accounting for error bars)
                    y_pos = row["Coefficient"] + (row["Upper_CI"] - row["Coefficient"]) * 1.15
                    sig_annotations.append(
                        dict(
                            x=row["Variable"],
                            y=y_pos,
                            text=row["Significance"],
                            showarrow=False,
                            font=dict(size=16, color="#EE6C4D", family="Arial"),
                            xref="x",
                            yref="y"
                        )
                    )
            annotations_dict[var] = sig_annotations

        # Add zero reference line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(self._model_matrix.columns) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="#EE6C4D", width=2),
            xref="x",
            yref="y"
        )

        # Add legend traces
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            marker_color="#003153",
            name="Linear",
            showlegend=True
        ))
        if len(int2):
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color="#708090",
                name="2-Term",
                showlegend=True
            ))
        if len(sq):
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color="#1B9E9E",
                name="Quadratic",
                showlegend=True
            ))
        if len(int3):
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color="#4682B4",
                name="3-Term",
                showlegend=True
            ))

        # Build dropdown menu
        n_resp = len(self._response_list)
        n_legend = 4  # number of legend traces
        
        for i, var in enumerate(self._response_list):
            # Visibility: show current response bar and all legend items
            vis = [False] * n_resp + [True] * n_legend
            vis[i] = True

            buttons.append(
                dict(
                    label=var,
                    method="update",
                    args=[
                        {"visible": vis},
                        {
                            "yaxis.title.text": "<b>Coefficient Value</b>",
                            "yaxis.range": y_ranges[var],
                            "legend": legend_positions[var],
                            "annotations": [
                                dict(
                                    text=f"<b>Regression Coefficients - {var}</b>",
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.13,
                                    showarrow=False,
                                    font=dict(size=20, family="Arial", color="black")
                                ),
                                dict(
                                    text="* p<0.05   ** p<0.01   *** p<0.001",
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.05,
                                    showarrow=False,
                                    font=dict(size=11, family="Arial", color="#666"),
                                    align="center"
                                )
                            ] + annotations_dict[var]
                        }
                    ]
                )
            )

        # Style axes
        fig.update_xaxes(
            title_text="<b>Model Terms</b>",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            showgrid=False,
            tickangle=-45,
            domain=[0.05, 0.95]
        )
        fig.update_yaxes(
            title_text="<b>Coefficient Value</b>",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
            domain=[0.1, 0.95],
            range=y_ranges[self._response_list[0]]
        )

        # Layout configuration
        fig.update_layout(
            annotations=[
                dict(
                    text=f"<b>Regression Coefficients - {self._response_list[0]}</b>",
                    xref="paper", yref="paper",
                    x=0.5, y=1.13,
                    showarrow=False,
                    font=dict(size=20, family="Arial", color="black")
                ),
                dict(
                    text="* p<0.05   ** p<0.01   *** p<0.001",
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=11, family="Arial", color="#666"),
                    align="center"
                )
            ] + annotations_dict[self._response_list[0]],
            width=800,
            height=500,
            font=dict(size=12, family="Arial"),
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.13,
                yanchor="top",
                pad=dict(r=10, t=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )],
            margin=dict(l=40, r=40, t=50, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(
                x=legend_positions[self._response_list[0]]["x"],
                y=legend_positions[self._response_list[0]]["y"],
                xanchor=legend_positions[self._response_list[0]]["xanchor"],
                yanchor=legend_positions[self._response_list[0]]["yanchor"],
                bgcolor="white"
            ),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#003153",
                font=dict(size=12, family="Arial", color="black")
            )
        )

        return fig
    
    def _model_diagnostics(self, cv: bool = False) -> go.Figure:
        """
        Create comprehensive model diagnostic plots in a 2x2 subplot layout.
        
        Subplots:
        1. Top-left: Experimental vs Predicted
           - Shows how well model predictions match actual experimental values
           - Perfect fit would lie on 45° line
           - Includes R² (coefficient of determination) showing % variance explained
           - Includes RMSE (root mean square error) showing average prediction error
        
        2. Top-right: Residuals vs Fitted Values
           - Shows patterns in prediction errors across the response range
           - Should show random scatter around zero (no patterns)
           - Funnel shape indicates heteroscedasticity (non-constant variance)
           - Curved patterns indicate non-linearity
        
        3. Bottom-left: Normal Q-Q Plot
           - Checks if residuals follow a normal distribution
           - Points should follow the diagonal line
           - Deviations suggest violation of normality assumption
           - Includes Shapiro-Wilk test p-value (p>0.05 = normal)
        
        4. Bottom-right: Histogram of Residuals
           - Shows distribution shape of prediction errors
           - Should be bell-shaped and centered at zero
           - Shows mean (should be ~0) and std deviation
           - Includes ±2σ bounds (95% of data should fall within)
        
        Parameters:
        cv : bool
            If True, uses cross-validation predictions (more realistic performance)
            If False, uses training set predictions (may be optimistic)
        """
        
        from scipy import stats
        
        # Safety check
        if self._mlr_wrapper is None:
            raise ValueError("No OLS model has been computed, please call 'mlr_model_computation' first")
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Experimental vs Predicted",
                "Residuals vs Fitted Values", 
                "Normal Q-Q Plot",
                "Histogram of Residuals"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )
        
        buttons = []
        
        # Store annotations, statistics, and axis ranges for each response
        annotations_dict = {}
        axis_ranges_dict = {}
        
        for idx, var in enumerate(self._response_list):
            exp = self._responses[var].values
            
            if cv:
                pred = self._mlr_wrapper.results[var].y_hat_cv
            else:
                pred = self._mlr_wrapper.results[var].y_hat.values
            
            residuals = exp - pred
            
            # Calculate statistics
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals, ddof=1)
            
            # Shapiro-Wilk normality test
            if len(residuals) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
            else:
                shapiro_p = np.nan
            
            # --- Subplot 1: Experimental vs Predicted ---
            y_min = exp.min()
            y_max = exp.max()
            y_range = y_max - y_min
            y_min_plot = y_min - 0.1 * y_range
            y_max_plot = y_max + 0.1 * y_range
            
            # Store axis ranges for this response
            axis_ranges_dict[var] = {
                'x': [y_min_plot, y_max_plot],
                'y': [y_min_plot, y_max_plot]
            }
            
            # Scatter points
            fig.add_trace(
                go.Scatter(
                    x=exp, y=pred,
                    mode="markers",
                    marker=dict(color="#003153", size=7, opacity=0.7),
                    name="",
                    hovertemplate=(
                        "Exp: %{x:.3f}<br>"
                        "Pred: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=(idx == 0)
                ),
                row=1, col=1
            )
            
            # 45-degree line
            fig.add_trace(
                go.Scatter(
                    x=[y_min_plot, y_max_plot],
                    y=[y_min_plot, y_max_plot],
                    mode="lines",
                    line=dict(dash="dash", color="#EE6C4D", width=2),
                    name="",
                    showlegend=False,
                    visible=(idx == 0),
                    hoverinfo="skip"
                ),
                row=1, col=1
            )
            
            # --- Subplot 2: Residuals vs Fitted ---
            fig.add_trace(
                go.Scatter(
                    x=pred, y=residuals,
                    mode="markers",
                    marker=dict(color="#003153", size=7, opacity=0.7),
                    name="",
                    hovertemplate=(
                        "Fitted: %{x:.3f}<br>"
                        "Residual: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=(idx == 0)
                ),
                row=1, col=2
            )
            
            # Zero line
            fig.add_trace(
                go.Scatter(
                    x=[pred.min(), pred.max()],
                    y=[0, 0],
                    mode="lines",
                    line=dict(dash="dash", color="#EE6C4D", width=2),
                    name="",
                    showlegend=False,
                    visible=(idx == 0),
                    hoverinfo="skip"
                ),
                row=1, col=2
            )
            
            # --- Subplot 3: Q-Q Plot ---
            # Theoretical quantiles (normal distribution)
            sorted_residuals = np.sort(residuals)
            n = len(sorted_residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_residuals,
                    mode="markers",
                    marker=dict(color="#003153", size=6, opacity=0.7),
                    name="",
                    hovertemplate=(
                        "Theoretical: %{x:.2f}<br>"
                        "Sample: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=(idx == 0)
                ),
                row=2, col=1
            )
            
            # Diagonal reference line
            qq_min = min(theoretical_quantiles.min(), sorted_residuals.min())
            qq_max = max(theoretical_quantiles.max(), sorted_residuals.max())
            fig.add_trace(
                go.Scatter(
                    x=[qq_min, qq_max],
                    y=[qq_min, qq_max],
                    mode="lines",
                    line=dict(dash="dash", color="#EE6C4D", width=2),
                    name="",
                    showlegend=False,
                    visible=(idx == 0),
                    hoverinfo="skip"
                ),
                row=2, col=1
            )
            
            # --- Subplot 4: Histogram of Residuals ---
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    marker=dict(color="#003153", opacity=0.7, line=dict(color="black", width=1)),
                    name="",
                    showlegend=False,
                    visible=(idx == 0),
                    hovertemplate=(
                        "Residual range: %{x}<br>"
                        "Count: %{y}<br>"
                        "<extra></extra>"
                    ),
                    nbinsx=15
                ),
                row=2, col=2
            )
            
            # Store annotations for this response (using paper coordinates)
            annotations_dict[var] = [
                # Shapiro-Wilk p-value (subplot 3)
                dict(
                    text=f"Shapiro-Wilk<br>p = {shapiro_p:.4f}" if not np.isnan(shapiro_p) else "Shapiro-Wilk<br>N/A",
                    xref="paper", yref="paper",
                    x=0.03, y=0.39,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=10, family="Arial", color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                ),
                # Mean and Std Dev (subplot 4)
                dict(
                    text=f"μ = {residual_mean:.4f}<br>σ = {residual_std:.4f}",
                    xref="paper", yref="paper",
                    x=0.66, y=0.39,
                    xanchor="right", yanchor="top",
                    showarrow=False,
                    font=dict(size=10, family="Arial", color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                )
            ]
        
        # Build dropdown menu
        n_resp = len(self._response_list)
        traces_per_response = 7  # 2 + 2 + 2 + 1
        
        for i, var in enumerate(self._response_list):
            vis = [False] * (n_resp * traces_per_response)
            for j in range(traces_per_response):
                vis[i * traces_per_response + j] = True
            
            buttons.append(
                dict(
                    label=var,
                    method="update",
                    args=[
                        {"visible": vis},
                        {
                            "annotations": list(fig.layout.annotations[:4]) + annotations_dict[var],
                            "xaxis.range": axis_ranges_dict[var]['x'],
                            "yaxis.range": axis_ranges_dict[var]['y']
                        }
                    ]
                )
            )
        
        # Update subplot titles styling
        for annotation in fig.layout.annotations[:4]:
            annotation.font.size = 13
            annotation.font.family = "Arial"
            annotation.font.color = "black"
            annotation.update(
                y=annotation.y + 0.01,  # Shift up by 0.01
                font=dict(size=13, family="Arial", color="black")
            )
        
        # Update axes
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True,
            showgrid=True, gridcolor="lightgray", zeroline=False
        )
        fig.update_yaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True,
            showgrid=True, gridcolor="lightgray", zeroline=False
        )
        
        # Specific axis labels and ranges
        first_response = self._response_list[0]
        fig.update_xaxes(
            title_text="<b>Experimental</b>", 
            range=axis_ranges_dict[first_response]['x'],
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="<b>Predicted</b>", 
            range=axis_ranges_dict[first_response]['y'],
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="<b>Fitted Values</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Residuals</b>", row=1, col=2)
        
        fig.update_xaxes(title_text="<b>Theoretical Quantiles</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Sample Quantiles</b>", row=2, col=1)
        
        fig.update_xaxes(title_text="<b>Residual Value</b>", row=2, col=2)
        fig.update_yaxes(title_text="<b>Frequency</b>", row=2, col=2)
        
        # Main title
        title_text = "<b>Model Diagnostics (CV)</b>" if cv else "<b>Model Diagnostics</b>"
        
        # Global layout
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5, y=0.98,
                xanchor="center", yanchor="top",
                font=dict(size=20, family="Arial", color="black")
            ),
            width=1000,
            height=800,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=11, family="Arial"),
            margin=dict(l=60, r=40, t=80, b=60),
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.07,
                yanchor="top",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )],
            annotations=list(fig.layout.annotations[:4]) + annotations_dict[self._response_list[0]],
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#003153",
                font=dict(size=11, family="Arial", color="black")
            )
        )
        
        return fig

    def _exp_vs_pred(self, cv : bool = False) -> go.Figure:

            if self._mlr_wrapper is None:
                raise ValueError("No OLS model has been computed, please call 'mlr_model_computation' first")

            fig = go.Figure()
            buttons = []

            for idx, var in enumerate(self._response_list):
                exp  = self._responses[var]
                if cv:
                    pred = self._mlr_wrapper.results[var].y_hat_cv
                else:
                    pred = self._mlr_wrapper.results[var].y_hat

                vmin, vmax = min(exp.min(), pred.min()), max(exp.max(), pred.max())

                # scatter trace
                fig.add_trace(
                    go.Scatter(
                        x=exp, y=pred, mode="markers+text",
                        marker=dict(color="#003153", size=8),
                        text=[str(i) for i in exp.index],
                        textposition="top right",
                        hovertemplate=(f"Exp. {var}: %{{x:.2f}}<br>"
                                        f"Pred. {var}: %{{y:.2f}}<br>"
                                        "Index: %{text}<extra></extra>"),
                        showlegend=False,
                        visible=(idx == 0)
                    )
                )

                # 45 degrees line
                fig.add_trace(
                    go.Scatter(
                        x=[vmin-1, vmax+1],
                        y=[vmin-1, vmax+1],
                        mode="lines",
                        line=dict(dash="dash", color="#EE6C4D"),
                        showlegend=False,
                        visible=(idx == 0)
                    )
                )

            # Dropdown Toggle
            n_traces = fig.data.__len__()
            for i, var in enumerate(self._response_list):
                vis = [False] * n_traces
                vis[2*i]     = True
                vis[2*i + 1] = True

                # Axis titles
                buttons.append(dict(
                    label  = var,
                    method = "update",
                    args   = [
                        {"visible": vis},
                        {
                            "xaxis.title.text": f"Experimental {var}",
                            "yaxis.title.text": f"Predicted {var}"
                        }
                    ]
                ))

            # Layout Tweaks
            fig.update_xaxes(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True
            )
            fig.update_yaxes(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True
            )
            if cv: 
                title_text = "<b>Experimental vs Predicted CV<b>"
            else:
                title_text = "<b>Experimental vs Predicted<b>"
            fig.update_layout(
                title=dict(text=title_text,
                                        x=0.5, y=0.91, xanchor="center", yanchor="top",
                                        font=dict(size=26, color="black", family="Arial")),
                xaxis_title=f"Experimental {self._response_list[0]}",
                yaxis_title=f"Predicted {self._response_list[0]}",
                updatemenus=[dict(
                    buttons   = buttons,
                    direction = "down",
                    x=0.0,  xanchor="left",
                    y=1.12, yanchor="top"
                )],
                width=800, height=600, font=dict(size=14),
                margin=dict(t=100),
                plot_bgcolor="white"
            )

            return fig
    
    def _validate_plot_response(self, response: str) -> None:
        if self._mlr_wrapper is None:
            raise ValueError(
                "No OLS model has been computed, please call "
                "'compute_mlr_model' first."
            )
        if response not in (self._response_list or []):
            raise ValueError(
                f"Response '{response}' not found. Available responses: "
                f"{list(self._response_list or [])}."
            )

    def _diagnostic_values(
        self,
        response: str,
        cv: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return observed, predicted, and observed-minus-predicted values."""
        self._validate_plot_response(response)
        observed = self._responses[response].to_numpy(dtype=float)
        result = self._mlr_wrapper.results[response]
        predicted = np.asarray(
            result.y_hat_cv if cv else result.y_hat,
            dtype=float,
        ).ravel()
        return observed, predicted, observed - predicted

    def _experimental_run_axis(
        self,
        n_runs: int,
        x_axis: Literal["exp_order", "exp_idx", "sequence"] = "exp_order",
    ) -> tuple[np.ndarray, str]:
        """Resolve the best available experimental-run coordinate."""
        if x_axis not in {"exp_order", "exp_idx", "sequence"}:
            raise ValueError(
                "x_axis must be 'exp_order', 'exp_idx', or 'sequence'."
            )

        metadata = getattr(self, "_experimental_metadata", None)
        if isinstance(metadata, pd.DataFrame) and len(metadata) == n_runs:
            preferred = []
            if x_axis == "exp_order":
                preferred = [
                    ("Exp. Order", "Experimental Order"),
                    ("Exp. Idx", "Experimental Run"),
                ]
            elif x_axis == "exp_idx":
                preferred = [("Exp. Idx", "Experimental Run")]

            for column, label in preferred:
                if column in metadata.columns:
                    values = pd.to_numeric(
                        metadata[column], errors="coerce"
                    ).to_numpy()
                    if np.isfinite(values).all():
                        return values, label

        return np.arange(1, n_runs + 1), "Experimental Run"

    @staticmethod
    def _style_model_plot(
        fig: go.Figure,
        *,
        title: str,
        x_title: str,
        y_title: str,
        width: int = 800,
        height: int = 560,
    ) -> go.Figure:
        fig.update_xaxes(
            title_text=f"<b>{x_title}</b>",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
        )
        fig.update_yaxes(
            title_text=f"<b>{y_title}</b>",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                y=0.96,
                xanchor="center",
                yanchor="top",
                font=dict(size=20, color="black", family="Arial"),
            ),
            width=width,
            height=height,
            margin=dict(l=70, r=35, t=85, b=65),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=12, family="Arial"),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#003153",
                font=dict(size=11, family="Arial", color="black"),
            ),
        )
        return fig

    def _regression_coefficients_for_response(self, response: str) -> go.Figure:
        """Render regression coefficients for one response."""
        self._validate_plot_response(response)
        coef_df = self._mlr_wrapper.results[response].coef.copy()
        coef_df = coef_df.loc[coef_df["Variable"] != "Int"].copy()

        linear = set(self._model_spec.main or [])
        interaction2 = {
            f"{a}:{b}" for a, b in (self._model_spec.interaction2 or [])
        }
        interaction3 = {
            f"{a}:{b}:{c}"
            for a, b, c in (self._model_spec.interaction3 or [])
        }
        quadratic = {
            f"{factor}^2" for factor in (self._model_spec.quadratic or [])
        }

        def term_type(term: str) -> str:
            if term in quadratic:
                return "Quadratic"
            if term in interaction3:
                return "3-Term"
            if term in interaction2:
                return "2-Term"
            if term in linear:
                return "Linear"
            return "Other"

        colors = {
            "Linear": "#003153",
            "2-Term": "#708090",
            "Quadratic": "#1B9E9E",
            "3-Term": "#4682B4",
            "Other": "#777777",
        }
        coef_df["Type"] = coef_df["Variable"].map(term_type)
        coef_df["Color"] = coef_df["Type"].map(colors)
        coef_df["Significance"] = coef_df["p_value"].map(
            lambda value: (
                "***"
                if value < 0.001
                else "**"
                if value < 0.01
                else "*"
                if value < 0.05
                else ""
            )
        )

        fig = go.Figure(
            go.Bar(
                x=coef_df["Variable"],
                y=coef_df["Coefficient"],
                error_y=dict(
                    type="data",
                    array=(
                        coef_df["Upper_CI"] - coef_df["Lower_CI"]
                    )
                    / 2,
                    visible=True,
                    thickness=2,
                    width=8,
                    color="black",
                ),
                marker_color=coef_df["Color"],
                customdata=coef_df[
                    ["Upper_CI", "Lower_CI", "p_value", "Type"]
                ],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Type: %{customdata[3]}<br>"
                    "Coefficient: %{y:.3f}<br>"
                    "Upper CI: %{customdata[0]:.3f}<br>"
                    "Lower CI: %{customdata[1]:.3f}<br>"
                    "p-value: %{customdata[2]:.4f}"
                    "<extra></extra>"
                ),
                text=coef_df["Significance"],
                textposition="outside",
                showlegend=False,
            )
        )
        fig.add_hline(y=0, line_color="#EE6C4D", line_width=2)
        self._style_model_plot(
            fig,
            title=f"Regression Coefficients - {response}",
            x_title="Model Terms",
            y_title="Coefficient Value",
        )
        fig.update_xaxes(tickangle=-45)
        fig.add_annotation(
            text="* p<0.05   ** p<0.01   *** p<0.001",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.06,
            showarrow=False,
            font=dict(size=11, color="#666666"),
        )
        return fig

    def _exp_vs_pred_for_response(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        observed, predicted, _ = self._diagnostic_values(response, cv)
        lower, upper = self._exact_range(
            np.concatenate([observed, predicted])
        )
        padding = 0.05 * (upper - lower)
        lower -= padding
        upper += padding

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=observed,
                y=predicted,
                mode="markers",
                marker=dict(color="#003153", size=8),
                customdata=np.arange(1, len(observed) + 1),
                hovertemplate=(
                    "Experimental: %{x:.3f}<br>"
                    "Predicted: %{y:.3f}<br>"
                    "Experimental Run: %{customdata}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line=dict(dash="dash", color="#EE6C4D", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        suffix = " CV" if cv else ""
        self._style_model_plot(
            fig,
            title=f"Experimental vs Predicted{suffix} - {response}",
            x_title=f"Experimental {response}",
            y_title=f"Predicted {response}",
        )
        fig.update_xaxes(range=[lower, upper])
        fig.update_yaxes(range=[lower, upper], scaleanchor="x", scaleratio=1)
        return fig

    def _confirmation_exp_vs_pred_for_response(
        self,
        response: str,
    ) -> go.Figure:
        """Render observed confirmation means against model predictions."""
        results = self.get_confirmation_results(response)
        observed = results["Observed"].to_numpy(dtype=float)
        predicted = results["Predicted"].to_numpy(dtype=float)
        lower, upper = self._exact_range(
            np.concatenate([observed, predicted])
        )
        padding = 0.05 * (upper - lower)
        lower -= padding
        upper += padding

        customdata = results[["Setting", "n", "Residual", "Within PI"]].to_numpy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=observed,
                y=predicted,
                mode="markers",
                marker=dict(color="#003153", size=8),
                customdata=customdata,
                hovertemplate=(
                    "Setting: %{customdata[0]}<br>"
                    "Replicates: %{customdata[1]}<br>"
                    "Observed mean: %{x:.3f}<br>"
                    "Predicted: %{y:.3f}<br>"
                    "Residual: %{customdata[2]:.3f}<br>"
                    "Within PI: %{customdata[3]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line=dict(dash="dash", color="#EE6C4D", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        self._style_model_plot(
            fig,
            title=f"Confirmation Experimental vs Predicted - {response}",
            x_title=f"Observed Mean {response}",
            y_title=f"Predicted {response}",
        )
        fig.update_xaxes(range=[lower, upper])
        fig.update_yaxes(range=[lower, upper], scaleanchor="x", scaleratio=1)
        return fig

    def _confirmation_residuals_for_response(
        self,
        response: str,
    ) -> go.Figure:
        """Render confirmation residuals against observed group means."""
        results = self.get_confirmation_results(response)
        observed = results["Observed"].to_numpy(dtype=float)
        residuals = results["Residual"].to_numpy(dtype=float)
        customdata = results[["Setting", "n", "Predicted", "Within PI"]].to_numpy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=observed,
                y=residuals,
                mode="markers",
                marker=dict(color="#003153", size=8),
                customdata=customdata,
                hovertemplate=(
                    "Setting: %{customdata[0]}<br>"
                    "Replicates: %{customdata[1]}<br>"
                    "Observed mean: %{x:.3f}<br>"
                    "Predicted: %{customdata[2]:.3f}<br>"
                    "Residual: %{y:.3f}<br>"
                    "Within PI: %{customdata[3]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        x_min, x_max = self._exact_range(observed)
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[0, 0],
                mode="lines",
                line=dict(dash="dash", color="#EE6C4D", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        return self._style_model_plot(
            fig,
            title=f"Confirmation Residuals vs Response - {response}",
            x_title=f"Observed Mean {response}",
            y_title="Residuals",
        )

    def _residuals_vs_fitted_for_response(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        _, predicted, residuals = self._diagnostic_values(response, cv)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=residuals,
                mode="markers",
                marker=dict(color="#003153", size=8),
                customdata=np.arange(1, len(residuals) + 1),
                hovertemplate=(
                    "Fitted: %{x:.3f}<br>"
                    "Residual: %{y:.3f}<br>"
                    "Experimental Run: %{customdata}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        x_min, x_max = self._exact_range(predicted)
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[0, 0],
                mode="lines",
                line=dict(dash="dash", color="#EE6C4D", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        suffix = " CV" if cv else ""
        return self._style_model_plot(
            fig,
            title=f"Residuals vs Fitted{suffix} - {response}",
            x_title="Fitted Values",
            y_title="Residuals",
        )

    def _residuals_by_run_for_response(
        self,
        response: str,
        cv: bool = False,
        x_axis: Literal["exp_order", "exp_idx", "sequence"] = "exp_order",
    ) -> go.Figure:
        _, _, residuals = self._diagnostic_values(response, cv)
        run_axis, run_label = self._experimental_run_axis(
            len(residuals), x_axis
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=run_axis,
                y=residuals,
                mode="markers",
                marker=dict(color="#003153", size=8),
                customdata=np.arange(1, len(residuals) + 1),
                hovertemplate=(
                    f"{run_label}: %{{x}}<br>"
                    "Residual: %{y:.3f}<br>"
                    "Model Row: %{customdata}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        x_min, x_max = self._exact_range(run_axis)
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[0, 0],
                mode="lines",
                line=dict(dash="dash", color="#EE6C4D", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        suffix = " CV" if cv else ""
        return self._style_model_plot(
            fig,
            title=f"Residuals by Experimental Run{suffix} - {response}",
            x_title=run_label,
            y_title=f"Residuals ({response})",
        )

    def _qq_residuals_for_response(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        from scipy import stats

        _, _, residuals = self._diagnostic_values(response, cv)
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(
            residuals, dist="norm"
        )
        line = slope * theoretical + intercept
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=ordered,
                mode="markers",
                marker=dict(color="#003153", size=8),
                hovertemplate=(
                    "Theoretical Quantile: %{x:.3f}<br>"
                    "Residual Quantile: %{y:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=line,
                mode="lines",
                line=dict(dash="dash", color="#EE6C4D", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        suffix = " CV" if cv else ""
        return self._style_model_plot(
            fig,
            title=f"Normal Q-Q Plot{suffix} - {response}",
            x_title="Theoretical Quantiles",
            y_title="Residual Quantiles",
        )

    def _residuals_histogram_for_response(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        _, _, residuals = self._diagnostic_values(response, cv)
        fig = go.Figure(
            go.Histogram(
                x=residuals,
                marker=dict(
                    color="#003153",
                    opacity=0.78,
                    line=dict(color="black", width=1),
                ),
                nbinsx=min(15, max(5, int(np.sqrt(len(residuals))))),
                hovertemplate=(
                    "Residual Range: %{x}<br>"
                    "Count: %{y}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#EE6C4D",
            line_width=2,
        )
        suffix = " CV" if cv else ""
        return self._style_model_plot(
            fig,
            title=f"Residual Distribution{suffix} - {response}",
            x_title="Residual",
            y_title="Frequency",
        )

    def _model_diagnostics_for_response(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        """Render the four standard diagnostics for one response."""
        figures = (
            self._exp_vs_pred_for_response(response, cv),
            self._residuals_vs_fitted_for_response(response, cv),
            self._qq_residuals_for_response(response, cv),
            self._residuals_histogram_for_response(response, cv),
        )
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Experimental vs Predicted",
                "Residuals vs Fitted Values",
                "Normal Q-Q Plot",
                "Histogram of Residuals",
            ),
            horizontal_spacing=0.12,
            vertical_spacing=0.16,
        )
        for source, row, column in zip(
            figures,
            (1, 1, 2, 2),
            (1, 2, 1, 2),
        ):
            for trace in source.data:
                fig.add_trace(trace, row=row, col=column)

        axis_titles = (
            ("Experimental", "Predicted"),
            ("Fitted Values", "Residuals"),
            ("Theoretical Quantiles", "Residual Quantiles"),
            ("Residual", "Frequency"),
        )
        for (x_title, y_title), row, column in zip(
            axis_titles,
            (1, 1, 2, 2),
            (1, 2, 1, 2),
        ):
            fig.update_xaxes(
                title_text=f"<b>{x_title}</b>", row=row, col=column
            )
            fig.update_yaxes(
                title_text=f"<b>{y_title}</b>", row=row, col=column
            )

        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black",
            mirror=True, showgrid=True, gridcolor="lightgray"
        )
        fig.update_yaxes(
            showline=True, linewidth=2, linecolor="black",
            mirror=True, showgrid=True, gridcolor="lightgray"
        )
        suffix = " CV" if cv else ""
        fig.update_layout(
            title=dict(
                text=f"<b>Model Diagnostics{suffix} - {response}</b>",
                x=0.5,
                y=0.98,
                xanchor="center",
                yanchor="top",
                font=dict(size=20, family="Arial", color="black"),
            ),
            width=1000,
            height=800,
            margin=dict(l=65, r=35, t=90, b=60),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=11, family="Arial"),
            hovermode="closest",
        )
        return fig

    def _residuals_dashboard(
        self,
        response: str | None = None,
        cv: bool = False,
        x_axis: Literal["exp_order", "exp_idx", "sequence"] = "exp_order",
    ) -> go.Figure:
        """Render one run-residual plot or a backward-compatible dropdown."""
        responses = [response] if response is not None else list(
            self._response_list or []
        )
        if not responses:
            raise ValueError("No responses are available for plotting.")

        figures = [
            self._residuals_by_run_for_response(item, cv, x_axis)
            for item in responses
        ]
        if response is not None:
            return figures[0]

        fig = go.Figure()
        for index, source in enumerate(figures):
            for trace in source.data:
                trace.visible = index == 0
                fig.add_trace(trace)

        buttons = []
        traces_per_response = len(figures[0].data)
        for index, item in enumerate(responses):
            visible = [False] * len(fig.data)
            start = index * traces_per_response
            visible[start:start + traces_per_response] = [True] * traces_per_response
            _, run_label = self._experimental_run_axis(
                len(self._responses), x_axis
            )
            suffix = " CV" if cv else ""
            buttons.append(
                dict(
                    label=item,
                    method="update",
                    args=[
                        {"visible": visible},
                        {
                            "title.text": (
                                "<b>Residuals by Experimental Run"
                                f"{suffix} - {item}</b>"
                            ),
                            "xaxis.title.text": f"<b>{run_label}</b>",
                            "yaxis.title.text": f"<b>Residuals ({item})</b>",
                        },
                    ],
                )
            )
        fig.update_layout(figures[0].layout)
        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )]
        )
        return fig

    def _base_prediction_row(self) -> dict[str, float]:
        """Return a valid coded reference point for model profiles."""
        base: dict[str, float] = {}
        mixture_names = []
        for name, factor in self._factors.items():
            if factor.type == "cont":
                base[name] = 0.0
            elif factor.type == "cat":
                base[name] = factor.reference_code
            elif factor.type == "mix":
                mixture_names.append(name)

        if mixture_names:
            mixture_reference = self._coded_design_matrix[
                mixture_names
            ].mean(axis=0).to_numpy(dtype=float)
            mixture_reference = np.clip(
                mixture_reference,
                [self._factors[name].lower_bound for name in mixture_names],
                [self._factors[name].upper_bound for name in mixture_names],
            )
            mixture_reference /= mixture_reference.sum()
            base.update(dict(zip(mixture_names, mixture_reference)))
        return base

    def _validate_effect_factor(self, factor: str) -> None:
        if factor not in self._factors:
            raise ValueError(
                f"Factor '{factor}' not found. Available factors: "
                f"{list(self._factors)}."
            )
        if self._factors[factor].type == "mix":
            raise ValueError(
                "Classical main-effect and interaction plots are not valid "
                "for mixture components. Use plot_mixture_trace() or a "
                "mixture response contour/surface instead."
            )

    def _effect_display_values(
        self,
        factor: str,
        coded_values: np.ndarray,
        coded: bool,
    ) -> np.ndarray:
        if coded:
            return np.asarray(coded_values)
        matrix = pd.DataFrame(
            {name: [value] for name, value in self._base_prediction_row().items()}
        )
        values = []
        for value in coded_values:
            row = matrix.copy()
            row[factor] = value
            values.append(self._decode_matrix(row).iloc[0][factor])
        return np.asarray(values)

    def _main_effect_data(
        self,
        response: str,
        factor: str,
        coded: bool,
        n_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_plot_response(response)
        self._validate_effect_factor(factor)
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        factor_info = self._factors[factor]
        if factor_info.type == "cont":
            coded_values = np.linspace(-1.0, 1.0, n_points)
        else:
            coded_values = np.asarray(
                factor_info.coded_levels, dtype=float
            )

        base = self._base_prediction_row()
        grid = pd.DataFrame([base] * len(coded_values))
        grid[factor] = coded_values
        predictions = self.predict(grid, [response])[response].to_numpy()
        display_values = self._effect_display_values(
            factor, coded_values, coded
        )
        return display_values, predictions

    def _main_effect_model(
        self,
        response: str,
        factor: str,
        coded: bool = True,
        n_points: int = 50,
    ) -> go.Figure:
        x_values, predictions = self._main_effect_data(
            response, factor, coded, n_points
        )
        is_categorical = self._factors[factor].type == "cat"
        fig = go.Figure(
            go.Scatter(
                x=x_values,
                y=predictions,
                mode="lines+markers" if is_categorical else "lines",
                line=dict(color="#003153", width=3),
                marker=dict(color="#003153", size=9),
                hovertemplate=(
                    f"{factor}: %{{x}}<br>"
                    f"Predicted {response}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        baseline = self._base_prediction_row()
        baseline_prediction = float(
            self.predict(pd.DataFrame([baseline]), [response]).iloc[0][response]
        )
        fig.add_hline(
            y=baseline_prediction,
            line_dash="dash",
            line_color="#777777",
            line_width=1.5,
        )
        return self._style_model_plot(
            fig,
            title=f"Main Effect - {factor} - {response}",
            x_title=factor,
            y_title=f"Predicted {response}",
        )

    def _main_effects_model(
        self,
        response: str,
        coded: bool = True,
        n_points: int = 50,
        ncols: int = 3,
    ) -> go.Figure:
        mixture_factors = [
            name for name, factor in self._factors.items()
            if factor.type == "mix"
        ]
        if mixture_factors:
            raise ValueError(
                "Classical main-effect plots are not valid for mixture "
                "components. Use plot_mixture_trace() or mixture response "
                "contours/surfaces instead."
            )
        self._validate_plot_response(response)
        factors = list(self._factors)
        ncols = max(1, min(ncols, len(factors)))
        nrows = int(np.ceil(len(factors) / ncols))
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=factors,
            horizontal_spacing=0.08,
            vertical_spacing=0.16,
        )

        all_predictions = []
        for index, factor in enumerate(factors):
            x_values, predictions = self._main_effect_data(
                response, factor, coded, n_points
            )
            all_predictions.extend(predictions)
            row = index // ncols + 1
            column = index % ncols + 1
            categorical = self._factors[factor].type == "cat"
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=predictions,
                    mode="lines+markers" if categorical else "lines",
                    line=dict(color="#003153", width=3),
                    marker=dict(color="#003153", size=8),
                    hovertemplate=(
                        f"{factor}: %{{x}}<br>"
                        f"Predicted {response}: %{{y:.3f}}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
            fig.update_xaxes(
                title_text=f"<b>{factor}</b>", row=row, col=column
            )
            if column == 1:
                fig.update_yaxes(
                    title_text=f"<b>Predicted {response}</b>",
                    row=row,
                    col=column,
                )

        y_min, y_max = self._padded_range(
            np.asarray(all_predictions), padding=0.08
        )
        fig.update_yaxes(range=[y_min, y_max])
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black",
            mirror=True, showgrid=True, gridcolor="lightgray"
        )
        fig.update_yaxes(
            showline=True, linewidth=2, linecolor="black",
            mirror=True, showgrid=True, gridcolor="lightgray"
        )
        fig.update_layout(
            title=dict(
                text=f"<b>Main Effects - {response}</b>",
                x=0.5, y=0.98, xanchor="center", yanchor="top",
                font=dict(size=20, family="Arial", color="black"),
            ),
            width=360 * ncols,
            height=340 * nrows,
            margin=dict(l=65, r=30, t=90, b=55),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=11, family="Arial"),
        )
        return fig

    def _interaction_profiles(
        self,
        response: str,
        factor1: str,
        factor2: str,
        coded: bool,
        n_points: int,
    ) -> tuple[str, list[tuple[str, np.ndarray, np.ndarray]]]:
        self._validate_plot_response(response)
        self._validate_effect_factor(factor1)
        self._validate_effect_factor(factor2)
        if factor1 == factor2:
            raise ValueError("factor1 and factor2 must be different.")
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        first = self._factors[factor1]
        second = self._factors[factor2]
        if first.type == "cat" and second.type == "cont":
            factor1, factor2 = factor2, factor1
            first, second = second, first

        if first.type == "cont":
            x_codes = np.linspace(-1.0, 1.0, n_points)
        else:
            x_codes = np.asarray(first.coded_levels, dtype=float)

        if second.type == "cont":
            group_codes = np.array([-1.0, 0.0, 1.0])
        else:
            group_codes = np.asarray(second.coded_levels, dtype=float)

        x_display = self._effect_display_values(
            factor1, x_codes, coded
        )
        group_display = self._effect_display_values(
            factor2, group_codes, coded
        )
        profiles = []
        base = self._base_prediction_row()
        for group_code, group_label in zip(group_codes, group_display):
            grid = pd.DataFrame([base] * len(x_codes))
            grid[factor1] = x_codes
            grid[factor2] = group_code
            predictions = self.predict(
                grid, [response]
            )[response].to_numpy()
            profiles.append(
                (f"{factor2} = {group_label}", x_display, predictions)
            )
        return factor1, profiles

    def _interaction_model(
        self,
        response: str,
        factor1: str,
        factor2: str,
        coded: bool = True,
        n_points: int = 50,
    ) -> go.Figure:
        x_factor, profiles = self._interaction_profiles(
            response, factor1, factor2, coded, n_points
        )
        palette = ["#003153", "#05A6A6", "#EE6C4D", "#4682B4"]
        fig = go.Figure()
        for index, (label, x_values, predictions) in enumerate(profiles):
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=predictions,
                    mode="lines+markers",
                    line=dict(color=palette[index % len(palette)], width=2.5),
                    marker=dict(
                        color=palette[index % len(palette)], size=6
                    ),
                    name=str(label),
                    hovertemplate=(
                        f"{x_factor}: %{{x}}<br>"
                        f"{label}<br>"
                        f"Predicted {response}: %{{y:.3f}}"
                        "<extra></extra>"
                    ),
                )
            )
        self._style_model_plot(
            fig,
            title=f"Interaction - {factor1} : {factor2} - {response}",
            x_title=x_factor,
            y_title=f"Predicted {response}",
        )
        fig.update_layout(
            legend=dict(
                x=1.02, y=1, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
            ),
            margin=dict(l=70, r=165, t=85, b=65),
        )
        return fig

    def _interactions_model(
        self,
        response: str,
        coded: bool = True,
        n_points: int = 50,
    ) -> go.Figure:
        mixture_factors = [
            name for name, factor in self._factors.items()
            if factor.type == "mix"
        ]
        if mixture_factors:
            raise ValueError(
                "Classical interaction plots are not valid for mixture "
                "components. Use plot_mixture_trace() or mixture response "
                "contours/surfaces instead."
            )
        pairs = list(combinations(self._factors, 2))
        if not pairs:
            raise ValueError(
                "At least two process or categorical factors are required."
            )

        fig = go.Figure()
        trace_groups = []
        x_factors = []
        for pair_index, (factor1, factor2) in enumerate(pairs):
            x_factor, profiles = self._interaction_profiles(
                response, factor1, factor2, coded, n_points
            )
            x_factors.append(x_factor)
            trace_ids = []
            palette = ["#003153", "#05A6A6", "#EE6C4D", "#4682B4"]
            for index, (label, x_values, predictions) in enumerate(profiles):
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=predictions,
                        mode="lines+markers",
                        line=dict(
                            color=palette[index % len(palette)], width=2.5
                        ),
                        marker=dict(
                            color=palette[index % len(palette)], size=6
                        ),
                        name=str(label),
                        visible=pair_index == 0,
                        hovertemplate=(
                            f"{x_factor}: %{{x}}<br>"
                            f"{label}<br>"
                            f"Predicted {response}: %{{y:.3f}}"
                            "<extra></extra>"
                        ),
                    )
                )
                trace_ids.append(len(fig.data) - 1)
            trace_groups.append(trace_ids)

        buttons = []
        for pair_index, (factor1, factor2) in enumerate(pairs):
            visible = [False] * len(fig.data)
            for trace_index in trace_groups[pair_index]:
                visible[trace_index] = True
            buttons.append(
                dict(
                    label=f"{factor1} : {factor2}",
                    method="update",
                    args=[
                        {"visible": visible},
                        {
                            "title.text": (
                                f"<b>Interaction - {factor1} : "
                                f"{factor2} - {response}</b>"
                            ),
                            "xaxis.title.text": (
                                f"<b>{x_factors[pair_index]}</b>"
                            ),
                        },
                    ],
                )
            )

        first1, first2 = pairs[0]
        self._style_model_plot(
            fig,
            title=f"Interaction - {first1} : {first2} - {response}",
            x_title=x_factors[0],
            y_title=f"Predicted {response}",
        )
        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0,
                xanchor="left",
                y=1.13,
                yanchor="top",
            )],
            legend=dict(
                x=1.02, y=1, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
            ),
            margin=dict(l=70, r=165, t=95, b=65),
        )
        return fig

    def _mixture_trace_model(
        self,
        response: str,
        reference: Literal["centroid"] = "centroid",
        n_points: int = 50,
    ) -> go.Figure:
        """Render bound-respecting mixture perturbation traces."""
        self._validate_plot_response(response)
        if reference != "centroid":
            raise ValueError("reference currently supports only 'centroid'.")
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        mixture_names = [
            name for name, factor in self._factors.items()
            if factor.type == "mix"
        ]
        if len(mixture_names) < 2:
            raise ValueError(
                "plot_mixture_trace requires at least two mixture components."
            )

        base = self._base_prediction_row()
        reference_values = np.array(
            [base[name] for name in mixture_names], dtype=float
        )
        lower = np.array(
            [self._factors[name].lower_bound for name in mixture_names],
            dtype=float,
        )
        upper = np.array(
            [self._factors[name].upper_bound for name in mixture_names],
            dtype=float,
        )

        palette = ["#003153", "#05A6A6", "#EE6C4D", "#4682B4", "#8E5EA2"]
        fig = go.Figure()
        for component_index, component in enumerate(mixture_names):
            other = np.arange(len(mixture_names)) != component_index
            feasible_min = max(
                lower[component_index],
                1.0 - float(upper[other].sum()),
            )
            feasible_max = min(
                upper[component_index],
                1.0 - float(lower[other].sum()),
            )
            component_values = np.linspace(
                feasible_min, feasible_max, n_points
            )
            rows = []
            for component_value in component_values:
                mixture = reference_values.copy()
                delta = component_value - reference_values[component_index]
                mixture[component_index] = component_value
                if delta >= 0:
                    capacity = reference_values[other] - lower[other]
                    capacity_sum = capacity.sum()
                    if capacity_sum > self.TOLERANCE:
                        mixture[other] -= delta * capacity / capacity_sum
                else:
                    capacity = upper[other] - reference_values[other]
                    capacity_sum = capacity.sum()
                    if capacity_sum > self.TOLERANCE:
                        mixture[other] += (-delta) * capacity / capacity_sum
                mixture = np.clip(mixture, lower, upper)
                correction = 1.0 - mixture.sum()
                if abs(correction) > self.TOLERANCE:
                    adjustable = np.where(other)[0]
                    mixture[adjustable[-1]] += correction

                row = base.copy()
                row.update(dict(zip(mixture_names, mixture)))
                rows.append(row)

            predictions = self.predict(
                pd.DataFrame(rows), [response]
            )[response].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=component_values,
                    y=predictions,
                    mode="lines",
                    line=dict(
                        color=palette[component_index % len(palette)],
                        width=3,
                    ),
                    name=component,
                    customdata=np.full((n_points, 1), component),
                    hovertemplate=(
                        "Component: %{customdata[0]}<br>"
                        "Proportion: %{x:.3f}<br>"
                        f"Predicted {response}: %{{y:.3f}}"
                        "<extra></extra>"
                    ),
                )
            )

        reference_prediction = float(
            self.predict(pd.DataFrame([base]), [response]).iloc[0][response]
        )
        fig.add_hline(
            y=reference_prediction,
            line_dash="dash",
            line_color="#777777",
            line_width=1.5,
            annotation_text="Reference",
            annotation_position="bottom right",
        )
        self._style_model_plot(
            fig,
            title=f"Mixture Trace - {response}",
            x_title="Component Proportion",
            y_title=f"Predicted {response}",
        )
        fig.update_layout(
            legend=dict(
                x=1.02, y=1, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
            ),
            margin=dict(l=70, r=140, t=85, b=65),
        )
        return fig
