from typing import Literal
from doetools.graphs.design_plot_builder import DesignPlotOptions, build_design_plot
from doetools.graphs.renderers import Renderer
from .data_builder import DataBuilder
import pandas as pd
import plotly.graph_objects as go
import numpy as np

class GraphsMixin(Renderer, DataBuilder):
    
    def plot_design(self, 
                    ax1 : str,
                    ax2 : str,
                    ax3 : str = None,
                    ax4 : str = None,
                    coded : bool  = False,
                    *,
                    show_title: bool = True,
                    show_summary: bool = True,
                    show_legend: bool = True,
                    show_grid: bool = True,
                    marker_color: str = "#1971c2",
                    show_hover: bool = True,
                    show_run_labels: bool = False,
                    show_replicate_count: bool = True,
                    aggregate_projected_points: bool = True,
                    axis_label_mode: str = "symbol_unit",
                    height: int | None = None,
                    domain: Literal["full", "allowed"] = "full"):
        """Visualize an experimental design in factor space.

        Two or three non-mixture factors produce a Cartesian scatter plot. Two
        mixture components produce a mixture-line plot, three produce a ternary
        plot, and four produce a tetrahedral plot.

        Args:
            ax1 (str): First factor, used as the x-axis or first simplex vertex.
            ax2 (str): Second factor, used as the y-axis or second simplex
                vertex.
            ax3 (str, optional): Third factor, used as the z-axis or third
                simplex vertex.
            ax4 (str, optional): Fourth mixture component for a tetrahedral
                plot.
            coded (bool): Whether to display coded rather than actual units.
            show_title (bool): Whether to display the generated title.
            show_summary (bool): Whether to display the design summary.
            show_legend (bool): Whether to display the legend.
            show_grid (bool): Whether to display grid lines.
            marker_color (str): Plotly-compatible marker color.
            show_hover (bool): Whether to display point information on hover.
            show_run_labels (bool): Whether to label points with run numbers.
            show_replicate_count (bool): Whether to include replicate counts.
            aggregate_projected_points (bool): Whether points with identical
                plotted coordinates are combined when other factors are omitted.
            axis_label_mode (str): Axis-label format. Supported values are
                ``"symbol_unit"``, ``"symbol"``, and ``"name"``.
            height (int, optional): Figure height in pixels.
            domain (str): Mixture domain to draw. ``"full"`` draws the full
                simplex or tetrahedron; ``"allowed"`` draws the domain covered
                by the design points.

        Returns:
            plotly.graph_objects.Figure: Interactive design plot.

        Notes:
            Either all selected factors must be mixture components, or none of
            them may be mixture components. Continuous and categorical factors
            can be combined in a Cartesian plot.
        """
        options = DesignPlotOptions(
            coded=coded,
            marker_color=marker_color,
            show_title=show_title,
            show_summary=show_summary,
            show_legend=show_legend,
            show_grid=show_grid,
            show_hover=show_hover,
            show_run_labels=show_run_labels,
            show_replicate_count=show_replicate_count,
            aggregate_projected_points=aggregate_projected_points,
            axis_label_mode=axis_label_mode,
            height=height,
            domain=domain,
        )
        return build_design_plot(
            design_matrix=self._design_matrix,
            coded_design_matrix=self._coded_design_matrix,
            factors=self._factors,
            design_type=self._design_type,
            axes=[axis for axis in (ax1, ax2, ax3, ax4) if axis is not None],
            options=options,
        )

    def _build_process_surface_domain_for_plot(
        self,
        grid_df_coded: pd.DataFrame,
        grid_df: pd.DataFrame,
        ax1: str,
        ax2: str,
        coded: bool,
        domain: Literal["full", "allowed"],
    ):
        if domain not in {"full", "allowed"}:
            raise ValueError("domain must be either 'full' or 'allowed'.")
        if domain == "full":
            return None

        candidate_attr = "_coded_cp" if coded else "_cp"
        design_attr = "_coded_design_matrix" if coded else "_design_matrix"
        fallback_points = getattr(self, candidate_attr, None)
        if fallback_points is None or len(fallback_points) == 0:
            fallback_points = getattr(self, design_attr, None)

        return self.build_process_surface_domain(
            grid=grid_df,
            x_title=ax1,
            y_title=ax2,
            mode=domain,
            filters=getattr(self, "_domain_filters", []),
            filter_grid=self._decode_matrix(grid_df_coded),
            fallback_points=fallback_points,
        )
    
    def plot_leverage(self,
                        ax1: str,
                        ax2: str,
                        ax3: str = None,
                        constant_levels: dict[str, float] = None,
                        resolution: int = 100,
                        x_min: float = -1.0,
                        x_max: float = 1.0,
                        y_min: float = -1.0,
                        y_max: float = 1.0,
                        coded : bool = False,
                        show_title: bool = True,
                        show_legend: bool = True,
                        show_summary: bool = True,
                        colorscale: str | list | None = None,
                        domain: Literal["full", "allowed"] = "full",
                        ) -> tuple[go.Figure, go.Figure]:
        """Visualize leverage values across the design space.
        
        Leverage measures the influence of design points on model predictions.
        High leverage points are candidates for replication to improve precision.
        
        Args:
            ax1 (str): First free factor.
            ax2 (str): Second free factor.
            ax3 (str, optional): Third free mixture component for a ternary
                plot.
            constant_levels (dict[str, float], optional): Actual-unit levels
                at which to hold all remaining factors.
            resolution (int): Number of grid points per process axis, or the
                mixture-grid resolution.
            x_min (float): Lower bound for ``ax1`` in coded units.
            x_max (float): Upper bound for ``ax1`` in coded units.
            y_min (float): Lower bound for ``ax2`` in coded units.
            y_max (float): Upper bound for ``ax2`` in coded units.
            coded (bool): Whether to display coded rather than actual units.
            show_title (bool): Whether to display the generated title.
            show_legend (bool): Whether to display the legend.
            show_summary (bool): Whether to display the fixed-level and
                leverage-range annotation.
            colorscale (str or list, optional): Plotly colorscale applied to
                the contour and surface traces.
            domain (str): Plotting domain. ``"allowed"`` applies saved process
                domain filters or restricts a mixture plot to its allowed hull.

        Returns:
            tuple[plotly.graph_objects.Figure, plotly.graph_objects.Figure]:
                Contour and three-dimensional surface figures.

        Notes:
            Leverage is computed as ``x.T @ (X.T @ X)^+ @ x``, where ``+``
            denotes the Moore-Penrose pseudoinverse. A model matrix must have
            been defined, but response results and a fitted model are not
            required.
        """
        
        grid_df_coded, coded_constant_levels = self._create_grid(
                                        ax1=ax1,
                                        ax2=ax2,
                                        ax3=ax3,
                                        constant_levels=constant_levels,
                                        resolution=resolution,
                                        x_min=x_min,
                                        x_max=x_max,
                                        y_min=y_min,
                                        y_max=y_max
                                        )

        leverage = self._compute_leverage(grid_df_coded)
    
        if ax3 is None:
            grid_df = self._decode_matrix(grid_df_coded) if not coded else grid_df_coded
            process_surface_domain = self._build_process_surface_domain_for_plot(
                grid_df_coded,
                grid_df,
                ax1,
                ax2,
                coded,
                domain,
            )
            labels = list(grid_df.columns)
            hovertemplate = self.build_hovertemplate(labels, ["Leverage"], precision=3)
            customdata = grid_df.copy()
            customdata['Leverage'] = leverage
            customdata = customdata.to_numpy()
            # Extract constant levels in decoded space
            constant_levels = {}
            for name in coded_constant_levels.keys():
                constant_levels[name] = grid_df[name].iloc[0]
                
            fig_cp = self.render_contour_process(                               
                               grid = grid_df,
                               x_title = ax1,
                               y_title = ax2,
                               z_title="Leverage",
                               response = leverage,
                               constant_levels= constant_levels,
                               hovertemplate = hovertemplate,
                               customdata = customdata,
                               process_surface_domain=process_surface_domain)
            
            fig_surf = self.render_surface_process(
                               grid = grid_df,
                               x_title = ax1,
                               y_title = ax2,
                               z_title="Leverage",
                               response = leverage,
                               constant_levels= constant_levels,
                               hovertemplate = hovertemplate,
                               customdata = customdata,
                               process_surface_domain=process_surface_domain)
            
            return self._configure_leverage_figures(
                fig_cp,
                fig_surf,
                show_title=show_title,
                show_legend=show_legend,
                show_summary=show_summary,
                colorscale=colorscale,
            )
        
        else:
            # Decode grid if needed
            grid_df = self._decode_matrix(grid_df_coded) if not coded else grid_df_coded
            # Extract constant levels in decoded space
            constant_levels = {}
            for name in coded_constant_levels.keys():
                constant_levels[name] = grid_df[name].iloc[0]
            # Scale the grid for mixture plotting
            grid_df_scaled = self.mixture_scaler(grid_df, constant_levels)
            surface_domain = self.build_mixture_surface_domain(
                grid_df_scaled=grid_df_scaled,
                a_title=ax1,
                b_title=ax2,
                c_title=ax3,
                mode=domain,
            )
            interp_leverage = self.interpolate_mixture_surface(
                surface_domain,
                leverage,
            )
            # Create the hovertemplate
            labels = list(grid_df.columns)
            hovertemplate = self.build_hovertemplate(labels, ["Leverage"], precision=3)
            customdata = grid_df.copy()
            customdata["Leverage"] = leverage
            customdata = customdata.to_numpy()
            # Build vertex text
            other_mix = [f for f in constant_levels.keys() if self._factors[f].type == "mix" and f not in [ax1, ax2, ax3]]
            L = np.sum([constant_levels[f] for f in other_mix]) if len(other_mix) > 0 else 0
            vertex_texts = [
            f"{ax1}:{1-L:.2f}<br>{ax2}:0<br>{ax3}:0<br>{other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:1<br>{ax2}:0<br>{ax3}:0",
            f"{ax1}:0<br>{ax2}:{1-L:.2f}<br>{ax3}:0<br>{other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:0<br>{ax2}:1<br>{ax3}:0",
            f"{ax1}:0, {ax2}:0, {ax3}:{1-L:.2f}, {other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:0, {ax2}:0, {ax3}:1"
            ]
            # Create the figure
            fig1 = self.render_contour_mixture(
                                            grid_df_scaled = grid_df_scaled,
                                            grid_df = grid_df,
                                            response = interp_leverage,
                                            a_title = ax1,
                                            b_title = ax2,
                                            c_title = ax3,
                                            z_title = "Leverage",
                                            constant_levels = constant_levels,
                                            hovertemplate = hovertemplate,
                                            customdata= customdata,
                                            grid_step=0.1,
                                            vertex_texts=vertex_texts,
                                            show_grid=True,
                                            show_border=True,
                                            min = 0,
                                            max = 1 - L,
                                            domain=domain,
                                            surface_domain=surface_domain,
                                        )
            fig2 = self.render_surface_mixture(
                                            grid_df_scaled = grid_df_scaled,
                                            grid_df = grid_df,
                                            response = leverage,
                                            a_title = ax1,
                                            b_title = ax2,
                                            c_title = ax3,
                                            z_title = "Leverage",
                                            constant_levels = constant_levels,
                                            hovertemplate = hovertemplate,
                                            customdata= customdata,
                                            min = 0,
                                            max = 1 - L,
                                            domain=domain,
                                            surface_domain=surface_domain,
                                        )
            return self._configure_leverage_figures(
                fig1,
                fig2,
                show_title=show_title,
                show_legend=show_legend,
                show_summary=show_summary,
                colorscale=colorscale,
            )

    @staticmethod
    def _configure_leverage_figures(
        figure_2d: go.Figure,
        figure_3d: go.Figure,
        *,
        show_title: bool,
        show_legend: bool,
        show_summary: bool,
        colorscale,
    ) -> tuple[go.Figure, go.Figure]:
        """Apply optional presentation controls without changing leverage data."""
        for figure in (figure_2d, figure_3d):
            annotations = []
            for annotation in figure.layout.annotations or ():
                text = str(annotation.text or "")
                is_title = "Contour Plot - Leverage" in text or "3D Surface Plot - Leverage" in text
                is_summary = "Max. Leverage:" in text or "Min. Leverage:" in text
                if (is_title and not show_title) or (is_summary and not show_summary):
                    continue
                annotations.append(annotation)
            figure.layout.annotations = tuple(annotations)
            figure.layout.showlegend = show_legend
            if not show_title:
                figure.layout.title = None
            if colorscale is not None:
                for trace in figure.data:
                    if trace.type in {"contour", "surface", "mesh3d"}:
                        trace.colorscale = colorscale
                    if trace.type == "contour":
                        trace.contours.coloring = "lines"
                        trace.line.color = None
        return figure_2d, figure_3d
            

    def plot_response(self,
                        ax1: str,
                        ax2: str,
                        response : str,
                        constant_levels: dict[str, float] = None,
                        second_response: str = None,
                        corrected : Literal["replicates", "residuals"] = None,
                        feasible_region: bool = False,
                        ax3 : str = None,
                        x_min: float = -1.0,
                        x_max: float = 1.0,
                        y_min: float = -1.0,
                        y_max: float = 1.0,
                        resolution : int = 300,
                        alpha : float = 0.05,
                        coded : bool = False,
                        domain: Literal["full", "allowed"] = "full",
                        ):
        """Visualize a fitted response across the design space.

        The surface can include a second response, a conservative confidence
        correction, and the region satisfying configured response conditions.

        Args:
            ax1 (str): First free factor.
            ax2 (str): Second free factor.
            response (str): Primary fitted response.
            constant_levels (dict[str, float], optional): Actual-unit levels
                at which to hold all remaining factors.
            second_response (str, optional): Second fitted response to overlay.
            corrected (str, optional): Confidence correction applied to each
                prediction. ``"replicates"`` uses pure-error variance and
                ``"residuals"`` uses residual mean square. Configured response
                conditions determine the conservative direction.
            feasible_region (bool): Whether to mark the region satisfying the
                response conditions configured with
                :meth:`~doetools.ModelMixin.set_response_conditions`.
            ax3 (str, optional): Third free mixture component for a ternary
                plot.
            x_min (float): Lower bound for ``ax1`` in coded units.
            x_max (float): Upper bound for ``ax1`` in coded units.
            y_min (float): Lower bound for ``ax2`` in coded units.
            y_max (float): Upper bound for ``ax2`` in coded units.
            resolution (int): Number of grid points per process axis, or the
                mixture-grid resolution.
            alpha (float): Significance level used by the confidence
                correction; ``0.05`` gives a 95% confidence level.
            coded (bool): Whether to display coded rather than actual units.
            domain (str): Plotting domain. ``"allowed"`` applies saved process
                domain filters or restricts a mixture plot to its allowed hull.

        Returns:
            tuple[plotly.graph_objects.Figure, plotly.graph_objects.Figure]:
                Contour and three-dimensional surface figures.

        Notes:
            A fitted model is required. Any confidence correction requires
            response conditions; its half-width is subtracted for a response
            being maximized and added for one being minimized. The
            ``"replicates"`` correction also requires replicate-based
            lack-of-fit statistics. ``feasible_region=True`` likewise requires
            response conditions.
        """
        
        #1) Check constant levels and Create the grid
        grid_df_coded, coded_constant_levels = self._create_grid(
                                ax1=ax1,
                                ax2=ax2,
                                ax3=ax3,
                                constant_levels=constant_levels,
                                resolution=resolution,
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max
                                )
        
        #3) Predict response(s)
        responses = [response] if second_response is None else [response, second_response]
        predicted_responses = self.predict(matrix_to_pred=grid_df_coded, responses=responses)

        #4) Correct response if needed
        if corrected is not None:
            corrected_resposes = pd.DataFrame()
            for resp in responses:
                if corrected == "replicates":
                    conf_int = self.calculate_confidence_interval(
                                                grid_of_points=grid_df_coded,
                                                response=resp,
                                                type_of_correction="replicates",
                                                alpha=alpha
                                            )
                elif corrected == "residuals":
                    conf_int = self.calculate_confidence_interval(
                                                grid_of_points=grid_df_coded,
                                                response=resp,
                                                type_of_correction="residuals",
                                                alpha=alpha
                                            )
                corrected_resposes[resp] = self.add_confidence_interval(
                                            predicted_response=predicted_responses[resp],
                                            response=resp,
                                            conf_int=conf_int
                                            )
        if ax3 is not None:
            # Decode grid if needed
            grid_df = self._decode_matrix(grid_df_coded) if not coded else grid_df_coded
            # Extract constant levels in decoded space
            constant_levels = {}
            for name in coded_constant_levels.keys():
                constant_levels[name] = grid_df[name].iloc[0]
            # Scale the grid for mixture plotting
            grid_df_scaled = self.mixture_scaler(grid_df, constant_levels)
            surface_domain = self.build_mixture_surface_domain(
                grid_df_scaled=grid_df_scaled,
                a_title=ax1,
                b_title=ax2,
                c_title=ax3,
                mode=domain,
            )
            interp_responses = {}
            for resp in responses:
                response_values = (
                    predicted_responses[resp].to_numpy()
                    if corrected is None
                    else corrected_resposes[resp].to_numpy()
                )
                interp_responses[resp] = self.interpolate_mixture_surface(
                    surface_domain,
                    response_values,
                )
                
        #5) Add feasible region if needed
        if feasible_region:
            if ax3 is None:
                Z_feasible = self.compute_feasible_region(resp1 = response,
                                                        response1 = corrected_resposes[response] if corrected is not None else predicted_responses[response],
                                                        resp2 = second_response if second_response is not None else None,
                                                        response2 = corrected_resposes[second_response] if (corrected is not None and second_response is not None) else (predicted_responses[second_response] if second_response is not None else None)
                                                        )
                Z_feasible = Z_feasible.reshape((resolution, resolution))
            else:
                Z_feasible = self.compute_feasible_region(resp1 = response,
                                                        response1 = interp_responses[response],
                                                        resp2 = second_response if second_response is not None else None,
                                                        response2 = interp_responses[second_response] if second_response is not None else None
                                                        )                
            
        #6) Plot contour and surface for process
        if ax3 is None:
            grid_df = self._decode_matrix(grid_df_coded) if not coded else grid_df_coded
            process_surface_domain = self._build_process_surface_domain_for_plot(
                grid_df_coded,
                grid_df,
                ax1,
                ax2,
                coded,
                domain,
            )
            labels = list(grid_df.columns)
            hovertemplate = self.build_hovertemplate(labels, responses, precision=3)
            customdata = self.build_customdata(grid_df, predicted_responses if corrected is None else corrected_resposes)
            # Extract constant levels in decoded space
            constant_levels = {}
            for name in coded_constant_levels.keys():
                constant_levels[name] = grid_df[name].iloc[0]
                
            fig_cp = self.render_contour_process(                               
                               grid = grid_df,
                               x_title = ax1,
                               y_title = ax2,
                               z_title=response,
                               response = predicted_responses[response].to_numpy() if corrected is None else corrected_resposes[response].to_numpy(),
                               constant_levels= constant_levels,
                               hovertemplate = hovertemplate,
                               customdata = customdata,
                               second_response = predicted_responses[second_response].to_numpy() if (corrected is None and second_response is not None) else (corrected_resposes[second_response].to_numpy() if (corrected is not None and second_response is not None) else None),
                               z2_title=second_response,
                               feasible_region = Z_feasible if feasible_region else None,
                               process_surface_domain=process_surface_domain)
            
            fig_surf = self.render_surface_process(
                               grid = grid_df,
                               x_title = ax1,
                               y_title = ax2,
                               z_title=response,
                               response = predicted_responses[response].to_numpy() if corrected is None else corrected_resposes[response].to_numpy(),
                               constant_levels= constant_levels,
                               hovertemplate = hovertemplate,
                               customdata = customdata,
                               feasible_region = feasible_region,
                               z2_title=second_response,
                               second_response = predicted_responses[second_response].to_numpy() if (corrected is None and second_response is not None) else (corrected_resposes[second_response].to_numpy() if (corrected is not None and second_response is not None) else None),
                               process_surface_domain=process_surface_domain)

            return fig_cp, fig_surf
                    
        else:
            # Create the hovertemplate
            labels = list(grid_df.columns)
            hovertemplate = self.build_hovertemplate(labels, responses, precision=3)
            # Build vertex text
            other_mix = [f for f in constant_levels.keys() if self._factors[f].type == "mix" and f not in [ax1, ax2, ax3]]
            L = np.sum([constant_levels[f] for f in other_mix]) if len(other_mix) > 0 else 0
            vertex_texts = [
            f"{ax1}:{1-L:.2f}<br>{ax2}:0<br>{ax3}:0<br>{other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:1<br>{ax2}:0<br>{ax3}:0",
            f"{ax1}:0<br>{ax2}:{1-L:.2f}<br>{ax3}:0<br>{other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:0<br>{ax2}:1<br>{ax3}:0",
            f"{ax1}:0, {ax2}:0, {ax3}:{1-L:.2f}, {other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:0, {ax2}:0, {ax3}:1"
            ]  
            # Customdata for hover
            customdata = self.build_customdata(grid_df, predicted_responses if corrected is None else corrected_resposes)
            # Create the figure
            fig1 = self.render_contour_mixture(
                                            grid_df_scaled = grid_df_scaled,
                                            grid_df = grid_df,
                                            response = interp_responses[response],
                                            a_title = ax1,
                                            b_title = ax2,
                                            c_title = ax3,
                                            z_title = response,
                                            constant_levels = constant_levels,
                                            hovertemplate = hovertemplate,
                                            customdata= customdata,
                                            second_response = interp_responses[second_response] if second_response is not None else None,
                                            z2_title = second_response,
                                            grid_step=0.1,
                                            vertex_texts=vertex_texts,
                                            show_grid=True,
                                            show_border=True,
                                            min = 0,
                                            max = 1-L,
                                            feasible_region = Z_feasible if feasible_region else None,
                                            domain=domain,
                                            surface_domain=surface_domain,
                                        )
            fig2 = self.render_surface_mixture(
                                            grid_df_scaled = grid_df_scaled,
                                            grid_df = grid_df,
                                            response = corrected_resposes[response] if corrected is not None else predicted_responses[response],
                                            a_title = ax1,
                                            b_title = ax2,
                                            c_title = ax3,
                                            z_title = response,
                                            constant_levels= constant_levels,
                                            hovertemplate = hovertemplate,
                                            customdata= customdata,
                                            second_response = corrected_resposes[second_response] if (corrected is not None and second_response is not None) else (predicted_responses[second_response] if second_response is not None else None),
                                            z2_title = second_response,
                                            feasible_region= feasible_region,
                                            min = 0,
                                            max = 1-L,
                                            domain=domain,
                                            surface_domain=surface_domain,
                                        )
            return fig1, fig2
        
    def plot_confidence_interval(self,
                        ax1: str,
                        ax2: str,
                        response : str,
                        constant_levels: dict[str, float] = None,
                        type : Literal["replicates", "residuals"] = "residuals",
                        ax3 : str = None,
                        x_min: float = -1.0,
                        x_max: float = 1.0,
                        y_min: float = -1.0,
                        y_max: float = 1.0,
                        resolution : int = 300,
                        alpha : float = 0.05,
                        coded : bool = False,
                        domain: Literal["full", "allowed"] = "full",
                        ):
        """Visualize the confidence half-width for a fitted mean response.

        Narrow values indicate greater precision in the estimated mean. This
        is not a prediction interval for a future observation: the variance
        expression does not include an additional observation-error term.

        Args:
            ax1 (str): First free factor.
            ax2 (str): Second free factor.
            response (str): Fitted response.
            constant_levels (dict[str, float], optional): Actual-unit levels
                at which to hold all remaining factors.
            type (str): Variance estimate. ``"replicates"`` uses pure error;
                ``"residuals"`` uses residual mean square.
            ax3 (str, optional): Third free mixture component for a ternary
                plot.
            x_min (float): Lower bound for ``ax1`` in coded units.
            x_max (float): Upper bound for ``ax1`` in coded units.
            y_min (float): Lower bound for ``ax2`` in coded units.
            y_max (float): Upper bound for ``ax2`` in coded units.
            resolution (int): Number of grid points per process axis, or the
                mixture-grid resolution.
            alpha (float): Significance level; ``0.05`` gives a 95% confidence
                level.
            coded (bool): Whether to display coded rather than actual units.
            domain (str): Plotting domain. ``"allowed"`` applies saved process
                domain filters or restricts a mixture plot to its allowed hull.

        Returns:
            tuple[plotly.graph_objects.Figure, plotly.graph_objects.Figure]:
                Contour and three-dimensional surface figures.

        Notes:
            The plotted half-width is ``t * sqrt(MS * leverage)``. A fitted
            model is required; ``type="replicates"`` additionally requires
            replicate-based lack-of-fit statistics.
        """
        
        #1) Check constant levels and Create the grid
        grid_df_coded, coded_constant_levels = self._create_grid(
                                ax1=ax1,
                                ax2=ax2,
                                ax3=ax3,
                                constant_levels=constant_levels,
                                resolution=resolution,
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max
                                )
        
        if type == "replicates":
            conf_int = self.calculate_confidence_interval(
                                        grid_of_points=grid_df_coded,
                                        response=response,
                                        type_of_correction="replicates",
                                        alpha=alpha
                                    )
            
        elif type == "residuals":
            conf_int = self.calculate_confidence_interval(
                                        grid_of_points=grid_df_coded,
                                        response=response,
                                        type_of_correction="residuals",
                                        alpha=alpha
                                    )
        
        if ax3 is None:
            
            grid_df = self._decode_matrix(grid_df_coded) if not coded else grid_df_coded
            process_surface_domain = self._build_process_surface_domain_for_plot(
                grid_df_coded,
                grid_df,
                ax1,
                ax2,
                coded,
                domain,
            )
            labels = list(grid_df.columns)
            hovertemplate = self.build_hovertemplate(labels, ["Conf. Interval"], precision=3)
            customdata = grid_df.copy()
            customdata['conf_int'] = conf_int
            customdata = customdata.to_numpy()
            # Extract constant levels in decoded space
            constant_levels = {}
            for name in coded_constant_levels.keys():
                constant_levels[name] = grid_df[name].iloc[0]
                
            fig_cp = self.render_contour_process(                               
                               grid = grid_df,
                               x_title = ax1,
                               y_title = ax2,
                               z_title="Conf. Interval",
                               response = conf_int,
                               constant_levels= constant_levels,
                               hovertemplate = hovertemplate,
                               customdata = customdata,
                               process_surface_domain=process_surface_domain)
            
            fig_surf = self.render_surface_process(
                               grid = grid_df,
                               x_title = ax1,
                               y_title = ax2,
                               z_title="Conf. Interval",
                               response = conf_int,
                               constant_levels= constant_levels,
                               hovertemplate = hovertemplate,
                               customdata = customdata,
                               process_surface_domain=process_surface_domain)
            
            return fig_cp, fig_surf
        
        else:
             # Decode grid if needed
            grid_df = self._decode_matrix(grid_df_coded) if not coded else grid_df_coded
            # Extract constant levels in decoded space
            constant_levels = {}
            for name in coded_constant_levels.keys():
                constant_levels[name] = grid_df[name].iloc[0]
            # Scale the grid for mixture plotting
            grid_df_scaled = self.mixture_scaler(grid_df, constant_levels)
            surface_domain = self.build_mixture_surface_domain(
                grid_df_scaled=grid_df_scaled,
                a_title=ax1,
                b_title=ax2,
                c_title=ax3,
                mode=domain,
            )
            interp_conf_int = self.interpolate_mixture_surface(
                surface_domain,
                conf_int,
            )
            # Create the hovertemplate
            labels = list(grid_df.columns)
            hovertemplate = self.build_hovertemplate(labels, ["conf_int"], precision=3)
            # Build vertex text
            other_mix = [f for f in constant_levels.keys() if self._factors[f].type == "mix" and f not in [ax1, ax2, ax3]]
            L = np.sum([constant_levels[f] for f in other_mix]) if len(other_mix) > 0 else 0
            vertex_texts = [
            f"{ax1}:{1-L:.2f}<br>{ax2}:0<br>{ax3}:0<br>{other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:1<br>{ax2}:0<br>{ax3}:0",
            f"{ax1}:0<br>{ax2}:{1-L:.2f}<br>{ax3}:0<br>{other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:0<br>{ax2}:1<br>{ax3}:0",
            f"{ax1}:0, {ax2}:0, {ax3}:{1-L:.2f}, {other_mix[0]}:{L:.2f}" if L != 0 else f"{ax1}:0, {ax2}:0, {ax3}:1"
            ]  
            # Customdata for hover
            customdata = grid_df.copy()
            customdata['conf_int'] = conf_int
            customdata = customdata.to_numpy()
            # Create the figure
            fig1 = self.render_contour_mixture(
                                            grid_df_scaled = grid_df_scaled,
                                            grid_df = grid_df,
                                            response = interp_conf_int,
                                            a_title = ax1,
                                            b_title = ax2,
                                            c_title = ax3,
                                            z_title = "Conf. Interval",
                                            constant_levels = constant_levels,
                                            hovertemplate = hovertemplate,
                                            customdata= customdata,
                                            grid_step=0.1,
                                            vertex_texts=vertex_texts,
                                            show_grid=True,
                                            show_border=True,
                                            min = 0,
                                            max = 1-L,
                                            domain=domain,
                                            surface_domain=surface_domain,
                                        )
            
            fig2 = self.render_surface_mixture(
                                            grid_df_scaled = grid_df_scaled,
                                            grid_df = grid_df,
                                            response = conf_int,
                                            a_title = ax1,
                                            b_title = ax2,
                                            c_title = ax3,
                                            z_title = "Conf. Interval",
                                            constant_levels= constant_levels,
                                            hovertemplate = hovertemplate,
                                            customdata= customdata,
                                            min = 0,
                                            max = 1-L,
                                            domain=domain,
                                            surface_domain=surface_domain,
                                        )
            return fig1, fig2

    
    def plot_regression_coefficients(
        self,
        response: str | None = None,
    ):
        """Plot regression coefficients with confidence intervals.

        Pass ``response`` to obtain a single dashboard-ready figure. If it is
        omitted, the figure keeps the response dropdown for compatibility.
        
        Displays coefficient estimates as bars with error bars representing
        confidence intervals. Coefficients whose intervals do not cross zero
        are statistically significant.
        
        Args:
            response (str, optional): Response to display. If omitted, the
                figure provides a response dropdown.

        Returns:
            plotly.graph_objects.Figure: Interactive coefficient plot.

        Notes:
            A coefficient is significant at the plotted confidence level when
            its interval does not include zero.
        """
        if response is None:
            fig = self.regression_coefficients()
        else:
            fig = self.regression_coefficients_for_response(response)
        return fig
    
    def plot_residuals(
        self,
        response: str | None = None,
        cv: bool = False,
        x_axis: Literal["exp_order", "exp_idx", "sequence"] = "exp_order",
    ):
        """Plot residuals by experimental run.

        Args:
            response (str, optional): Response to display. If omitted, the
                figure provides a response dropdown.
            cv (bool): Whether to use leave-one-out cross-validation residuals
                instead of training residuals.
            x_axis (str): Run coordinate: ``"exp_order"``, ``"exp_idx"``, or
                ``"sequence"``. Experimental order falls back to experiment
                index and then to a one-based sequence when unavailable.

        Returns:
            plotly.graph_objects.Figure: Residual-by-run plot.

        Notes:
            Use :meth:`plot_residuals_vs_fitted` to inspect curvature and
            changing variance against fitted values.
        """
        if isinstance(response, bool):
            cv = response
            response = None
        fig = self.residuals_dashboard(
            response=response,
            cv=cv,
            x_axis=x_axis,
        )
        return fig
    
    def plot_exp_vs_pred(
        self,
        response: str | None = None,
        cv: bool = False,
    ):
        """Plot experimental values against model predictions.

        Args:
            response (str, optional): Response to display. If omitted, the
                figure provides a response dropdown.
            cv (bool): Whether to use leave-one-out cross-validation
                predictions instead of training predictions.

        Returns:
            plotly.graph_objects.Figure: Observed-versus-predicted scatter plot
            with a one-to-one reference line.

        Notes:
            Points should cluster around the one-to-one line. Systematic
            departures may indicate model bias or inadequacy.
        """
        if isinstance(response, bool):
            cv = response
            response = None
        if response is None:
            fig = self.exp_vs_pred(cv=cv)
        else:
            fig = self.exp_vs_pred_for_response(response, cv=cv)
        return fig

    def plot_confirmation_exp_vs_pred(self, response: str) -> go.Figure:
        """Plot grouped confirmation means against OLS predictions.

        One point is shown for each distinct confirmation setting. The
        experimental coordinate is the mean of its confirmation replicates.

        Args:
            response (str): Fitted response to plot.

        Returns:
            plotly.graph_objects.Figure: Observed-versus-predicted confirmation
                plot.

        Raises:
            ValueError: If the response, fitted model, or confirmation data are
                unavailable.
        """
        return self.confirmation_exp_vs_pred_for_response(response)

    def plot_confirmation_residuals(self, response: str) -> go.Figure:
        """Plot confirmation residuals against grouped observed means.

        Residuals follow the library convention ``observed - predicted``.

        Args:
            response (str): Fitted response to plot.

        Returns:
            plotly.graph_objects.Figure: Confirmation residual plot.

        Raises:
            ValueError: If the response, fitted model, or confirmation data are
                unavailable.
        """
        return self.confirmation_residuals_for_response(response)
    
    def plot_model_diagnostics(
        self,
        response: str | None = None,
        cv: bool = False,
    ):
        """Create a four-panel model-diagnostics figure.

        The panels show observed versus predicted values, residuals versus
        fitted values, a normal Q-Q plot, and a residual histogram.

        Args:
            response (str, optional): Response to display. If omitted, the
                figure provides a response dropdown.
            cv (bool): Whether to use leave-one-out cross-validation
                predictions and residuals instead of training values.

        Returns:
            plotly.graph_objects.Figure: Four-panel diagnostic figure.
        """
        if isinstance(response, bool):
            cv = response
            response = None
        if response is None:
            fig = self.model_diagnostics(cv=cv)
        else:
            fig = self.model_diagnostics_for_response(response, cv=cv)
        return fig

    def plot_residuals_vs_fitted(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        """Plot residuals against fitted values for one response.

        Args:
            response (str): Fitted response to display.
            cv (bool): Whether to use leave-one-out cross-validation values.

        Returns:
            plotly.graph_objects.Figure: Residuals-versus-fitted plot.
        """
        return self.residuals_vs_fitted_for_response(response, cv=cv)

    def plot_residuals_by_run(
        self,
        response: str,
        cv: bool = False,
        x_axis: Literal["exp_order", "exp_idx", "sequence"] = "exp_order",
    ) -> go.Figure:
        """Plot residuals by experimental run for one response.

        Args:
            response (str): Fitted response to display.
            cv (bool): Whether to use leave-one-out cross-validation residuals.
            x_axis (str): Run coordinate: ``"exp_order"``, ``"exp_idx"``, or
                ``"sequence"``. Experimental order falls back to experiment
                index and then to a one-based sequence when unavailable.

        Returns:
            plotly.graph_objects.Figure: Residual-by-run plot.
        """
        return self.residuals_by_run_for_response(
            response,
            cv=cv,
            x_axis=x_axis,
        )

    def plot_qq_residuals(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        """Plot normal Q-Q residual diagnostics for one response.

        Args:
            response (str): Fitted response to display.
            cv (bool): Whether to use leave-one-out cross-validation residuals.

        Returns:
            plotly.graph_objects.Figure: Normal Q-Q plot.
        """
        return self.qq_residuals_for_response(response, cv=cv)

    def plot_residuals_histogram(
        self,
        response: str,
        cv: bool = False,
    ) -> go.Figure:
        """Plot the residual distribution for one response.

        Args:
            response (str): Fitted response to display.
            cv (bool): Whether to use leave-one-out cross-validation residuals.

        Returns:
            plotly.graph_objects.Figure: Residual histogram.
        """
        return self.residuals_histogram_for_response(response, cv=cv)
        
    
    def plot_main_effects(self, 
                            response: str,
                            coded: bool = True,
                            n_points: int = 50,
                            ) -> go.Figure:
        """Plot fitted main-effect profiles for all non-mixture factors.

        Each factor varies across its range while all other process factors are
        held at their centers and categorical factors at their reference levels.

        Args:
            response (str): Fitted response to display.
            coded (bool): Whether to display coded rather than actual units.
            n_points (int): Number of evaluation points for each continuous
                factor profile.

        Returns:
            plotly.graph_objects.Figure: Main-effect dashboard with one panel
                per factor.

        Raises:
            ValueError: If the design contains mixture components. Use
                :meth:`plot_mixture_trace` for mixture models.
        """
        fig = self.main_effects_model(
            response=response,
            coded=coded,
            n_points=n_points,
        )
        return fig

    def plot_main_effect(
        self,
        response: str,
        factor: str,
        coded: bool = True,
        n_points: int = 50,
    ) -> go.Figure:
        """Plot the fitted main-effect profile for one non-mixture factor.

        Args:
            response (str): Fitted response to display.
            factor (str): Process or categorical factor to vary.
            coded (bool): Whether to display coded rather than actual units.
            n_points (int): Number of evaluation points for a continuous
                factor profile.

        Returns:
            plotly.graph_objects.Figure: Main-effect profile.

        Raises:
            ValueError: If ``factor`` is a mixture component or is unavailable.
        """
        return self.main_effect_model(
            response=response,
            factor=factor,
            coded=coded,
            n_points=n_points,
        )
    
    def plot_interactions(self, 
                            response: str,
                            coded: bool = True,
                            n_points: int = 50,
                            ) -> go.Figure:
        """Plot fitted interactions for all non-mixture factor pairs.

        Args:
            response (str): Fitted response to display.
            coded (bool): Whether to display coded rather than actual units.
            n_points (int): Number of evaluation points along the factor shown
                on each horizontal axis.

        Returns:
            plotly.graph_objects.Figure: Interaction figure with a factor-pair
                selector.

        Raises:
            ValueError: If the design contains mixture components. Use
                :meth:`plot_mixture_trace` for mixture models.

        Notes:
            Non-parallel profiles indicate that one factor's fitted effect
            depends on the level of the other factor.
        """
        fig = self.interactions_model(
            response=response,
            coded=coded,
            n_points=n_points,
        )
        return fig

    def plot_interaction(
        self,
        response: str,
        factor1: str,
        factor2: str,
        coded: bool = True,
        n_points: int = 50,
    ) -> go.Figure:
        """Plot one fitted two-factor interaction profile.

        Args:
            response (str): Fitted response to display.
            factor1 (str): Factor varied along the horizontal axis.
            factor2 (str): Factor represented by separate profiles.
            coded (bool): Whether to display coded rather than actual units.
            n_points (int): Number of evaluation points along ``factor1``.

        Returns:
            plotly.graph_objects.Figure: Two-factor interaction plot.

        Raises:
            ValueError: If either factor is a mixture component, the factors
                are identical, or either factor is unavailable.
        """
        return self.interaction_model(
            response=response,
            factor1=factor1,
            factor2=factor2,
            coded=coded,
            n_points=n_points,
        )

    def plot_mixture_trace(
        self,
        response: str,
        reference: Literal["centroid"] = "centroid",
        n_points: int = 50,
    ) -> go.Figure:
        """Plot fitted mixture traces through a centroid reference blend.

        Each component is varied in turn while the other mixture components are
        adjusted using their available capacity. Component bounds and the
        mixture sum are preserved. Process and categorical factors remain at
        their standard reference levels.

        Args:
            response (str): Fitted response to display.
            reference (str): Reference blend. Currently only ``"centroid"`` is
                supported.
            n_points (int): Number of points on each feasible component path.

        Returns:
            plotly.graph_objects.Figure: Mixture trace plot.

        Raises:
            ValueError: If fewer than two mixture components are available, the
                reference is unsupported, or ``n_points`` is less than two.
        """
        return self.mixture_trace_model(
            response=response,
            reference=reference,
            n_points=n_points,
        )
