import numpy as np
import pandas as pd

from doetools.graphs.data_builder import DataBuilder
from doetools import FullFactorialDesign
from doetools.utils.factors import ContinuousFactor, MixtureFactor
from doetools.utils.model_spec import ModelTerms


class MixtureGridFixture(DataBuilder):
    def __init__(self):
        self._factors = {
            name: MixtureFactor(lower_bound=0.0, upper_bound=1.0)
            for name in ("A", "B", "C")
        }
        self._coded_design_matrix = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            columns=["A", "B", "C"],
        )

    def _code_dict(self, values):
        return values or {}

    def _check_constant_levels(self, matrix, ax1, ax2, ax3, values):
        return values or {}


def test_mixture_grid_respects_requested_resolution():
    fixture = MixtureGridFixture()
    low, _ = fixture._create_grid("A", "B", "C", {}, resolution=4)
    high, _ = fixture._create_grid("A", "B", "C", {}, resolution=8)

    assert len(low) == 15
    assert len(high) == 45
    assert np.allclose(low[["A", "B", "C"]].sum(axis=1), 1.0)


def _process_design():
    design = FullFactorialDesign(
        {
            "A": ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10),
            "B": ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=3),
            "C": ContinuousFactor(n_levels=2, lower_bound=5, upper_bound=15),
        }
    )
    design.set_model_terms(
        ModelTerms(
            intercept=True,
            pro_main="all",
            pro_int2=None,
            pro_quadratic=None,
        )
    )
    return design


def test_plot_leverage_presentation_options_are_backward_compatible():
    design = _process_design()
    default_2d, default_3d = design.plot_leverage("A", "B", resolution=6)
    assert default_2d.layout.showlegend is not False
    assert any("Contour Plot - Leverage" in str(item.text) for item in default_2d.layout.annotations)

    compact_2d, compact_3d = design.plot_leverage(
        "A",
        "B",
        resolution=6,
        show_title=False,
        show_legend=False,
        show_summary=False,
        colorscale="Plasma",
    )
    for figure in (compact_2d, compact_3d):
        assert figure.layout.showlegend is False
        assert not any("Max. Leverage:" in str(item.text) for item in figure.layout.annotations)
    assert not any("Contour Plot - Leverage" in str(item.text) for item in compact_2d.layout.annotations)
    assert compact_3d.data[0].colorscale is not None
