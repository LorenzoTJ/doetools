"""Export options and optional dependencies must not affect interactive figures."""

from copy import deepcopy

import plotly.graph_objects as go
import pytest

from doetools import export_plot
from doetools.utils import plot_export


@pytest.fixture
def renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(plot_export, "import_module", lambda name: object())
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *args, **kwargs: calls.append((args, kwargs)))
    return calls


@pytest.mark.parametrize("extension", ["png", "jpg", "jpeg", "webp", "svg", "pdf", "PNG", "SVG"])
@pytest.mark.parametrize("string_path", [False, True])
def test_export_preserves_figure_and_dimensions(renderer, tmp_path, extension, string_path):
    fig = go.Figure(go.Scatter(x=[1, 2], y=[2, 1]))
    fig.update_layout(width=800, height=500)
    original = deepcopy(fig.to_dict())
    path = tmp_path / f"plot.{extension}"
    assert export_plot(fig, str(path) if string_path else path) is None
    assert fig.to_dict() == original
    expected_format = "jpeg" if extension == "jpg" else extension.lower()
    assert renderer == [((path,), dict(format=expected_format, scale=3, width=None, height=None))]


def test_custom_export_dimensions(renderer, tmp_path):
    export_plot(go.Figure(), tmp_path / "plot.png", scale=4, width=1000, height=650)
    assert renderer[0][1] == dict(format="png", scale=4, width=1000, height=650)


@pytest.mark.parametrize("options,error", [
    ({"scale": 0}, ValueError), ({"scale": -1}, ValueError),
    ({"scale": float("nan")}, ValueError), ({"scale": float("inf")}, ValueError),
    ({"scale": True}, TypeError), ({"scale": "3"}, TypeError),
    ({"width": 0}, ValueError), ({"height": -1}, ValueError),
    ({"width": 1.5}, TypeError), ({"height": True}, TypeError),
])
def test_invalid_options_fail_before_rendering(renderer, tmp_path, options, error):
    with pytest.raises(error):
        export_plot(go.Figure(), tmp_path / "plot.png", **options)
    assert renderer == []


@pytest.mark.parametrize("filename", ["plot", "plot.html", "plot.eps", "plot.gif"])
def test_unsupported_extension(renderer, tmp_path, filename):
    with pytest.raises(ValueError, match="Supported export formats"):
        export_plot(go.Figure(), tmp_path / filename)
    assert renderer == []


def test_invalid_figure_and_destination(renderer):
    with pytest.raises(TypeError, match="fig must be"):
        export_plot({}, "plot.png")
    with pytest.raises(TypeError, match="destination must be"):
        export_plot(go.Figure(), 123)
    assert renderer == []


def test_missing_kaleido_has_installation_instructions(monkeypatch, tmp_path):
    def unavailable(name):
        raise ModuleNotFoundError("No module named kaleido", name="kaleido")
    monkeypatch.setattr(plot_export, "import_module", unavailable)
    with pytest.raises(ImportError, match=r'pip install "doetools\[export\]"'):
        export_plot(go.Figure(), tmp_path / "plot.png")


@pytest.mark.parametrize("error", [RuntimeError("Chrome is required: plotly_get_chrome"),
                                   OSError("destination is not writable")])
def test_rendering_errors_are_preserved(monkeypatch, renderer, tmp_path, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(go.Figure, "write_image", fail)
    with pytest.raises(type(error)) as raised:
        export_plot(go.Figure(), tmp_path / "plot.png")
    assert raised.value is error
