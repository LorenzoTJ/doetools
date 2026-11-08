"""Optional high-resolution static export for Plotly figures."""

from importlib import import_module
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

from plotly.graph_objects import Figure


def export_plot(
    fig: Figure,
    destination: str | Path,
    *,
    scale: float = 3,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Export a Plotly figure without changing its layout or display settings.

    Args:
        fig (plotly.graph_objects.Figure): Figure returned by a plotting method.
        destination (str | Path): Output path ending in ``.png``, ``.jpg``,
            ``.jpeg``, ``.webp``, ``.svg``, or ``.pdf`` (case-insensitive).
            The parent directory must already exist.
        scale (float): Positive resolution multiplier. Defaults to 3, producing
            a 2400 x 1500 PNG from an 800 x 500 figure.
        width (int | None): Positive layout width in logical pixels. By default,
            use the figure width, or Plotly's default if no width was specified.
        height (int | None): Positive layout height in logical pixels. By default,
            use the figure height, or Plotly's default if unspecified.

    Raises:
        TypeError: If the figure, destination, or dimensions have invalid types.
        ValueError: If the format, scale, or dimensions are invalid.
        ImportError: If the optional Kaleido dependency is unavailable.
        RuntimeError: If Kaleido cannot find Chrome/Chromium or rendering fails.
        OSError: If the destination cannot be written.

    Notes:
        Install ``doetools[export]`` and Chrome/Chromium for static export.
        SVG/PDF preserve vector elements, but WebGL traces such as 3D surfaces
        contain rasterized content even in vector files. This helper does not
        change the notebook camera-button configuration or Plotly defaults.
    """
    if not isinstance(fig, Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure.")
    if not isinstance(destination, (str, Path)):
        raise TypeError("destination must be a string or Path.")
    path = Path(destination)
    image_format = path.suffix.lower().lstrip(".")
    if image_format not in {"png", "jpg", "jpeg", "webp", "svg", "pdf"}:
        raise ValueError("Supported export formats: .png, .jpg, .jpeg, .webp, .svg, .pdf.")
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("scale must be a positive finite number.")
    if not isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number.")
    for name, value in (("width", width), ("height", height)):
        if value is not None:
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be a positive integer or None.")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    try:
        import_module("kaleido")
    except ModuleNotFoundError as exc:
        if exc.name != "kaleido":
            raise
        raise ImportError(
            'Static plot export requires Kaleido. Install it with: pip install "doetools[export]". '
            "Chrome/Chromium must also be installed."
        ) from exc

    # Let Plotly report missing browsers and rendering/file errors without masking them.
    fig.write_image(
        path,
        format="jpeg" if image_format == "jpg" else image_format,
        scale=scale,
        width=width,
        height=height,
    )
