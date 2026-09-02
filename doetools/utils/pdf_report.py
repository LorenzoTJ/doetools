"""Reliable PDF reporting for Design of Experiments analyses."""

from __future__ import annotations

import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable  # noqa: UP035

import numpy as np
import pandas as pd
from fpdf import FPDF, FontFace
from fpdf.enums import MethodReturnValue, XPos, YPos


@dataclass(frozen=True)
class _OptionalSection:
    """Data or an explanatory notice for a non-mandatory report section."""

    data: pd.DataFrame | None = None
    notice: str | None = None
    applicable: bool = True


@dataclass(frozen=True)
class _ResponseSnapshot:
    coefficients: pd.DataFrame
    model_f_test: _OptionalSection
    lof_f_test: _OptionalSection
    metrics: _OptionalSection
    anova: _OptionalSection
    replicates: _OptionalSection


@dataclass(frozen=True)
class _ReportSnapshot:
    design_summary: pd.DataFrame
    factor_summary: pd.DataFrame
    design_matrix: pd.DataFrame
    model_matrix: pd.DataFrame
    model_terms: pd.DataFrame
    responses: pd.DataFrame
    predictions: _OptionalSection
    vif: _OptionalSection
    leverages: _OptionalSection
    response_conditions: _OptionalSection
    response_analysis: dict[object, _ResponseSnapshot]


class DesignReportPDF(FPDF):
    """Generate a sectioned, robust PDF report from public design getters."""

    NAVY = (28, 49, 76)
    BLUE = (42, 111, 151)
    LIGHT_BLUE = (232, 242, 248)
    LIGHT_GRAY = (244, 246, 248)
    TABLE_BORDER = (190, 213, 228)
    TABLE_FONT_SIZE = 8
    MID_GRAY = (105, 115, 125)
    DARK = (35, 39, 43)
    GREEN = (38, 122, 75)
    LIGHT_GREEN = (230, 245, 235)
    AMBER = (174, 111, 0)
    LIGHT_AMBER = (255, 245, 218)
    RED = (176, 48, 48)
    LIGHT_RED = (253, 232, 232)

    def __init__(self, design, title: str = "Design and Model Summary Report"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.design = design
        self.title = title
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)
        self._report_notices: list[str] = []

    # ------------------------------------------------------------------
    # Text, values, and low-level layout
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_text(value) -> str:
        """Normalize arbitrary values for FPDF core fonts."""
        if value is None or value is pd.NA:
            return "N/A"
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if value else "No"
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            return ", ".join(DesignReportPDF._safe_text(item) for item in value)
        try:
            if bool(pd.isna(value)):
                return "N/A"
        except (TypeError, ValueError):
            pass
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "N/A" if np.isnan(value) else ("inf" if value > 0 else "-inf")
            magnitude = abs(float(value))
            if magnitude != 0 and (magnitude >= 10000 or magnitude < 0.001):
                return f"{float(value):.3g}"
            return f"{float(value):.4f}".rstrip("0").rstrip(".")
        if isinstance(value, (int, np.integer)):
            return str(int(value))

        text = str(value)
        replacements = {
            "×": "x", "²": "^2", "³": "^3", "≤": "<=", "≥": ">=",
            "−": "-", "–": "-", "—": "-", "∞": "inf",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        text = unicodedata.normalize("NFKD", text)
        return text.encode("latin-1", "replace").decode("latin-1")

    @classmethod
    def _display_dataframe(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Report sections must be pandas DataFrames.")
        result = dataframe.copy(deep=True)
        result.columns = [cls._safe_text(column) for column in result.columns]
        return result.map(cls._safe_text)

    @property
    def _usable_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def _ensure_space(self, height: float) -> None:
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page(orientation=self.cur_orientation)

    def header(self):
        if self.page_no() == 1:
            self.set_fill_color(*self.NAVY)
            self.rect(0, 0, self.w, 14, style="F")
            self.set_xy(self.l_margin, 2)
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(255, 255, 255)
            self.cell(self._usable_width, 10, self._safe_text(self.title), align="L")
        self.set_text_color(*self.DARK)

    def footer(self):
        self.set_y(-11)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.MID_GRAY)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(*self.DARK)

    def add_title(self, text: str, size: int = 15, color: str = "BLACK"):
        """Backward-compatible section-title helper."""
        self.add_section_title(text, color=self.RED if color.upper() == "RED" else self.NAVY, size=size)

    def add_subtitle(self, text: str, size: int = 11):
        """Backward-compatible subsection helper."""
        self.add_subsection(text, size=size)

    def add_paragraph(self, text: str, size: int = 9):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", size)
        self.set_text_color(*self.DARK)
        self.multi_cell(self._usable_width, 4.7, self._safe_text(text))

    def add_separator(self, height: float = 0.4):
        self.set_x(self.l_margin)
        self.set_fill_color(215, 220, 225)
        self.cell(self._usable_width, height, "", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def add_section_title(self, text: str, *, color=None, size: int = 15):
        self._ensure_space(14)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*(color or self.NAVY))
        self.cell(self._usable_width, 9, self._safe_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*(color or self.BLUE))
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(*self.DARK)

    def add_appendix_title(self, suffix: str):
        """Add an appendix heading while keeping the appendix label prominent."""
        self._ensure_space(14)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*self.NAVY)
        self.write(9, "Appendix A")
        self.set_font("Helvetica", "", 15)
        self.write(9, self._safe_text(suffix))
        self.ln(9)
        self.set_draw_color(*self.BLUE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(*self.DARK)

    def add_subsection(self, text: str, *, size: int = 11):
        self._ensure_space(10)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*self.NAVY)
        self.cell(self._usable_width, 7, self._safe_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*self.DARK)

    def add_notice(self, title: str, message: str, *, kind: str = "info"):
        """Add a semantically coloured, centred information box."""
        palettes = {
            "success": (self.GREEN, self.LIGHT_GREEN),
            "warning": (self.AMBER, self.LIGHT_AMBER),
            "danger": (self.RED, self.LIGHT_RED),
            "info": (self.BLUE, self.LIGHT_BLUE),
            "neutral": (self.MID_GRAY, self.LIGHT_GRAY),
        }
        foreground, background = palettes.get(kind, palettes["info"])
        title_text = self._safe_text(title)
        message_text = self._safe_text(message)
        width = self._usable_width
        self.set_font("Helvetica", "B", 9)
        title_lines = self.multi_cell(
            width - 6, 4.2, title_text, max_line_height=4.2,
            dry_run=True, output=MethodReturnValue.LINES,
        )
        self.set_font("Helvetica", "", 8)
        message_lines = self.multi_cell(
            width - 6, 3.8, message_text, max_line_height=3.8,
            dry_run=True, output=MethodReturnValue.LINES,
        )
        content_height = 4.2 * len(title_lines) + 1.5 + 3.8 * len(message_lines)
        height = max(16.0, content_height + 5.0)
        self._ensure_space(height + 7)
        self.ln(4)
        x, y = self.l_margin, self.get_y()
        self.set_fill_color(*background)
        self.set_draw_color(*foreground)
        self.rect(x, y, width, height, style="DF")
        content_y = y + (height - content_height) / 2
        self.set_xy(x + 3, content_y)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*foreground)
        self.multi_cell(width - 6, 4.2, title_text, max_line_height=4.2, align="L")
        self.set_x(x + 3)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.DARK)
        self.multi_cell(width - 6, 3.8, message_text, max_line_height=3.8, align="L")
        self.set_y(y + height + 3)

    def add_kpi_cards(self, values: dict[str, object], columns: int = 3):
        items = list(values.items())
        gap, height = 3.0, 20.0
        width = (self._usable_width - gap * (columns - 1)) / columns
        for start in range(0, len(items), columns):
            self._ensure_space(height + 4)
            row = items[start:start + columns]
            y = self.get_y()
            for offset, (label, value) in enumerate(row):
                x = self.l_margin + offset * (width + gap)
                self.set_fill_color(*self.LIGHT_BLUE)
                self.set_draw_color(205, 220, 230)
                self.rect(x, y, width, height, style="DF")
                self.set_xy(x + 3, y + 3)
                self.set_font("Helvetica", "", 7.5)
                self.set_text_color(*self.MID_GRAY)
                self.cell(width - 6, 4, self._safe_text(label).upper())
                self.set_xy(x + 3, y + 9)
                self.set_font("Helvetica", "B", 12)
                self.set_text_color(*self.NAVY)
                self.cell(width - 6, 6, self._safe_text(value))
            self.set_y(y + height + 4)
        self.set_text_color(*self.DARK)

    @staticmethod
    def _overview_card_values(summary: dict[str, object]) -> dict[str, object]:
        """Return structural cards and only factor-type cards that are present."""
        cards = {
            label: summary.get(label, "N/A")
            for label in ("Design", "Runs", "Replicates", "Center Points")
            if label in summary
        }
        for label in ("Continuous", "Categorical", "Mixture Components"):
            value = summary.get(label, 0)
            try:
                present = float(value) > 0
            except (TypeError, ValueError):
                present = False
            if present:
                cards[label] = value
        return cards

    # ------------------------------------------------------------------
    # Tables and matrix pagination
    # ------------------------------------------------------------------
    @classmethod
    def _column_weights(cls, dataframe: pd.DataFrame) -> list[float]:
        weights: list[float] = []
        for column in dataframe.columns:
            values = [cls._safe_text(column)] + [cls._safe_text(value) for value in dataframe[column].head(100)]
            longest = max((len(value) for value in values), default=8)
            weights.append(float(min(max(longest, 7), 32)))
        return weights

    def add_table_from_df(
        self,
        dataframe: pd.DataFrame,
        tot_width: float | None = None,
        col_widths: list[float] | None = None,
        heading_style: dict | None = None,
        row_color_mode: str = "ROWS",
        text_alignment: str | list[str] = "CENTER",
        borders_layout: str = "MINIMAL",
        font_size: float | None = None,
    ):
        """Render a normalized table using the report-wide fixed palette."""
        data = self._display_dataframe(dataframe)
        if data.empty or data.shape[1] == 0:
            self.add_notice("Not applicable", "No data are available for this section.", kind="neutral")
            return
        width = min(float(tot_width or self._usable_width), self._usable_width)
        if col_widths is None or len(col_widths) != data.shape[1]:
            col_widths = self._column_weights(data)
        self._ensure_space(8)
        self.ln(2.5)
        # ``font_size`` is retained for compatibility; the report deliberately
        # uses one type scale for every table, including the appendix.
        self.set_font("Helvetica", "", self.TABLE_FONT_SIZE)
        self.set_text_color(*self.DARK)
        self.set_draw_color(*self.TABLE_BORDER)
        # These arguments are retained for compatibility.  All tables use the
        # same visual grammar, regardless of their analytical section.
        style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=self.NAVY)
        with self.table(
            borders_layout=borders_layout,
            cell_fill_mode="NONE",
            col_widths=col_widths,
            headings_style=style,
            line_height=max(self.font_size * 1.8, 4.3),
            repeat_headings=1,
            text_align=text_alignment,
            width=width,
        ) as table:
            heading = table.row()
            for column in data.columns:
                heading.cell(column)
            for index, values in enumerate(data.itertuples(index=False, name=None)):
                row = table.row(style=FontFace(
                    fill_color=self.LIGHT_BLUE if index % 2 == 0 else self.LIGHT_GRAY
                ))
                for value in values:
                    row.cell(value)
        self.set_x(self.l_margin)

    def to_multicol_df(self, dataframe: pd.DataFrame, max_rows: int = 10, runs: bool = True) -> pd.DataFrame:
        """Retain the legacy vertical-to-horizontal table helper."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        data = dataframe.reset_index(drop=True).copy(deep=True)
        if data.empty:
            return data
        blocks = []
        for start in range(0, len(data), max_rows):
            block = data.iloc[start:start + max_rows].copy().reset_index(drop=True)
            if runs:
                block.insert(0, "Run", [str(index + 1) for index in range(start, start + len(block))])
            blocks.append(block)
        height = max(len(block) for block in blocks)
        for index, block in enumerate(blocks):
            if len(block) < height:
                padding = pd.DataFrame("", index=range(height - len(block)), columns=block.columns)
                blocks[index] = pd.concat([block, padding], ignore_index=True)
        return pd.concat(blocks, axis=1)

    @classmethod
    def _split_design_matrix(cls, dataframe: pd.DataFrame, max_width: float) -> list[pd.DataFrame]:
        """Split wide matrices into stable column groups while repeating Run."""
        if "Run" not in dataframe.columns:
            raise ValueError("Design matrix appendix requires a Run column.")
        columns = [column for column in dataframe.columns if column != "Run"]
        if not columns:
            return [dataframe[["Run"]].copy()]

        def desired_width(column) -> float:
            strings = [cls._safe_text(column)] + [cls._safe_text(value) for value in dataframe[column].head(200)]
            longest = max((len(value) for value in strings), default=8)
            return min(max(18.0, longest * 2.0 + 6.0), 55.0)

        groups: list[list[str]] = []
        current: list[str] = []
        current_width = 16.0
        for column in columns:
            width = desired_width(column)
            if current and current_width + width > max_width:
                groups.append(current)
                current = []
                current_width = 16.0
            current.append(column)
            current_width += width
        if current:
            groups.append(current)
        return [dataframe[["Run", *group]].copy(deep=True) for group in groups]

    # ------------------------------------------------------------------
    # Getter snapshot and diagnostics
    # ------------------------------------------------------------------
    def _capture_optional(
        self,
        label: str,
        provider: Callable[[], pd.DataFrame],
        *,
        empty_notice: str = "No applicable data are available.",
        missing_is_expected: bool = False,
    ) -> _OptionalSection:
        try:
            data = provider()
        except Exception as error:  # isolate getter/data failures before rendering
            if missing_is_expected:
                return _OptionalSection(notice=empty_notice, applicable=False)
            notice = f"{label}: {error}"
            self._report_notices.append(notice)
            return _OptionalSection(notice=str(error), applicable=True)
        if not isinstance(data, pd.DataFrame):
            notice = f"{label}: getter did not return a DataFrame"
            self._report_notices.append(notice)
            return _OptionalSection(notice="Getter did not return a DataFrame.", applicable=True)
        if data.empty:
            return _OptionalSection(data=data.copy(deep=True), notice=empty_notice, applicable=False)
        return _OptionalSection(data=data.copy(deep=True))

    def _create_snapshot(self) -> _ReportSnapshot:
        """Read and validate all report data before creating a page."""
        design_summary = self.design.get_design_summary()
        factor_summary = self.design.get_factor_summary()
        design_matrix = self.design.get_design_matrix()
        model_matrix = self.design.get_model_matrix()
        model_terms = self.design.get_model_term_summary()
        responses = self.design.get_responses()

        required = {
            "design summary": design_summary,
            "factor summary": factor_summary,
            "design matrix": design_matrix,
            "model matrix": model_matrix,
            "model terms": model_terms,
            "responses": responses,
        }
        for label, dataframe in required.items():
            if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
                raise ValueError(f"The {label} is unavailable or empty.")
        if len(design_matrix) != len(model_matrix) or len(design_matrix) != len(responses):
            raise ValueError("Design matrix, model matrix, and responses must have the same number of rows.")

        response_analysis: dict[object, _ResponseSnapshot] = {}
        for response in responses.columns:
            coefficients = self.design.get_coefficient_summary(response)
            if not isinstance(coefficients, pd.DataFrame) or coefficients.empty:
                raise ValueError(f"No fitted coefficient summary is available for response {response!r}.")
            response_analysis[response] = _ResponseSnapshot(
                coefficients=coefficients.copy(deep=True),
                model_f_test=self._capture_optional(
                    f"Model F-test ({response})", lambda response=response: self.design.get_model_f_test(response)
                ),
                lof_f_test=self._capture_optional(
                    f"Lack-of-fit F-test ({response})", lambda response=response: self.design.get_lof_f_test(response),
                    empty_notice="No replicated-run lack-of-fit test is applicable.",
                ),
                metrics=self._capture_optional(
                    f"Model metrics ({response})", lambda response=response: self.design.get_metric_summary(response)
                ),
                anova=self._capture_optional(
                    f"ANOVA ({response})", lambda response=response: self.design.get_anova_summary(response)
                ),
                replicates=self._capture_optional(
                    f"Replicate summary ({response})", lambda response=response: self.design.get_replicate_summary(response),
                    empty_notice="No replicated runs are present.",
                ),
            )

        return _ReportSnapshot(
            design_summary=design_summary.copy(deep=True),
            factor_summary=factor_summary.copy(deep=True),
            design_matrix=design_matrix.copy(deep=True),
            model_matrix=model_matrix.copy(deep=True),
            model_terms=model_terms.copy(deep=True),
            responses=responses.copy(deep=True),
            predictions=self._capture_optional("Predicted responses", self.design.get_predicted_responses),
            vif=self._capture_optional("VIF", self.design.get_vif),
            leverages=self._capture_optional("Leverages", self.design.get_leverages),
            response_conditions=self._capture_optional(
                "Response conditions", self.design.get_response_condition_summary,
                empty_notice="Response conditions are not defined.",
                missing_is_expected=True,
            ),
            response_analysis=response_analysis,
        )

    @staticmethod
    def _model_matrix_diagnostics(model_matrix: pd.DataFrame) -> dict[str, object]:
        values = model_matrix.to_numpy(dtype=float)
        runs, columns = model_matrix.shape
        rank = int(np.linalg.matrix_rank(values))
        # Match statsmodels' structural definition: a constant is present when
        # the all-ones vector belongs to the column space. This also detects
        # Scheffe mixture models where sum(component columns) == 1.
        augmented = np.column_stack([values, np.ones(runs)])
        intercept = int(np.linalg.matrix_rank(augmented)) == rank
        residual_df = runs - rank
        statuses = ["Full rank" if rank == columns else "Rank deficient"]
        if residual_df == 0:
            statuses.append("Saturated")
        return {
            "Observations": runs,
            "Columns": columns,
            "Rank": rank,
            "Aliased columns": columns - rank,
            "Intercept": "Yes" if intercept else "No",
            "Regression DF": rank - int(intercept),
            "Residual DF": residual_df,
            "Status": ", ".join(statuses),
        }

    @staticmethod
    def _anova_df(anova: pd.DataFrame | None, source: str) -> float | None:
        if anova is None or anova.empty or "Source" not in anova.columns or "df" not in anova.columns:
            return None
        row = anova.loc[anova["Source"] == source, "df"]
        if row.empty:
            return None
        try:
            return float(row.iloc[0])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _f_value(table: pd.DataFrame | None, column: str) -> float | None:
        if table is None or table.empty or column not in table.columns:
            return None
        try:
            return float(table.iloc[0][column])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_p_value(value: float) -> str:
        if value < 0.0001:
            return "p < 0.0001"
        return f"p = {value:.4g}"

    @classmethod
    def _interpret_model_f_test(
        cls,
        f_test: pd.DataFrame,
        anova: pd.DataFrame | None,
        has_constant: bool | None = None,
    ) -> tuple[str, str, str]:
        p_value = cls._f_value(f_test, "p_value")
        f_value = cls._f_value(f_test, "F_value")
        model_df = cls._anova_df(anova, "Regression")
        residual_df = cls._anova_df(anova, "Residuals")
        if any(value is None or not np.isfinite(value) for value in (p_value, f_value, model_df, residual_df)) or model_df <= 0 or residual_df <= 0:
            return ("Not interpretable", "The F-test is undefined, commonly because model or residual degrees of freedom are insufficient.", "warning")
        p_text = cls._format_p_value(p_value)
        if has_constant is not False:
            if p_value < 0.01:
                return (
                    "Highly significant",
                    f"At the 1% level, there is evidence that the model explains variation beyond a constant response ({p_text}).",
                    "success",
                )
            if p_value < 0.05:
                return (
                    "Significant",
                    f"At the 5% level, there is evidence that the model explains variation beyond a constant response ({p_text}).",
                    "success",
                )
            return (
                "Not significant",
                f"At the 5% level, the F-test does not reject a constant-response model ({p_text}); this does not establish that all effects are absent.",
                "warning",
            )
        if p_value < 0.01:
            return (
                "Highly significant",
                f"At the 1% level, there is evidence against the joint null hypothesis that all coefficients are zero ({p_text}).",
                "success",
            )
        if p_value < 0.05:
            return (
                "Significant",
                f"At the 5% level, there is evidence against the joint null hypothesis that all coefficients are zero ({p_text}).",
                "success",
            )
        return (
            "Not significant",
            f"At the 5% level, the F-test does not reject the joint null hypothesis that all coefficients are zero ({p_text}); it gives no conclusion about individual terms.",
            "warning",
        )

    @classmethod
    def _interpret_lof_f_test(cls, f_test: pd.DataFrame, anova: pd.DataFrame | None) -> tuple[str, str, str]:
        p_value = cls._f_value(f_test, "p_value")
        f_value = cls._f_value(f_test, "F_value")
        lof_df = cls._anova_df(anova, "Lack of Fit")
        pure_error_df = cls._anova_df(anova, "Pure Error")
        if any(value is None or not np.isfinite(value) for value in (p_value, f_value, lof_df, pure_error_df)) or lof_df <= 0 or pure_error_df <= 0:
            return ("Not interpretable", "The lack-of-fit test requires replicated runs and positive lack-of-fit and pure-error degrees of freedom.", "warning")
        p_text = cls._format_p_value(p_value)
        if p_value < 0.05:
            return (
                "Significant lack of fit",
                f"At the 5% level, there is evidence of lack of fit relative to the estimated pure error ({p_text}).",
                "danger",
            )
        return (
            "No significant lack of fit",
            f"At the 5% level, there is no statistical evidence of lack of fit relative to pure error ({p_text}); this does not prove model adequacy.",
            "success",
        )

    @staticmethod
    def _assess_vif(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "Not defined"
        if not np.isfinite(number):
            return "Not estimable"
        if number < 5:
            return "Low"
        if number <= 10:
            return "Elevated"
        return "High"

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------
    def _render_optional_table(
        self,
        title: str,
        section: _OptionalSection,
        *,
        description: str = "",
        transform=None,
        after_table_gap: float = 4.0,
    ):
        self.add_subsection(title)
        if description:
            self.add_paragraph(description)
        if section.data is None or section.data.empty:
            notice = section.notice or "No data are available."
            if section.applicable:
                label = "Section unavailable"
            elif "not defined" in notice.lower():
                label = "Not defined"
            else:
                label = "Not applicable"
            self.add_notice(label, notice, kind="warning" if section.applicable else "neutral")
            return
        data = transform(section.data.copy(deep=True)) if transform else section.data
        self.add_table_from_df(data.reset_index(drop=True), row_color_mode="ROWS")
        if after_table_gap:
            self.ln(after_table_gap)

    def _render_design_overview(self, snapshot: _ReportSnapshot):
        self.add_page(orientation="P")
        self.set_y(20)
        self.add_section_title("Design Overview")
        summary = snapshot.design_summary.iloc[0].to_dict()
        self.add_kpi_cards(self._overview_card_values(summary), columns=3)

        factors = snapshot.factor_summary
        process = factors.loc[factors["Type"].isin(["cont", "cat"])].copy()
        mixture = factors.loc[factors["Type"] == "mix"].copy()
        if not process.empty:
            self.add_subsection("Process Factors")
            rows = []
            for _, factor in process.iterrows():
                factor_type = "Continuous" if factor["Type"] == "cont" else "Categorical"
                if factor["Type"] == "cont" and pd.notna(factor["Lower Bound"]) and pd.notna(factor["Upper Bound"]):
                    settings = f"{self._safe_text(factor['Lower Bound'])} to {self._safe_text(factor['Upper Bound'])}"
                else:
                    settings = self._safe_text(factor["Levels"])
                rows.append({
                    "Factor": factor["Factor"], "Type": factor_type, "Settings": settings,
                    "Coded levels": factor["Coded Levels"],
                })
            self.add_table_from_df(pd.DataFrame(rows), row_color_mode="ROWS")
            self.ln(4)
        if not mixture.empty:
            self.add_subsection("Mixture Components")
            table = mixture[["Factor", "Lower Bound", "Upper Bound"]].rename(columns={"Factor": "Component"})
            self.add_table_from_df(table, row_color_mode="ROWS")

        self.add_notice(
            "Design matrix",
            "The complete design matrix in actual factor units is provided in Appendix A.",
            kind="info",
        )

    def _render_model_structure(self, snapshot: _ReportSnapshot, diagnostics: dict[str, object]):
        self.add_page(orientation="P")
        self.add_section_title("Model Structure")
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.NAVY)
        self.write(5, "Responses: ")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        self.write(5, self._safe_text(list(snapshot.responses.columns)))
        self.ln(7)
        self.add_subsection("Included Terms")
        labels = [self._safe_text(value) for value in snapshot.model_terms["Terms"]]
        self.set_font("Helvetica", "B", 8.5)
        label_width = min(
            max((self.get_string_width(label) for label in labels), default=0) + 5,
            self._usable_width * 0.4,
        )
        for _, row in snapshot.model_terms.iterrows():
            label = self._safe_text(row["Terms"])
            value = self._safe_text(row["Included"])
            if value in {"", "N/A", "False", "No"}:
                value = "Not included"
            self.set_font("Helvetica", "B", 8.5)
            self.set_text_color(*self.NAVY)
            self.set_x(self.l_margin)
            self.cell(label_width, 5, label)
            self.set_font("Helvetica", "", 8.5)
            self.set_text_color(*self.DARK)
            self.multi_cell(self._usable_width - label_width, 5, value)
        self.ln(3)
        self.add_subsection("Model Matrix Diagnostics")
        diagnostic_table = pd.DataFrame({"Metric": list(diagnostics), "Value": list(diagnostics.values())})
        self.add_table_from_df(diagnostic_table, col_widths=[1.2, 1], row_color_mode="ROWS")
        if diagnostics["Aliased columns"]:
            self.add_notice("Rank deficient", "Some model-matrix columns are linearly dependent; individual coefficients may not be estimable.", kind="danger")
        elif diagnostics["Residual DF"] == 0:
            self.add_notice("Saturated model", "The model has no residual degrees of freedom, so residual variance and related tests are not interpretable.", kind="warning")
        else:
            self.add_notice("Full-rank model", "All model-matrix columns are linearly independent.", kind="success")

    def _render_general_diagnostics(self, snapshot: _ReportSnapshot, diagnostics: dict[str, object]):
        self.add_page(orientation="P")
        self.add_section_title("General Diagnostics")

        def vif_table(data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()
            if "Variable" in result.columns:
                result = result.rename(columns={"Variable": "Term"})
            elif "Term" not in result.columns:
                result = result.reset_index().rename(columns={"index": "Term"})
            if "VIF" not in result.columns:
                raise ValueError("VIF summary does not contain a VIF column.")
            result["Flag"] = result["VIF"].map(self._assess_vif)
            return result[["Term", "VIF", "Flag"]]

        mixture_present = bool((snapshot.factor_summary["Type"] == "mix").any())
        vif_description = "VIF < 5 is low, 5-10 is elevated, and VIF > 10 indicates high multicollinearity."
        if mixture_present:
            vif_description += " For mixture terms these thresholds depend on parameterization; rank and alias diagnostics are primary."
        self._render_optional_table(
            "Variance Inflation Factors (VIF)", snapshot.vif,
            description=vif_description,
            transform=vif_table,
        )

        self.add_subsection("Leverage")
        if snapshot.leverages.data is None or snapshot.leverages.data.empty:
            self.add_notice(
                "Section unavailable" if snapshot.leverages.applicable else "Not applicable",
                snapshot.leverages.notice or "No leverage values are available.",
                kind="warning" if snapshot.leverages.applicable else "neutral",
            )
        else:
            leverage = snapshot.leverages.data.copy()
            if "Leverage" not in leverage.columns:
                self.add_notice("Section unavailable", "Leverage summary does not contain a Leverage column.", kind="warning")
            else:
                leverage.insert(0, "Run", range(1, len(leverage) + 1))
                average = pd.to_numeric(leverage["Leverage"], errors="coerce").mean()
                self.add_paragraph(f"Average leverage: {self._safe_text(average)}")
                self.add_table_from_df(leverage[["Run", "Leverage"]], row_color_mode="ROWS")
                self.ln(4)

        self._render_optional_table(
            "Response Conditions", snapshot.response_conditions,
            description="Optimization limits and goals configured for each response.",
        )

    def _render_response(self, response: object, snapshot: _ReportSnapshot, diagnostics: dict[str, object]):
        analysis = snapshot.response_analysis[response]
        self.add_page(orientation="P")
        self.add_section_title(f"Response Analysis: {response}")

        self._render_optional_table(
            "Model Metrics", analysis.metrics,
            transform=lambda data: data.rename(columns={
                "R2": "R^2", "R2_adj": "Adj. R^2", "Q2": "Q^2 (LOO)",
                "PRESS": "PRESS (LOO)", "RMSE_CV": "RMSE (LOO)",
            }),
        )
        self.add_subsection("Coefficient Summary")
        coefficients = analysis.coefficients.rename(columns={
            "Variable": "Term", "Std_Error": "Std. Error", "Conf_Int": "95% CI half-width",
            "Lower_CI": "Lower CI", "Upper_CI": "Upper CI", "p_value": "p-value",
        })
        self.add_table_from_df(coefficients, row_color_mode="ROWS")
        self.ln(4)

        anova_data = analysis.anova.data
        self._render_optional_table(
            "Model F-test", analysis.model_f_test,
            transform=lambda data: data.rename(columns={
                "F_value": "F-value", "F_crit_95%": "F-crit (95%)",
                "F_crit_99%": "F-crit (99%)", "p_value": "p-value",
            }),
            after_table_gap=0,
        )
        if analysis.model_f_test.data is not None and not analysis.model_f_test.data.empty:
            title, message, kind = self._interpret_model_f_test(
                analysis.model_f_test.data,
                anova_data,
                has_constant=diagnostics["Intercept"] == "Yes",
            )
            self.add_notice(title, message, kind=kind)

        self._render_optional_table(
            "Lack-of-Fit F-test", analysis.lof_f_test,
            transform=lambda data: data.rename(columns={
                "F_value": "F-value", "F_crit_95%": "F-crit (95%)",
                "F_crit_99%": "F-crit (99%)", "p_value": "p-value",
            }),
            after_table_gap=0,
        )
        if analysis.lof_f_test.data is not None and not analysis.lof_f_test.data.empty:
            title, message, kind = self._interpret_lof_f_test(analysis.lof_f_test.data, anova_data)
            self.add_notice(title, message, kind=kind)

        self._render_optional_table(
            "ANOVA", analysis.anova,
            transform=lambda data: data.rename(columns={"df": "dof"}),
        )
        self._render_optional_table("Replicate Summary", analysis.replicates)

        self.add_subsection("Observed and Predicted Responses")
        if snapshot.predictions.data is None or response not in snapshot.predictions.data.columns:
            self.add_notice("Section unavailable", snapshot.predictions.notice or "Predictions are unavailable for this response.", kind="warning")
        else:
            table = pd.DataFrame({
                "Run": range(1, len(snapshot.responses) + 1),
                "Observed": snapshot.responses[response],
                "Predicted": snapshot.predictions.data[response],
            })
            self.add_table_from_df(table, row_color_mode="ROWS")

    def _render_report_notes(self):
        if not self._report_notices:
            return
        self.add_page(orientation="P")
        self.add_section_title("Report Notes")
        self.add_paragraph("The following optional sections could not be populated completely:")
        self.ln(2)
        for notice in self._report_notices:
            self.add_notice("Unavailable section", notice, kind="warning")

    def _render_design_matrix_appendix(self, design_matrix: pd.DataFrame):
        matrix = design_matrix.reset_index(drop=True).copy(deep=True)
        matrix.insert(0, "Run", range(1, len(matrix) + 1))
        landscape_width = max(self.w, self.h) - self.l_margin - self.r_margin
        groups = self._split_design_matrix(matrix, max_width=landscape_width)
        for index, group in enumerate(groups, start=1):
            self.add_page(orientation="L")
            suffix = " - Design Matrix" if len(groups) == 1 else f" - Design Matrix ({index}/{len(groups)})"
            self.add_appendix_title(suffix)
            if len(groups) > 1:
                self.add_paragraph("Wide matrices are split into column groups; the Run column is repeated in every group.")
                self.ln(2)
            self.add_table_from_df(group, row_color_mode="ROWS", font_size=7)

    # ------------------------------------------------------------------
    # Build and atomic output
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_output_path(filename: str | os.PathLike) -> Path:
        output = Path(filename)
        if output.suffix.lower() != ".pdf":
            raise ValueError("PDF filename must end with '.pdf'.")
        if not output.parent.exists() or not output.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {output.parent}")
        return output

    @staticmethod
    def _atomic_write(output: Path, payload: bytes) -> None:
        temporary_path: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{output.stem}-", suffix=".tmp", dir=output.parent
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, output)
            temporary_path = None
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def build(self, filename: str):
        """Build the complete report and publish it atomically."""
        output = self._validate_output_path(filename)
        snapshot = self._create_snapshot()
        diagnostics = self._model_matrix_diagnostics(snapshot.model_matrix)

        self._render_design_overview(snapshot)
        self._render_model_structure(snapshot, diagnostics)
        self._render_general_diagnostics(snapshot, diagnostics)
        for response in snapshot.responses.columns:
            self._render_response(response, snapshot, diagnostics)
        self._render_report_notes()
        self._render_design_matrix_appendix(snapshot.design_matrix)

        payload = bytes(self.output())
        self._atomic_write(output, payload)
