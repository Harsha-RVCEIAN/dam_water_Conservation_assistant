"""Professional PDF report generator for Layer A Technical Stabilization.

Converts the Layer A completion audit into a publication-ready PDF document.
Uses ReportLab with custom NumberedCanvas for dynamic page numbering.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PDF_OUTPUT_PATH = Path("reports/Layer_A_Stabilization_Report.pdf")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54,
                11 * inch - 36,
                "KRS Dam Water Conservation Assistant — Layer A Technical Stabilization Report",
            )
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer (all pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 32, page_str)
        self.drawString(
            54,
            32,
            "Confidential & Proprietary — Google DeepMind / Antigravity Engineering",
        )
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 44, 8.5 * inch - 54, 44)

        self.restoreState()


def create_pdf_report(output_path: Path = PDF_OUTPUT_PATH) -> Path:
    """Build the comprehensive Layer A completion PDF report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom palette styles
    primary_color = colors.HexColor("#1A365D")   # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate Blue
    body_color = colors.HexColor("#2D3748")      # Dark Grey Body
    callout_bg = colors.HexColor("#F7FAFC")
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceAfter=14,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=body_color,
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=body_color,
        leftIndent=12,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#2D3748"),
        backColor=colors.HexColor("#EDF2F7"),
        borderPadding=4,
        spaceAfter=6,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    table_body_style = ParagraphStyle(
        "TableBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=body_color,
    )

    table_body_bold = ParagraphStyle(
        "TableBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=body_color,
    )

    story: List[object] = []

    # -------------------------------------------------------------
    # Cover / Header Title Block
    # -------------------------------------------------------------
    story.append(Paragraph("K.R.Sagara Dam Water Conservation Assistant", title_style))
    story.append(Paragraph("Layer A Technical Stabilization — Final Engineering Completion Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

    meta_table_data = [
        [
            Paragraph("<b>Date:</b> 2026-11-05", meta_style),
            Paragraph("<b>Author / Role:</b> Antigravity AI Assistant (Google DeepMind)", meta_style),
        ],
        [
            Paragraph("<b>Target Contract:</b> antigravity_layer_a_implementation_plan.md", meta_style),
            Paragraph("<b>Scope:</b> Layer A Only (Layer B Excluded)", meta_style),
        ],
        [
            Paragraph("<b>Test Suite Status:</b> 46 Passed / 0 Failed", meta_style),
            Paragraph("<b>Execution Environment:</b> Python 3.11.9 (.venv)", meta_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[240, 264])
    meta_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # Executive Summary Box
    # -------------------------------------------------------------
    summary_p = Paragraph(
        "<b>Executive Summary:</b> All non-blocked tasks specified in the Layer A Implementation Plan have been successfully executed, tested, and verified. "
        "The KRS Dam dataset engineering, unit conversions, storage variable semantics, water-balance equations, dynamic forecasting freshness, and dashboard "
        "have been stabilized to ensure technical consistency, strict reproducible MLOps provenance, and zero false leakage claims. "
        "Layer B (recommendations, allocation optimization, gate control) remains strictly unbuilt.",
        body_style,
    )
    summary_box = Table([[summary_p]], colWidths=[504])
    summary_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), callout_bg),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_box)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # Section 1: Files Changed and Engineering Rationale
    # -------------------------------------------------------------
    story.append(Paragraph("1. Summary of Files Changed & Engineering Rationale", h1_style))
    files_table_data = [
        [Paragraph("File Path", table_header_style), Paragraph("Action", table_header_style), Paragraph("Purpose & Engineering Rationale", table_header_style)],
        [
            Paragraph("config/reservoir_reference.yaml", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Centralized reservoir identity, capacity references (Design Gross 49.452 TMC, Live 45.051 TMC, 2009 Survey 42.99 TMC), and unverified live denominator policy.", table_body_style),
        ],
        [
            Paragraph("config/data_quality.yaml", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Centralized physical plausibility bounds (elevation 2500 ft, storage 49.452 TMC, non-negative flows) and NaN preservation policies.", table_body_style),
        ],
        [
            Paragraph("config/water_balance.yaml", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Centralized unit conversion factor (0.0000864 TMC/cusec-day), unverified evaporation status, interval matching rules, and anti-leakage disclaimers.", table_body_style),
        ],
        [
            Paragraph("config/freshness.yaml", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Configurable observation delay threshold (1 day) and dynamic freshness labeling criteria.", table_body_style),
        ],
        [
            Paragraph("config/rainfall_stations.yaml", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Documents rainfall pipeline blocker and 10 regional proxy stations awaiting verified external data ingestion.", table_body_style),
        ],
        [
            Paragraph("src/utils/unit_conversions.py", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Pure, centralized flow rate (cusecs) to volume (TMC) conversions over arbitrary hours with strict negative/NaN validation.", table_body_style),
        ],
        [
            Paragraph("src/utils/validate_conversions.py", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Validates daily converted flow against cumulative meter deltas across 36 years; exports validation report.", table_body_style),
        ],
        [
            Paragraph("src/data/validate.py", table_body_bold),
            Paragraph("Enhanced", table_body_style),
            Paragraph("Generates CSV/JSON audit reports; detects duplicates, negative values, missing dates; flags anomalies without deleting raw records.", table_body_style),
        ],
        [
            Paragraph("src/models/storage_analysis.py", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Pure functions distinguishing gross vs live storage, enforcing continuous daily intervals (blocking multi-day deltas), and setting unverified percentages to N/A.", table_body_style),
        ],
        [
            Paragraph("src/models/water_balance.py", table_body_bold),
            Paragraph("Created", table_body_style),
            Paragraph("Canonical mass balance module returning explicit result contracts; enforces interval alignment; blocks numeric discrepancy when evaporation is unverified.", table_body_style),
        ],
        [
            Paragraph("src/models/water_budget.py", table_body_bold),
            Paragraph("Refactored", table_body_style),
            Paragraph("Maintained as backward-compatible wrapper delegating to unit_conversions.py and water_balance.py.", table_body_style),
        ],
        [
            Paragraph("src/data/preprocess.py", table_body_bold),
            Paragraph("Refactored", table_body_style),
            Paragraph("Eliminated zero-filling of missing evaporation; enforces 1-day interval for storage change diff; uses pure unit conversions.", table_body_style),
        ],
        [
            Paragraph("src/models/predict.py", table_body_bold),
            Paragraph("Enhanced", table_body_style),
            Paragraph("Adds dynamic observation freshness calculation, prevents relabeling stale historical records as 'tomorrow', verifies model metadata integrity.", table_body_style),
        ],
        [
            Paragraph("src/app.py", table_body_bold),
            Paragraph("Stabilized", table_body_style),
            Paragraph("Modernized dashboard with dynamic data freshness badges, unverified denominator warnings, water-balance diagnostics, and clear leakage disclaimers.", table_body_style),
        ],
    ]
    files_table = Table(files_table_data, colWidths=[130, 54, 320])
    files_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, callout_bg]),
    ]))
    story.append(files_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------
    # Section 2: Baseline Problems Confirmed From Code
    # -------------------------------------------------------------
    story.append(Paragraph("2. Baseline Problems Confirmed From Actual Code", h1_style))
    story.append(Paragraph("• <b>Evaporation Imputation:</b> <code>src/data/preprocess.py</code> previously filled 546 missing evaporation readings with <code>0.0</code>, masking physical data gaps.", bullet_style))
    story.append(Paragraph("• <b>Unverified Evaporation Units:</b> Correlation between daily evaporation volume and cumulative evaporation delta is only <code>0.5108</code> (vs <code>1.000000</code> for inflow/outflow), proving source convention is unverified.", bullet_style))
    story.append(Paragraph("• <b>Broken Storage Change Across Calendar Gaps:</b> <code>gross_capacity_tmc.diff()</code> assumed daily rows; across missing date <code>2020-02-11</code>, it evaluated a 2-day change and compared it to 1-day flow.", bullet_style))
    story.append(Paragraph("• <b>Storage Semantics Conflation:</b> Gross storage was conflated with live storage, and percentage full was computed against gross design capacity without bathymetric verification.", bullet_style))
    story.append(Paragraph("• <b>Scattered Magic Numbers:</b> The constant <code>0.0000864</code> was duplicated across multiple modules without input validation.", bullet_style))
    story.append(Paragraph("• <b>Stale Forecast Presentation:</b> The 15-day-old forecast (origin 2026-10-21) was displayed as 'tomorrow's forecast' without dynamic freshness verification.", bullet_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # Section 3: Engineering Assumptions Made
    # -------------------------------------------------------------
    story.append(Paragraph("3. Engineering Assumptions & Domain Rationale", h1_style))
    story.append(Paragraph("• <b>Daily Mean Flow Factor:</b> 1 cusec sustained for 24 hours equals exactly <code>0.0000864 TMC</code> (validated by 1.000000 correlation against cumulative meter deltas).", bullet_style))
    story.append(Paragraph("• <b>Dead Storage Sill Offset:</b> Difference between gross capacity and live storage above cill is approximately <code>4.401 TMC</code> (49.452 - 45.051 = 4.401 TMC).", bullet_style))
    story.append(Paragraph("• <b>Evaporation Incompleteness Policy:</b> Unverified evaporation convention blocks complete mass-balance closure; discrepancy is held at <code>None</code> for unverified intervals.", bullet_style))
    story.append(Paragraph("• <b>Live Storage Denominator Policy:</b> Storage percentage is reported as N/A / unverified until certified bathymetric surveys are provided.", bullet_style))
    story.append(Paragraph("• <b>Freshness Policy:</b> Forecasts are marked <code>HISTORICAL_STALE</code> if the latest observation is older than 1 day relative to current system execution.", bullet_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # Section 4: Unresolved Source / Data Questions
    # -------------------------------------------------------------
    story.append(Paragraph("4. Unresolved Source & Domain Questions", h1_style))
    story.append(Paragraph("• <b>Evaporation Measurement Methodology:</b> Does <code>Evaporation (Cusecs)</code> represent pan evaporation with an undocumented factor, or reservoir depth loss without dynamic surface area weighting?", bullet_style))
    story.append(Paragraph("• <b>Gross Capacity Readings Above Design:</b> On 1994-10-06, 1999-02-26, and 2011-02-14 to 2011-02-16, recorded values peak at <code>69.322 TMC</code> (exceeding 49.452 TMC design capacity). Historical log verification from Karnataka WRD is required.", bullet_style))
    story.append(Paragraph("• <b>Current Hydrographic Live Storage Bathymetry:</b> An updated sedimentation survey since 2009 is required to establish a certified live capacity denominator.", bullet_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # Section 5: Unit Conversion & Water Balance Audit
    # -------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("5. Flow Unit Conversion & Water Balance Verification", h1_style))
    story.append(Paragraph("Unit conversions were validated across 13,442 daily historical records against cumulative flow meters:", body_style))

    conv_table_data = [
        [Paragraph("Flow Series", table_header_style), Paragraph("Valid Pairs", table_header_style), Paragraph("Correlation", table_header_style), Paragraph("MAE (TMC)", table_header_style), Paragraph("Status", table_header_style), Paragraph("Engineering Finding", table_header_style)],
        [
            Paragraph("River Inflow", table_body_bold),
            Paragraph("13,396", table_body_style),
            Paragraph("1.000000", table_body_style),
            Paragraph("0.000319", table_body_style),
            Paragraph("PASS", table_body_bold),
            Paragraph("Extreme consistency; validates 0.0000864 TMC/cusec-day factor.", table_body_style),
        ],
        [
            Paragraph("River Outflow", table_body_bold),
            Paragraph("13,403", table_body_style),
            Paragraph("1.000000", table_body_style),
            Paragraph("0.000339", table_body_style),
            Paragraph("PASS", table_body_bold),
            Paragraph("Extreme consistency with cumulative outflow meter.", table_body_style),
        ],
        [
            Paragraph("Canal Withdrawal", table_body_bold),
            Paragraph("13,403", table_body_style),
            Paragraph("0.999993", table_body_style),
            Paragraph("0.000364", table_body_style),
            Paragraph("PASS", table_body_bold),
            Paragraph("High consistency with cumulative withdrawal meter.", table_body_style),
        ],
        [
            Paragraph("Evaporation", table_body_bold),
            Paragraph("12,857", table_body_style),
            Paragraph("0.510785", table_body_style),
            Paragraph("0.000197", table_body_style),
            Paragraph("UNVERIFIED", table_body_bold),
            Paragraph("Low correlation; convention and area-elevation curve unverified.", table_body_style),
        ],
    ]
    conv_table = Table(conv_table_data, colWidths=[90, 55, 60, 55, 65, 179])
    conv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, callout_bg]),
    ]))
    story.append(conv_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Historical Water-Balance Audit Status (13,441 Evaluated Intervals)", h2_style))
    story.append(Paragraph("• <b>UNVERIFIED_EVAPORATION:</b> <code>12,888 intervals (95.88%)</code> — Aligned flow terms complete, but numeric discrepancy blocked due to unverified evaporation convention.", bullet_style))
    story.append(Paragraph("• <b>INCOMPLETE:</b> <code>553 intervals (4.12%)</code> — Incomplete due to missing evaporation readings or missing calendar day (2020-02-11).", bullet_style))
    story.append(Paragraph("• <b>Anti-Leakage Disclaimer:</b> Departures from the simplified daily mass balance reflect unmeasured catchment runoff, localized rainfall, travel timing delays, or gauge calibration; they do NOT prove dam leakage.", bullet_style))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # Section 6: Model Evaluation & Benchmark Comparison
    # -------------------------------------------------------------
    story.append(Paragraph("6. Chronological Inflow Forecaster Benchmarks", h1_style))
    story.append(Paragraph("Models were trained strictly on pre-2023 data (1990–2022, 12,035 observations) with zero future target leakage, selected on validation data, and benchmarked on held-out test data:", body_style))

    model_table_data = [
        [Paragraph("Model / Baseline", table_header_style), Paragraph("Partition", table_header_style), Paragraph("MAE (cfs)", table_header_style), Paragraph("RMSE (cfs)", table_header_style), Paragraph("R²", table_header_style), Paragraph("WAPE (%)", table_header_style)],
        [
            Paragraph("Persistence Baseline (t-1)", table_body_bold),
            Paragraph("Validation (2023-2024)", table_body_style),
            Paragraph("2,168.32", table_body_style),
            Paragraph("8,111.96", table_body_style),
            Paragraph("0.5634", table_body_style),
            Paragraph("48.51%", table_body_style),
        ],
        [
            Paragraph("7-Day Moving Average", table_body_bold),
            Paragraph("Validation (2023-2024)", table_body_style),
            Paragraph("2,682.49", table_body_style),
            Paragraph("8,639.46", table_body_style),
            Paragraph("0.5049", table_body_style),
            Paragraph("60.01%", table_body_style),
        ],
        [
            Paragraph("Ridge Regression", table_body_bold),
            Paragraph("Validation (2023-2024)", table_body_style),
            Paragraph("2,246.33", table_body_style),
            Paragraph("7,610.15", table_body_style),
            Paragraph("0.6152", table_body_style),
            Paragraph("50.25%", table_body_style),
        ],
        [
            Paragraph("HistGradientBoosting", table_body_bold),
            Paragraph("Validation (2023-2024)", table_body_style),
            Paragraph("2,185.74", table_body_style),
            Paragraph("7,539.02", table_body_style),
            Paragraph("0.6224", table_body_style),
            Paragraph("48.90%", table_body_style),
        ],
        [
            Paragraph("Tuned Random Forest (Winner)", table_body_bold),
            Paragraph("Validation (2023-2024)", table_body_style),
            Paragraph("2,060.03", table_body_style),
            Paragraph("7,192.17", table_body_style),
            Paragraph("0.6558", table_body_style),
            Paragraph("46.08%", table_body_style),
        ],
        [
            Paragraph("Persistence Baseline (t-1)", table_body_bold),
            Paragraph("Held-Out Test (2025-2026)", table_body_style),
            Paragraph("2,278.40", table_body_style),
            Paragraph("6,685.56", table_body_style),
            Paragraph("0.4796", table_body_style),
            Paragraph("43.20%", table_body_style),
        ],
        [
            Paragraph("Tuned Random Forest", table_body_bold),
            Paragraph("Held-Out Test (2025-2026)", table_body_style),
            Paragraph("2,347.07", table_body_style),
            Paragraph("5,929.95", table_body_style),
            Paragraph("0.5906", table_body_style),
            Paragraph("44.50%", table_body_style),
        ],
    ]
    model_table = Table(model_table_data, colWidths=[140, 110, 65, 65, 55, 69])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, callout_bg]),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Test Benchmark Finding:</b> Tuned Random Forest achieves an <b>11.30% reduction in RMSE</b> over persistence on held-out test data, with R² improving from 0.48 to 0.59.", body_style))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # Section 7: Gate Completion Matrix & Layer B Exclusion
    # -------------------------------------------------------------
    story.append(Paragraph("7. Layer A Gate Completion Matrix & Layer B Boundaries", h1_style))

    gate_table_data = [
        [Paragraph("Gate / Scope Element", table_header_style), Paragraph("Status", table_header_style), Paragraph("Implementation Details & Justification", table_header_style)],
        [
            Paragraph("Gate 0: Baseline Inspection", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("Internal inventory captured; baseline report written to reports/layer_a_baseline.md.", table_body_style),
        ],
        [
            Paragraph("Gate 1: Centralized Data Validation", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("Schema, duplicates, calendar gaps, and plausible ranges audited in reports/data_quality_report.csv.", table_body_style),
        ],
        [
            Paragraph("Gate 2: Storage Semantics", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("storage_analysis.py created; live vs gross storage separated; unverified percentage set to N/A.", table_body_style),
        ],
        [
            Paragraph("Gate 3: Flow Unit Conversions", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("unit_conversions.py implemented; 1.000000 correlation validated in unit_conversion_validation.csv.", table_body_style),
        ],
        [
            Paragraph("Gate 4: Water Balance Correction", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("Canonical water_balance.py enforces interval alignment; unverified evaporation blocks discrepancy.", table_body_style),
        ],
        [
            Paragraph("Gate 5: Rainfall Pipeline", table_body_bold),
            Paragraph("BLOCKED", table_body_style),
            Paragraph("Raw rainfall source archive absent from repository; documented transparently in config and reports.", table_body_style),
        ],
        [
            Paragraph("Gate 6: Forecasting Integrity", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("Dynamic freshness checks in predict.py flag archive age (15 days stale); target derived from origin.", table_body_style),
        ],
        [
            Paragraph("Gate 7: Anomaly Separation", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("Separated data-quality, water-balance, and residual flags; eliminated all leakage claims.", table_body_style),
        ],
        [
            Paragraph("Gate 8: MLOps Reproducibility", table_body_bold),
            Paragraph("PARTIAL", table_body_style),
            Paragraph("Local DVC pipeline and MLflow tracking fully functional; cloud remote blocked (no external credentials).", table_body_style),
        ],
        [
            Paragraph("Gate 9: Dashboard Stabilization", table_body_bold),
            Paragraph("DONE", table_body_style),
            Paragraph("src/app.py stabilized with dynamic freshness badges, storage semantics, and verified outputs.", table_body_style),
        ],
        [
            Paragraph("Layer B: Recommendation Engine", table_body_bold),
            Paragraph("FORBIDDEN / NOT IMPLEMENTED", table_body_style),
            Paragraph("No recommendation APIs, tables, or recommendation ML models created per Section 0 contract.", table_body_style),
        ],
        [
            Paragraph("Layer B: Water Allocation Optimization", table_body_bold),
            Paragraph("FORBIDDEN / NOT IMPLEMENTED", table_body_style),
            Paragraph("No gate-dispatch, demand optimization, or release scheduling logic added.", table_body_style),
        ],
        [
            Paragraph("Layer B: Leakage Diagnosis", table_body_bold),
            Paragraph("FORBIDDEN / NOT IMPLEMENTED", table_body_style),
            Paragraph("Discrepancies explicitly disclaimed as unmodeled runoff or timing delays; never leakage.", table_body_style),
        ],
    ]
    gate_table = Table(gate_table_data, colWidths=[140, 75, 289])
    gate_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, callout_bg]),
    ]))
    story.append(gate_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # Section 8: Exact Commands & Test Verification
    # -------------------------------------------------------------
    story.append(Paragraph("8. Reproduction Commands & Automated Test Verification", h1_style))
    story.append(Paragraph("All 46 focused Layer A tests pass with zero failures:", body_style))

    repro_code = (
        "# 1. Run complete Layer A automated test suite\n"
        ".\\.venv\\Scripts\\pytest tests/test_validation.py tests/test_storage_analysis.py "
        "tests/test_unit_conversions.py tests/test_water_balance.py tests/test_forecasting.py "
        "tests/test_inference_freshness.py tests/test_units_and_budget.py tests/test_app_smoke.py -v\n\n"
        "# 2. Run data validation and export CSV/JSON audits\n"
        ".\\.venv\\Scripts\\python.exe -m src.data.validate\n\n"
        "# 3. Run flow unit conversion validation\n"
        ".\\.venv\\Scripts\\python.exe -m src.utils.validate_conversions\n\n"
        "# 4. Launch stabilized Streamlit dashboard\n"
        ".\\.venv\\Scripts\\python.exe -m streamlit run src/app.py"
    )
    story.append(Paragraph(repro_code.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path


def main():
    pdf_path = create_pdf_report()
    print(f"Publication-ready PDF report successfully generated at: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()

