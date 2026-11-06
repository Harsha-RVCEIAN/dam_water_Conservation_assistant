"""Streamlit dashboard for Dam Water Conservation Assistant (Layer A Stabilized).

Provides interactive reservoir monitoring, historical hydrological trends,
machine learning inflow forecasting with uncertainty intervals and dynamic freshness,
statistical anomaly review flags, water-balance diagnostics, and hypothetical
conservation scenario calculations.

Scope: Layer A Technical Stabilization Only. No Layer B recommendation engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.predict import InflowForecaster
from src.models.conservation_assessment import (
    assess_reservoir_condition,
    detect_abnormal_flow_patterns,
    assess_water_shortage_risk,
)
from src.models.conservation_recommender import generate_conservation_recommendations
from src.models.storage_analysis import extract_storage_summary, load_reservoir_config
from src.models.water_balance import compute_water_balance_interval, load_water_balance_config
from src.models.water_budget import (
    KRS_DEAD_STORAGE_TMC,
    KRS_DESIGN_GROSS_CAPACITY_TMC,
    KRS_DESIGN_LIVE_CAPACITY_TMC,
    compare_conservation_scenarios,
    cusec_day_to_tmc,
    simulate_conservation_scenario,
)

PROCESSED_DATA_PATH = Path("data/processed/krs_dam_processed.csv")
RAW_DATA_PATH = Path("data/raw/krs_dam_dataset.csv")
MODEL_PATH = Path("models/inflow_forecaster.joblib")
METADATA_PATH = Path("models/model_metadata.json")
CONVERSION_REPORT_PATH = Path("reports/unit_conversion_validation.csv")
WATER_BALANCE_AUDIT_PATH = Path("reports/water_balance_audit.csv")

st.set_page_config(
    page_title="K.R.Sagara Dam - Water Conservation Assistant (Layer A)",
    page_icon="🌊",
    layout="wide",
)


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Load cached processed data."""
    if not PROCESSED_DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    return df


@st.cache_resource
def load_forecaster() -> Optional[InflowForecaster]:
    """Load cached model forecaster with fail-safe error handling."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None
    try:
        return InflowForecaster(MODEL_PATH, METADATA_PATH)
    except Exception as exc:
        st.error(f"Failed to load forecaster: {exc}")
        return None


def main() -> None:
    st.title("🌊 K.R.Sagara Dam - Water Conservation Assistant")
    st.markdown(
        "**Student MLOps Project** &middot; Explainable Decision-Support Prototype &middot; *Cauvery Basin, Karnataka, India*  \n"
        "*(Layer A Technical Stabilization Only &middot; Layer B Recommendations Excluded)*"
    )

    df = load_processed_data()
    forecaster = load_forecaster()
    reservoir_cfg = load_reservoir_config()

    if df.empty:
        st.error(
            "⚠️ Processed dataset not found. Please run the preprocessing pipeline first: "
            "`python -m src.data.preprocess`"
        )
        return

    # Dynamic observation date and freshness calculation
    latest_row = df.iloc[-1]
    latest_date = latest_row["calendar_date"]
    latest_date_str = latest_date.strftime("%Y-%m-%d")

    # Evaluate dynamic data freshness against today
    today = pd.Timestamp.now().normalize()
    data_age_days = (today - latest_date).days
    is_stale = data_age_days > 1

    if is_stale:
        st.warning(
            f"⚠️ **Data Freshness Notice:** Latest observation in this archive is **{latest_date_str}** "
            f"({data_age_days} days old relative to system date {today.strftime('%Y-%m-%d')}). "
            "This application operates on a historical research archive; it is **not connected to live dam telemetry**."
        )
    else:
        st.success(
            f"✅ **Data Freshness:** Latest observation is current (**{latest_date_str}**)."
        )

    # Initialize model forecast safely before tab rendering
    forecast_res = None
    if forecaster:
        try:
            forecast_res = forecaster.predict_latest_from_processed(PROCESSED_DATA_PATH)
        except Exception:
            forecast_res = None

    tabs = st.tabs([
        "📊 Reservoir Overview",
        "📈 Historical Trends",
        "🤖 Inflow Forecast",
        "🚩 Unusual Pattern Flags & Balance",
        "💧 Conservation Scenario Simulator",
        "📋 Data Quality & MLOps Status",
        "🛡️ Water Conservation Assistant",
    ])

    # -------------------------------------------------------------
    # TAB 1: Reservoir Overview
    # -------------------------------------------------------------
    with tabs[0]:
        st.subheader("Current Reservoir Status [OBSERVED HISTORICAL RECORD]")
        st.caption(f"Observation Date: **{latest_date_str}** (Status: {'HISTORICAL_STALE' if is_stale else 'CURRENT'})")

        storage_summary = extract_storage_summary(df, reservoir_cfg)
        live_val = storage_summary.get("current_live_storage_tmc")
        gross_val = storage_summary.get("current_gross_storage_tmc")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label="Live Storage Above Sill (TMC)",
            value=f"{live_val:.2f} TMC" if live_val is not None else "N/A",
            help="Candidate canonical live water storage available above sill level",
        )
        c2.metric(
            label="Gross Capacity (TMC)",
            value=f"{gross_val:.2f} TMC" if gross_val is not None else "N/A",
            help=f"Gross storage (Live + ~{KRS_DEAD_STORAGE_TMC} TMC dead storage)",
        )
        c3.metric(
            label="Reservoir Elevation (ft MSL)",
            value=f"{latest_row['reservoir_level_ft']:.1f} ft",
            help="Normal Full Reservoir Level is ~2482.0 ft MSL",
        )
        c4.metric(
            label="Live Storage %",
            value="N/A (Denominator Unverified)",
            help="Percentage full denominator requires certified current bathymetry. Design gross reference is 49.452 TMC, design live is 45.051 TMC, and 2009 survey was 42.99 TMC.",
        )

        st.info(
            "📐 **Storage Semantics & Capacity References:**  \n"
            "- **Design Gross Reference:** `49.452 TMC` &middot; **Design Live Reference:** `45.051 TMC`  \n"
            "- **2009 Surveyed Live Reference:** `42.99 TMC` (historical hydrographic survey)  \n"
            f"- **Dead Storage Below Sill:** `~{KRS_DEAD_STORAGE_TMC} TMC` (gross - live above sill)  \n"
            "- *Percentage full is marked unverified/NA because no current authoritative operational denominator is certified.*"
        )

        st.markdown("---")
        st.subheader("Daily Flow Balance Rates [OBSERVED]")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("River Inflow", f"{latest_row['inflow_cusecs']:,.0f} cfs", f"{latest_row['inflow_vol_tmc']:.4f} TMC/day")
        f2.metric("River Outflow", f"{latest_row['outflow_cusecs']:,.0f} cfs", f"{latest_row['outflow_vol_tmc']:.4f} TMC/day")
        f3.metric("Canal Withdrawal", f"{latest_row['withdrawal_cusecs']:,.0f} cfs", f"{latest_row['withdrawal_vol_tmc']:.4f} TMC/day")
        evap_raw = latest_row["evaporation_cusecs"]
        if pd.isna(evap_raw):
            f4.metric("Evaporation", "Missing / Unknown", "N/A")
        else:
            f4.metric("Evaporation", f"{evap_raw:,.0f} cfs (Unverified)", "Unverified convention")

        st.caption(
            "Unit Rule: Flow rates in **cusecs** (cubic feet per second) represent daily mean rates. "
            "1 cusec-day = **0.0000864 TMC**. Evaporation measurement convention remains unverified at source."
        )

    # -------------------------------------------------------------
    # TAB 2: Historical Trends
    # -------------------------------------------------------------
    with tabs[1]:
        st.subheader("Historical Hydrological Trends [OBSERVED]")
        st.markdown("Explore observations across the 36-year chronological dataset (1990–2026):")

        years_available = sorted(df["calendar_date"].dt.year.unique())
        col_y1, col_y2 = st.columns(2)
        start_year = col_y1.selectbox("Start Year", years_available, index=max(0, len(years_available) - 6))
        end_year = col_y2.selectbox("End Year", years_available, index=len(years_available) - 1)

        filtered = df[
            (df["calendar_date"].dt.year >= start_year) & (df["calendar_date"].dt.year <= end_year)
        ]

        # Flow chart
        fig_flows = go.Figure()
        fig_flows.add_trace(go.Scatter(
            x=filtered["calendar_date"],
            y=filtered["inflow_cusecs"],
            name="Inflow (cusecs)",
            line=dict(color="#1f77b4", width=1.5),
        ))
        fig_flows.add_trace(go.Scatter(
            x=filtered["calendar_date"],
            y=filtered["outflow_cusecs"],
            name="River Outflow (cusecs)",
            line=dict(color="#ff7f0e", width=1.2),
        ))
        fig_flows.add_trace(go.Scatter(
            x=filtered["calendar_date"],
            y=filtered["withdrawal_cusecs"],
            name="Canal Withdrawal (cusecs)",
            line=dict(color="#2ca02c", width=1.2),
        ))
        fig_flows.update_layout(
            title=f"Daily Mean Flow Rates ({start_year} - {end_year})",
            xaxis_title="Date",
            yaxis_title="Flow Rate (cusecs)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
        )
        st.plotly_chart(fig_flows, use_container_width=True)

        # Storage chart
        fig_storage = go.Figure()
        fig_storage.add_trace(go.Scatter(
            x=filtered["calendar_date"],
            y=filtered["gross_capacity_tmc"],
            name="Gross Storage (TMC)",
            line=dict(color="#3b5998", width=2),
            fill="tozeroy",
        ))
        if "live_above_cill_tmc" in filtered.columns:
            fig_storage.add_trace(go.Scatter(
                x=filtered["calendar_date"],
                y=filtered["live_above_cill_tmc"],
                name="Live Storage Above Sill (TMC)",
                line=dict(color="#2ca02c", width=1.5, dash="dot"),
            ))
        fig_storage.add_hline(
            y=KRS_DESIGN_GROSS_CAPACITY_TMC,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Design Gross Capacity Reference: {KRS_DESIGN_GROSS_CAPACITY_TMC} TMC",
            annotation_position="bottom right",
        )
        fig_storage.update_layout(
            title=f"Reservoir Storage Volume Series ({start_year} - {end_year})",
            xaxis_title="Date",
            yaxis_title="Storage Volume (TMC)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=380,
        )
        st.plotly_chart(fig_storage, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Inflow Forecast
    # -------------------------------------------------------------
    with tabs[2]:
        st.subheader("Machine Learning Inflow Forecast [CHRONOLOGICAL MODEL]")
        st.markdown(
            "**Forecast Target:** Next-Day River Inflow (`Inflow (t+1)` in cusecs) &middot; "
            "**Model:** Tuned Random Forest Regressor trained on chronological data (1990–2022) with validation-based selection."
        )

        if forecaster is None:
            st.warning("⚠️ Model forecaster or metadata artifact not found. Please run training pipeline.")
        elif forecast_res is None:
            st.warning("⚠️ Forecast could not be generated from latest features.")
        else:
            latest_features = df.iloc[[-1]]

            origin_date = forecast_res["forecast_origin_date"]
            target_date = forecast_res["forecast_target_date"]
            pred_val = forecast_res["forecast_inflow_cusecs"]
            lower_val = forecast_res["interval_lower_cusecs"]
            upper_val = forecast_res["interval_upper_cusecs"]
            persistence_val = forecast_res["persistence_baseline_cusecs"]

            if forecast_res["is_stale"]:
                st.error(
                    f"🛑 **HISTORICAL EVALUATION ONLY (ARCHIVE IS {forecast_res['data_age_days']} DAYS STALE):**  \n"
                    f"- Forecast Origin Date: **`{origin_date}`**  \n"
                    f"- Forecast Target Date: **`{target_date}`** (Historical target, NOT tomorrow)  \n"
                    f"- *{forecast_res['freshness_note']}*"
                )
            else:
                st.success(f"Forecast Origin: `{origin_date}` &middot; Target Date: `{target_date}`")

            # Domain boundary warnings
            for w in forecast_res.get("domain_warnings", []):
                st.warning(f"⚠️ **Domain Notice:** {w}")

            c_fc1, c_fc2, c_fc3, c_fc4 = st.columns(4)
            label_prefix = "[HISTORICAL]" if forecast_res["is_stale"] else "[PREDICTED]"
            c_fc1.metric(
                label=f"Next-Day Forecast {label_prefix}",
                value=f"{pred_val:,.0f} cfs",
                delta=f"{pred_val - persistence_val:,.0f} cfs vs origin",
            )
            c_fc2.metric(
                label="80% Prediction Interval",
                value=f"{lower_val:,.0f} - {upper_val:,.0f} cfs",
                help="Empirical residual quantiles (10th and 90th percentiles) calibrated on 2023–2024 validation data",
            )
            c_fc3.metric(
                label="Persistence Baseline",
                value=f"{persistence_val:,.0f} cfs",
                help="Assumes target inflow equals latest observed inflow",
            )
            recent_inflow_mean = df["inflow_cusecs"].tail(7).mean()
            c_fc4.metric(
                label="7-Day Moving Average",
                value=f"{recent_inflow_mean:,.0f} cfs",
                help="Trailing 7-day average of observed inflows",
            )

            # Context plot
            recent_df = df.tail(30).copy()
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=recent_df["calendar_date"],
                y=recent_df["inflow_cusecs"],
                name="Observed Inflow",
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
            ))
            fig_fc.add_trace(go.Scatter(
                x=[pd.to_datetime(target_date)],
                y=[pred_val],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[upper_val - pred_val],
                    arrayminus=[pred_val - lower_val],
                    color="#d62728",
                    width=6,
                    thickness=2.5,
                ),
                name=f"Forecast ({target_date})",
                mode="markers",
                marker=dict(color="#d62728", size=14, symbol="star"),
            ))
            fig_fc.update_layout(
                title=f"30-Day Trailing Observed Inflow and Forecast Target ({target_date})",
                xaxis_title="Date",
                yaxis_title="Inflow Rate (cusecs)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            st.caption(
                "⚠️ **Forecasting Scope & Limitations:**  \n"
                "- Model uses strictly backward-looking lag and cyclical features (zero target leakage).  \n"
                "- Upstream radar telemetry and localized rainfall feeds are not currently integrated.  \n"
                "- This prototype is an academic model; never use for official dam gate dispatch or emergency flood warnings."
            )

    # -------------------------------------------------------------
    # TAB 4: Unusual Pattern Flags & Balance Diagnostics
    # -------------------------------------------------------------
    with tabs[3]:
        st.subheader("Data-Quality Flags & Water-Balance Diagnostics [AUDITED]")
        st.markdown(
            "Distinct inspection rules scan the historical archive.  \n"
            "> [!IMPORTANT]  \n"
            "> **Discrepancy is NOT proof of leakage.** Departures from simplified daily mass balances reflect "
            "unmeasured catchment runoff, rainfall, timing lags, or gauge calibration factors."
        )

        col_dq1, col_dq2 = st.columns(2)
        with col_dq1:
            st.markdown("#### 1. Input Data-Quality Anomalies")
            st.write("- **Reservoir Level Typo:** 1 record on `1992-08-28` (`14,782.0 ft` vs FRL ~2,482 ft MSL). Flagged and raw record preserved.")
            st.write("- **Design Capacity Exceedance:** 5 records (1994, 1999, 2011) exceed design gross capacity (49.452 TMC), peaking at `69.322 TMC` on `2011-02-14`. Preserved without arbitrary clipping.")
            st.write("- **Negative Values:** None detected in flow series.")
            st.write("- **Missing Calendar Date:** Exactly 1 calendar date missing across 36 years (`2020-02-11`). Preserved without synthetic interpolation.")

        with col_dq2:
            st.markdown("#### 2. Water-Balance Completeness Status")
            st.write("- **Status:** `UNVERIFIED_EVAPORATION` (or `INCOMPLETE`)")
            st.write("- **Evaporation Policy:** Evaporation measurement convention is unverified at source (correlation: 0.5108 vs cumulative delta). Missing evaporation is unknown, never zeroed.")
            st.write("- **Discrepancy Policy:** Incomplete intervals never receive a numeric discrepancy. A diagnostic balance excluding evaporation is computed for engineering analysis only.")

        st.markdown("---")
        st.subheader("Historical Water-Balance Audit Summary")
        if WATER_BALANCE_AUDIT_PATH.exists():
            audit_df = pd.read_csv(WATER_BALANCE_AUDIT_PATH)
            st.write(f"- Total Evaluated Intervals: `{len(audit_df):,}`")
            st.dataframe(audit_df["status"].value_counts().reset_index().rename(columns={"index": "Status", "count": "Interval Count"}), use_container_width=True)

            st.markdown("#### Recent 90 Days Diagnostic Discrepancy (Excluding Evaporation)")
            audit_tail = audit_df.tail(90).copy()
            fig_audit = px.bar(
                audit_tail,
                x="end_date",
                y="diagnostic_discrepancy_excluding_evap_tmc",
                title="Recent 90 Days Diagnostic Discrepancy [Observed Storage Change - (Inflow - Outflow - Withdrawal)] in TMC",
                labels={"end_date": "Date", "diagnostic_discrepancy_excluding_evap_tmc": "Diagnostic Discrepancy (TMC)"},
            )
            st.plotly_chart(fig_audit, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: Hypothetical Conservation Scenario Simulator
    # -------------------------------------------------------------
    with tabs[4]:
        st.subheader("Hypothetical Conservation Scenario Simulator [HYPOTHETICAL SIMULATION]")
        st.markdown(
            "Simulate hypothetical operational adjustments (e.g. canal rationing during dry spells) "
            "using consistent physical units (**TMC** and **cusecs**).  \n"
            "*(This is a hypothetical calculation tool only; it contains no automated recommendation engine).*"
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("##### Simulation Parameters")
            sim_days = st.slider("Simulation Horizon (Days)", min_value=1, max_value=30, value=7)
            ration_pct = st.slider("Canal Withdrawal Reduction (%)", min_value=0, max_value=50, value=15, step=5)
            curr_storage = st.number_input(
                "Starting Gross Storage (TMC)",
                min_value=0.0,
                max_value=float(KRS_DESIGN_GROSS_CAPACITY_TMC),
                value=float(round(latest_row["gross_capacity_tmc"], 2)),
            )

        with col_s2:
            st.markdown("##### Boundary Inflows & Releases")
            sim_inflow = st.number_input(
                "Assumed Inflow Rate (cusecs)",
                min_value=0.0,
                value=float(round(latest_row["inflow_cusecs"], 0)),
            )
            sim_outflow = st.number_input(
                "Assumed River Outflow Rate (cusecs)",
                min_value=0.0,
                value=float(round(latest_row["outflow_cusecs"], 0)),
            )
            sim_withdrawal = st.number_input(
                "Baseline Canal Withdrawal Rate (cusecs)",
                min_value=0.0,
                value=float(round(latest_row["withdrawal_cusecs"], 0)),
            )

        res = simulate_conservation_scenario(
            current_storage_tmc=curr_storage,
            assumed_inflow_cusecs=sim_inflow,
            planned_outflow_cusecs=sim_outflow,
            baseline_withdrawal_cusecs=sim_withdrawal,
            reduction_percentage=float(ration_pct),
            simulation_days=sim_days,
        )

        st.markdown("---")
        st.subheader("Scenario Results [HYPOTHETICAL]")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Water Conserved (Volume)", f"{res.water_saved_tmc:.4f} TMC")
        r2.metric("Savings vs Reservoir Capacity", f"{res.savings_as_pct_capacity:.2f}%")
        r3.metric("Projected Baseline Storage", f"{res.projected_baseline_storage_tmc:.2f} TMC")
        r4.metric("Projected Scenario Storage", f"{res.projected_scenario_storage_tmc:.2f} TMC")

        st.caption(f"🛡️ **Disclaimer:** {res.disclaimer}")

    # -------------------------------------------------------------
    # TAB 6: Data Quality & MLOps Status
    # -------------------------------------------------------------
    with tabs[5]:
        st.subheader("Data Quality, Provenance, & MLOps Reproducibility Status")

        st.markdown("#### 1. Unit Conversion Validation Summary")
        if CONVERSION_REPORT_PATH.exists():
            conv_df = pd.read_csv(CONVERSION_REPORT_PATH)
            st.dataframe(conv_df[["flow_type", "correlation", "validation_status", "engineering_note"]], use_container_width=True)

        st.markdown("---")
        st.markdown("#### 2. Model Evaluation Benchmarks (Chronological Held-Out Test Split)")
        if forecaster and forecaster.metadata:
            metrics = forecaster.metadata.get("metrics", {})
            m_rf = metrics.get("test", {}).get("selected_model", {})
            m_base = metrics.get("test", {}).get("persistence", {})
            split = forecaster.metadata.get("split", {})

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.write("**Evaluation Partitioning:**")
                st.write(f"- Test Date Range: `{split.get('test_range', ['N/A'])[0]}` to `{split.get('test_range', ['N/A'])[1]}`")
                st.write(f"- Held-Out Samples: `{split.get('test_samples', 'N/A')}` days")
                st.write(f"- Model Selection Split: Validation set (2023–2024), never test set")

            with col_m2:
                eval_table = pd.DataFrame([
                    {"Model": "Persistence Baseline (inflow_t)", "MAE (cfs)": m_base.get("mae"), "RMSE (cfs)": m_base.get("rmse"), "R²": m_base.get("r2")},
                    {"Model": "Tuned Random Forest", "MAE (cfs)": m_rf.get("mae"), "RMSE (cfs)": m_rf.get("rmse"), "R²": m_rf.get("r2")},
                ])
                st.table(eval_table)

        st.markdown("---")
        st.markdown("#### 3. MLOps Workflow Status")
        st.write("- **DVC Stages:** `validate`, `validate_conversions`, `explore`, `preprocess`, `train` defined in `dvc.yaml`.")
        st.write("- **DVC Remote:** None configured (local cache only; no remote credentials provided).")
        st.write("- **MLflow:** Tracking URI `file:./mlruns`, Experiment `dam_inflow_forecasting`.")
        st.write("- **Rainfall Pipeline:** `BLOCKED (EXTERNAL SOURCE ABSENT)` &middot; No external rainfall dataset or telemetry feed present in repository.")

    # -------------------------------------------------------------
    # TAB 7: Water Conservation Assistant (Layer B)
    # -------------------------------------------------------------
    with tabs[6]:
        st.subheader("🛡️ Water Conservation Decision-Support Assistant")
        st.caption("Academic prototype decision-support rules. Not a live dam-control or official allocation system.")

        # 1. Freshness & Observation Banner
        cond_assessment = assess_reservoir_condition(df)
        st.info(
            f"📅 **Observation Anchor:** `{cond_assessment.observation_date}` "
            f"({cond_assessment.data_age_days} days elapsed &middot; **Status: {cond_assessment.data_freshness}**). "
            "All recommendations reflect historical records; this system does NOT ingest live telemetry."
        )
        if (
            cond_assessment.storage_measurement_date
            and cond_assessment.storage_measurement_date != cond_assessment.observation_date
        ):
            st.warning(
                f"⚠️ **Note on Measurement Date:** Observation record anchor is `{cond_assessment.observation_date}`, "
                f"but the latest valid storage measurement was recorded on `{cond_assessment.storage_measurement_date}`."
            )

        # 2. Reservoir Condition Assessment
        st.markdown("#### 1. Reservoir Condition Assessment")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("Provisional Condition", cond_assessment.condition)
        c_col2.metric("Live Storage (TMC)", f"{cond_assessment.storage_value_tmc:.2f}" if cond_assessment.storage_value_tmc is not None else "N/A")
        c_col3.metric("Storage Trajectory", cond_assessment.storage_trend)
        c_col4.metric("Inflow Trajectory", cond_assessment.inflow_trend)

        with st.expander("Condition Evidence & Limitations", expanded=False):
            st.json(cond_assessment.evidence)
            for lim in cond_assessment.limitations:
                st.caption(f"&bull; {lim}")

        st.markdown("---")

        # 3. Shortage Risk Alerts
        st.markdown("#### 2. Shortage Risk Alerts")
        forecast_inflow = None
        target_dt = None
        if forecaster and forecast_res and not forecast_res.get("is_stale", True):
            forecast_inflow = forecast_res.get("forecast_inflow_cusecs")
            target_dt = forecast_res.get("forecast_target_date")

        risk_assessment = assess_water_shortage_risk(
            cond_assessment,
            inflow_forecast_cfs=forecast_inflow,
            forecast_target_date=target_dt,
        )

        risk_color = {
            "HIGH": "red",
            "MEDIUM": "orange",
            "LOW": "green",
            "INSUFFICIENT_DATA": "gray",
        }.get(risk_assessment.risk_level, "gray")

        st.markdown(f"**Assessed Potential Risk Level:** :{risk_color}[**{risk_assessment.risk_level}**]")
        if risk_assessment.risk_reasons:
            for r in risk_assessment.risk_reasons:
                st.write(f"- {r}")
        st.caption(f"**Recommended Review Action:** {risk_assessment.suggested_review_action}")

        st.markdown("---")

        # 4. Abnormal Conditions & Water Balance Review
        st.markdown("#### 3. Flow Anomalies & Water-Balance Diagnostics")
        flow_anomalies = detect_abnormal_flow_patterns(df)
        if flow_anomalies:
            st.warning(f"Found {len(flow_anomalies)} statistical flow flag(s) in recent window for review.")
            anom_data = [
                {
                    "Metric": a.metric,
                    "Date": a.date,
                    "Observed (cusecs)": a.observed_value,
                    "Historical Median (cusecs)": a.baseline_median,
                    "Type": a.flag_type,
                    "Details": a.reason,
                }
                for a in flow_anomalies
            ]
            st.dataframe(pd.DataFrame(anom_data), use_container_width=True)
        else:
            st.success("No statistical flow anomalies detected in the trailing observation window.")

        st.info("ℹ️ **Notice:** Statistical deviations and water-balance discrepancies indicate simplified accounting residuals or unmodeled tributary flow; they DO NOT confirm physical leakage.")

        # Water-balance interval diagnostic for recommendations
        wb_status = None
        wb_disc = None
        if len(df) >= 2:
            row_prev = df.iloc[-2]
            row_curr = df.iloc[-1]
            wb_calc = compute_water_balance_interval(
                start_date=row_prev["calendar_date"],
                end_date=row_curr["calendar_date"],
                start_storage_tmc=row_prev.get("live_above_cill_tmc"),
                end_storage_tmc=row_curr.get("live_above_cill_tmc"),
                inflow_cusecs=row_curr.get("inflow_cusecs"),
                river_outflow_cusecs=row_curr.get("outflow_cusecs"),
                canal_withdrawal_cusecs=row_curr.get("withdrawal_cusecs"),
                evaporation_cusecs=row_curr.get("evaporation_cusecs"),
                storage_basis="live_above_cill_tmc",
                expected_days=1,
                evaporation_verified=False,
            )
            wb_status = wb_calc.status
            if wb_calc.status == "COMPLETE" and wb_calc.balance_discrepancy_tmc is not None:
                wb_disc = wb_calc.balance_discrepancy_tmc
            else:
                wb_disc = None

        st.markdown("---")

        # 5. Explainable Advisory Recommendations
        st.markdown("#### 4. Explainable Conservation Recommendations")
        recs = generate_conservation_recommendations(
            condition=cond_assessment,
            risk=risk_assessment,
            anomalies=flow_anomalies,
            water_balance_status=wb_status,
            water_balance_discrepancy_tmc=wb_disc,
        )

        for rec in recs:
            prio_badge = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟢 LOW"}.get(rec.priority, rec.priority)
            with st.container():
                st.markdown(f"**{prio_badge} &middot; {rec.title}**")
                st.write(f"**Trigger:** {rec.trigger}")
                st.write(f"**Explanation:** {rec.explanation}")
                st.write(f"**Suggested Action:** {rec.suggested_action}")
                with st.expander("Evidence & Limitations"):
                    st.json(rec.evidence)
                    for l in rec.limitations:
                        st.caption(f"- {l}")
                st.markdown("")

        st.markdown("---")

        # 6. Conservation Scenario Comparisons (5%, 10%, 15%, 20%)
        st.markdown("#### 5. Hypothetical Withdrawal Rationing Scenarios")
        st.write("Evaluate potential water savings across standardized canal withdrawal reductions (5%, 10%, 15%, 20%).")

        if cond_assessment.storage_value_tmc is None:
            st.warning("⚠️ Live storage value is unavailable; hypothetical scenarios cannot be initiated without a valid initial storage volume.")
        else:
            sc_inflow = float(latest_row.get("inflow_cusecs", 2000.0))
            sc_outflow = float(latest_row.get("outflow_cusecs", 1000.0))
            sc_with = float(latest_row.get("withdrawal_cusecs", 2500.0))
            sc_storage = float(cond_assessment.storage_value_tmc)

            sim_scenarios = compare_conservation_scenarios(
                current_storage_tmc=sc_storage,
                assumed_inflow_cusecs=sc_inflow,
                planned_outflow_cusecs=sc_outflow,
                baseline_withdrawal_cusecs=sc_with,
                reductions=[5.0, 10.0, 15.0, 20.0],
                simulation_days=7,
                evaporation_cusecs=0.0,
                storage_basis=cond_assessment.storage_basis,
            )

            sc_table = []
            for s in sim_scenarios:
                sc_table.append({
                    "Scenario": s.scenario_name,
                    "Canal Withdrawal (cusecs)": round(s.conservation_withdrawal_cusecs, 1),
                    "Daily Saved (TMC/day)": round(cusec_day_to_tmc(s.baseline_withdrawal_cusecs - s.conservation_withdrawal_cusecs), 5),
                    "7-Day Conserved (TMC)": s.water_saved_tmc,
                    "Raw Projected Storage (TMC)": s.raw_projected_scenario_tmc,
                    "Bounds Flag": "OUT_OF_BOUNDS" if s.scenario_exceeds_bounds else "NORMAL",
                })
            st.dataframe(pd.DataFrame(sc_table), use_container_width=True)
            st.caption(
                f"Assumptions: 7-day constant flow; storage basis '{cond_assessment.storage_basis}' bounded against "
                f"reference capacity of {KRS_DESIGN_LIVE_CAPACITY_TMC} TMC; unverified evaporation excluded (0.0 TMC); hypothetical non-operational simulation."
            )


if __name__ == "__main__":
    main()
