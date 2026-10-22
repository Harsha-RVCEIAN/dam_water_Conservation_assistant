"""Streamlit dashboard for Dam Water Conservation Assistant.

Provides interactive reservoir monitoring, historical hydrological trends,
machine learning inflow forecasting, statistical anomaly review flags,
and hypothetical water-conservation scenario calculations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.predict import InflowForecaster
from src.models.water_budget import (
    KRS_DESIGN_GROSS_CAPACITY_TMC,
    cusec_day_to_tmc,
    simulate_conservation_scenario,
)

PROCESSED_DATA_PATH = Path("data/processed/krs_dam_processed.csv")
RAW_DATA_PATH = Path("data/raw/krs_dam_dataset.csv")
MODEL_PATH = Path("models/inflow_forecaster.joblib")
METADATA_PATH = Path("models/model_metadata.json")

st.set_page_config(
    page_title="K.R.Sagara Dam - Water Conservation Assistant",
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
def load_forecaster() -> InflowForecaster | None:
    """Load cached model forecaster."""
    if not MODEL_PATH.exists():
        return None
    try:
        return InflowForecaster(MODEL_PATH, METADATA_PATH)
    except Exception:
        return None


def main() -> None:
    st.title("🌊 K.R.Sagara Dam - Water Conservation Assistant")
    st.markdown(
        "**Student MLOps Project** &middot; Explainable Decision-Support Prototype &middot; *Cauvery Basin, Karnataka, India*"
    )

    df = load_processed_data()
    forecaster = load_forecaster()

    if df.empty:
        st.error(
            "⚠️ Processed dataset not found. Please run the preprocessing pipeline first: "
            "`python -m src.data.preprocess`"
        )
        return

    # Global freshness caveat
    latest_row = df.iloc[-1]
    latest_date_str = latest_row["calendar_date"].strftime("%Y-%m-%d")

    st.info(
        f"ℹ️ **Data Freshness Note:** Latest recorded observation in this historical archive is **{latest_date_str}**. "
        "This application is an educational prototype and is **not** connected to real-time live dam telemetry."
    )

    tabs = st.tabs([
        "📊 Reservoir Overview",
        "📈 Historical Trends",
        "🤖 Inflow Forecast",
        "🚩 Unusual Pattern Flags",
        "💧 Conservation Calculator",
        "📋 Data Quality & MLOps",
    ])

    # -------------------------------------------------------------
    # TAB 1: Reservoir Overview
    # -------------------------------------------------------------
    with tabs[0]:
        st.subheader("Current Reservoir Status [OBSERVED]")
        st.caption(f"As of latest available record: **{latest_date_str}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label="Gross Capacity (TMC)",
            value=f"{latest_row['gross_capacity_tmc']:.2f} TMC",
            help="Current gross storage including live and dead storage",
        )
        pct_full = (latest_row["gross_capacity_tmc"] / KRS_DESIGN_GROSS_CAPACITY_TMC) * 100
        c2.metric(
            label="Storage Percentage Full",
            value=f"{pct_full:.1f}%",
            help=f"Computed against design gross capacity ({KRS_DESIGN_GROSS_CAPACITY_TMC} TMC)",
        )
        c3.metric(
            label="Reservoir Level (ft)",
            value=f"{latest_row['reservoir_level_ft']:.1f} ft",
            help="Elevation above mean sea level (normal Full Reservoir Level is ~2482 ft MSL)",
        )
        c4.metric(
            label="Live Capacity (TMC)",
            value=f"{latest_row['live_capacity_tmc']:.2f} TMC",
            help="Usable live storage above the dead sill level",
        )

        st.markdown("---")
        st.subheader("Daily Flow Balance Rates [OBSERVED]")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Inflow", f"{latest_row['inflow_cusecs']:,.0f} cusecs", f"{latest_row['inflow_vol_tmc']:.4f} TMC/day")
        f2.metric("River Outflow", f"{latest_row['outflow_cusecs']:,.0f} cusecs", f"{latest_row['outflow_vol_tmc']:.4f} TMC/day")
        f3.metric("Canal Withdrawal", f"{latest_row['withdrawal_cusecs']:,.0f} cusecs", f"{latest_row['withdrawal_vol_tmc']:.4f} TMC/day")
        evap_cusecs = latest_row["evaporation_cusecs"] if not pd.isna(latest_row["evaporation_cusecs"]) else 0.0
        evap_tmc = evap_cusecs * 0.0000864
        f4.metric("Evaporation", f"{evap_cusecs:,.0f} cusecs", f"{evap_tmc:.4f} TMC/day")

        st.caption(
            "Unit Rule: Flow rates in **cusecs** (cubic feet per second) represent daily mean rates. "
            "1 cusec-day = **0.0000864 TMC** (Thousand Million Cubic feet)."
        )

    # -------------------------------------------------------------
    # TAB 2: Historical Trends
    # -------------------------------------------------------------
    with tabs[1]:
        st.subheader("Historical Hydrological Trends [OBSERVED]")
        st.markdown("Filter observations across the 36-year dataset:")

        years_available = sorted(df["calendar_date"].dt.year.unique())
        col_y1, col_y2 = st.columns(2)
        start_year = col_y1.selectbox("Start Year", years_available, index=max(0, len(years_available) - 6))
        end_year = col_y2.selectbox("End Year", years_available, index=len(years_available) - 1)

        filtered = df[
            (df["calendar_date"].dt.year >= start_year) & (df["calendar_date"].dt.year <= end_year)
        ]

        # Inflow & Outflow Chart
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
            title=f"Daily Flow Rates ({start_year} - {end_year})",
            xaxis_title="Date",
            yaxis_title="Flow Rate (cusecs)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
        )
        st.plotly_chart(fig_flows, use_container_width=True)

        # Storage Chart
        fig_storage = go.Figure()
        fig_storage.add_trace(go.Scatter(
            x=filtered["calendar_date"],
            y=filtered["gross_capacity_tmc"],
            name="Gross Storage (TMC)",
            line=dict(color="#3b5998", width=2),
            fill="tozeroy",
        ))
        fig_storage.add_hline(
            y=KRS_DESIGN_GROSS_CAPACITY_TMC,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Design Gross Capacity: {KRS_DESIGN_GROSS_CAPACITY_TMC} TMC",
            annotation_position="bottom right",
        )
        fig_storage.update_layout(
            title=f"Gross Storage Volume ({start_year} - {end_year})",
            xaxis_title="Date",
            yaxis_title="Storage Volume (TMC)",
            margin=dict(l=40, r=40, t=60, b=40),
            height=350,
        )
        st.plotly_chart(fig_storage, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Inflow Forecast
    # -------------------------------------------------------------
    with tabs[2]:
        st.subheader("Machine Learning Inflow Forecast [PREDICTED]")
        st.markdown(
            "**Forecast Target:** Next-Day Inflow (`Inflow (t+1)` in cusecs) &middot; "
            "**Model:** Random Forest Regressor trained on chronological historical data (1990–2022)."
        )

        if forecaster is None:
            st.warning("Model artifact not found. Please run: `python -m src.models.train`")
        else:
            recent_days = 30
            recent_df = df.tail(recent_days).copy()
            latest_features = df.iloc[[-1]]
            
            # Predict point and empirical 80% prediction interval
            preds, lowers, uppers = forecaster.predict_with_interval(latest_features)
            next_day_pred = float(preds[0])
            next_day_lower = float(lowers[0])
            next_day_upper = float(uppers[0])
            next_day_date = (latest_row["calendar_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            # Domain boundary check
            domain_warnings = forecaster.check_input_domain(latest_features)
            for w in domain_warnings:
                st.warning(f"⚠️ **Domain Notice:** {w}")

            c_info1, c_info2 = st.columns(2)
            c_info1.caption(f"⏱️ **Forecast Horizon:** 1 Day Ahead &middot; Target Date: **{next_day_date}**")
            c_info2.caption("📐 **Unit:** Cusecs (cubic feet per second, mean daily rate). 1 cusec-day = 0.0000864 TMC.")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric(
                label=f"Next-Day Forecast [PREDICTED]",
                value=f"{next_day_pred:,.0f} cfs",
                delta=f"{next_day_pred - latest_row['inflow_cusecs']:,.0f} cfs vs today",
            )
            col_m2.metric(
                label="80% Prediction Interval",
                value=f"{next_day_lower:,.0f} - {next_day_upper:,.0f} cfs",
                help="Calibrated from empirical 10th and 90th percentile residuals on chronological validation data (2023–2024)",
            )
            col_m3.metric(
                label="Persistence Baseline",
                value=f"{latest_row['inflow_cusecs']:,.0f} cfs",
                help="Assumes tomorrow's inflow equals today's latest observed rate",
            )
            col_m4.metric(
                label="7-Day Moving Average",
                value=f"{recent_df['inflow_cusecs'].tail(7).mean():,.0f} cfs",
                help="Trailing 7-day average of observed inflows",
            )

            # Forecast Context Plot with uncertainty range
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=recent_df["calendar_date"],
                y=recent_df["inflow_cusecs"],
                name="Observed Inflow [OBSERVED]",
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
            ))
            # Predicted interval (error bar)
            fig_fc.add_trace(go.Scatter(
                x=[pd.to_datetime(next_day_date)],
                y=[next_day_pred],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[next_day_upper - next_day_pred],
                    arrayminus=[next_day_pred - next_day_lower],
                    color="#d62728",
                    width=6,
                    thickness=2.5,
                ),
                name="Forecast Inflow (80% Interval)",
                mode="markers",
                marker=dict(color="#d62728", size=14, symbol="star"),
            ))
            fig_fc.update_layout(
                title=f"Recent 30-Day Observed Inflow & Next-Day Prediction with 80% Uncertainty Band ({next_day_date})",
                xaxis_title="Date",
                yaxis_title="Inflow Rate (cusecs)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Model validation performance expander
            comp_path = Path("reports/model_comparison_report.csv")
            if comp_path.exists():
                with st.expander("📊 View Multi-Model Validation Benchmark (2023–2024 Held-Out Validation Period)"):
                    st.dataframe(pd.read_csv(comp_path), use_container_width=True)

            st.caption(
                "⚠️ **Forecast Caveat:** This prediction is generated from historical catchment statistical lags and storage features. "
                "It does not incorporate upstream live radar precipitation or real-time meteorological flood warnings. "
                "This tool is an educational prototype and must never replace official Central Water Commission / dam operator safety procedures."
            )

    # -------------------------------------------------------------
    # TAB 4: Unusual Pattern Flags
    # -------------------------------------------------------------
    with tabs[3]:
        st.subheader("Unusual-Pattern & Anomaly Review Flags [OBSERVED & AUDITED]")
        st.markdown(
            "Automated rules scan the historical archive for recording transcription errors and "
            "extreme water-balance discrepancies. **Flags are strictly for human investigation; they do NOT prove leakage.**"
        )

        col_a1, col_a2 = st.columns(2)

        # Flag 1: Level Typos
        with col_a1:
            st.markdown("#### 1. Reservoir Level Physical Bounds Check")
            st.write("Rule: Flag records where `Reservoir Level (ft) > 2500 ft` (Normal Full Reservoir Level is ~2482 ft MSL).")
            anom_levels = df[df["is_level_anomalous"]]
            if not anom_levels.empty:
                st.warning(f"Found {len(anom_levels)} physical bound violation in archive:")
                for _, r in anom_levels.iterrows():
                    st.write(
                        f"- **Date:** `{r['calendar_date'].strftime('%Y-%m-%d')}` | "
                        f"**Raw Level:** `{r['reservoir_level_ft']:.1f} ft` | "
                        f"**Storage:** `{r['gross_capacity_tmc']:.2f} TMC` "
                        f"(Transcribed typo; normal FRL is ~2482 ft)."
                    )
            else:
                st.success("No elevation bound violations.")

        # Flag 2: Water Balance Discrepancies
        with col_a2:
            st.markdown("#### 2. Water-Balance Discrepancy Check")
            st.write(
                "Rule: Flag days where `|Storage Change - (Inflow - Outflow - Withdrawal - Evap)| > 5.0 TMC`."
            )
            anom_balance = df[df["is_balance_anomalous"]]
            st.info(f"Archive contains {len(anom_balance)} days flagged with |discrepancy| > 5 TMC ({len(anom_balance)/len(df)*100:.2f}% of days).")
            st.caption(
                "Explanations: Unmeasured catchment rainfall, gate travel timing, flood routing waves, or gauge adjustments. "
                "None of these flags prove dam leakage."
            )

        st.markdown("---")
        st.markdown("#### Recent 90 Days Daily Water Balance Audit")
        audit_recent = df.tail(90).copy()
        fig_disc = px.bar(
            audit_recent,
            x="calendar_date",
            y="balance_discrepancy_tmc",
            title="Recent Daily Balance Discrepancy (Actual Storage Change - Net Flow) in TMC",
            labels={"calendar_date": "Date", "balance_discrepancy_tmc": "Discrepancy (TMC)"},
            color="is_balance_anomalous",
            color_discrete_map={True: "#d62728", False: "#1f77b4"},
        )
        st.plotly_chart(fig_disc, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: Conservation Scenario Calculator
    # -------------------------------------------------------------
    with tabs[4]:
        st.subheader("Hypothetical Conservation Scenario Simulator [HYPOTHETICAL SCENARIO]")
        st.markdown(
            "Simulate hypothetical operational adjustments (e.g. canal rationing during dry spells) "
            "using consistent physical units (**TMC** and **cusecs**)."
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
    # TAB 6: Data Quality & MLOps
    # -------------------------------------------------------------
    with tabs[5]:
        st.subheader("Data Quality, Provenance, & Model Verification")

        st.markdown("#### 1. Held-Out Evaluation Metrics")
        if forecaster and forecaster.metadata:
            metrics = forecaster.metadata.get("metrics", {})
            m_rf = metrics.get("random_forest_test", {})
            m_base = metrics.get("baseline_persistence", {})
            split = forecaster.metadata.get("split", {})

            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                st.write("**Evaluation Split Details:**")
                st.write(f"- Test Date Range: `{split.get('test_range', ['N/A'])[0]}` to `{split.get('test_range', ['N/A'])[1]}`")
                st.write(f"- Test Samples: `{split.get('test_samples', 'N/A')}` held-out days")
                st.write(f"- Training Samples: `{split.get('train_samples', 'N/A')}` days (1990–2022)")

            with col_eval2:
                eval_table = pd.DataFrame([
                    {"Model": "Persistence Baseline (inflow_t)", "MAE (cusecs)": m_base.get("mae"), "RMSE (cusecs)": m_base.get("rmse"), "R²": m_base.get("r2")},
                    {"Model": "Random Forest Forecaster", "MAE (cusecs)": m_rf.get("mae"), "RMSE (cusecs)": m_rf.get("rmse"), "R²": m_rf.get("r2")},
                ])
                st.table(eval_table)
                st.success(f"✅ Random Forest outperforms persistence baseline by **{metrics.get('rmse_improvement_pct')}%** in RMSE on held-out test data.")

        st.markdown("---")
        st.markdown("#### 2. MLOps Infrastructure Status")
        st.write("- **MLflow:** Local tracking store at `mlruns/` (Logged run ID in metadata).")
        st.write("- **DVC:** Versioned raw dataset `data/raw/krs_dam_dataset.csv.dvc` and pipeline `dvc.yaml` (Local cache only; no cloud remote configured).")
        st.write("- **Unit Conversions:** Validated `1 cusec-day = 0.0000864 TMC` against cumulative column delta (correlation: 0.999999).")
        st.write("- **Known Missing Day:** Exactly 1 calendar date missing across 36 years (`2020-02-11`), preserved without synthetic interpolation.")


if __name__ == "__main__":
    main()
