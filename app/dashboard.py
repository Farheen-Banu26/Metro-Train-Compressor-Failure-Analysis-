"""
Interactive Streamlit Dashboard for Metro Train Compressor Failure Analysis.
Provides telemetry monitoring, EDA visualization viewer, ground-truth failure event inspection,
and real-time anomaly detection simulation.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DATA_DIR,
    FIGURES_EDA_DIR,
    FIGURES_EVAL_DIR,
    MODELS_DIR,
    FAILURE_EVENTS,
    ALL_FEATURES,
    ANALOG_FEATURES,
    DIGITAL_FEATURES,
    TIMESTAMP_COL,
    WINDOW_SIZE,
)
from src.inference import (
    load_inference_artifacts,
    validate_input_dataframe,
    run_inference,
    get_historical_sample,
    get_csv_template,
)

st.set_page_config(
    page_title="MetroPT-3 APU Compressor Anomaly Detection",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚇 Metro Train Compressor Failure Analysis Dashboard")
st.markdown(
    """
    **Course:** U21ADP05 — Exploratory Data Analysis and Visualization  
    **Dataset:** Official UCI MetroPT-3 Dataset (ID: 791) | Air Production Unit (APU) Telemetry  
    **Architecture:** Semi-Supervised Temporal LSTM Autoencoder
    """
)

# Sidebar Navigation
sidebar_nav = st.sidebar.radio(
    "Navigation",
    [
        "1. Executive Overview & Metrics",
        "2. Ground-Truth Failure Events",
        "3. Exploratory Data Analysis (12 Figures)",
        "4. Deep Learning Model Evaluation",
        "5. Telemetry Inspector",
        "6. Live Compressor Anomaly Prediction"
    ]
)

def render_inference_results(results_df: pd.DataFrame, summary: dict, input_df: pd.DataFrame):
    """
    Renders status banner, metrics cards, comparison chart, prediction table,
    and download button for live inference.
    """
    is_anom = summary["overall_status"] == "ANOMALY DETECTED"

    # 1. Main Status Banner
    if is_anom:
        st.error(
            f"🚨 **STATUS: ANOMALY DETECTED** — Telemetry reconstruction error ({summary['mean_reconstruction_error']:.4f}) "
            f"exceeds the healthy baseline threshold (τ = {summary['threshold']:.6f})."
        )
    else:
        st.success(
            f"✅ **STATUS: NORMAL OPERATION** — Telemetry sequence reconstructs cleanly within the healthy baseline manifold "
            f"(MSE = {summary['mean_reconstruction_error']:.4f} ≤ τ = {summary['threshold']:.6f})."
        )

    # 2. Key Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric(
        "Windows Evaluated",
        f"{summary['total_windows']} windows",
        f"{summary['anomalous_windows']} flagged ({summary['anomaly_percentage']:.1f}%)"
    )
    m_col2.metric(
        "Mean Reconstruction Error",
        f"{summary['mean_reconstruction_error']:.6f}",
        f"Max: {summary['max_reconstruction_error']:.6f}"
    )
    m_col3.metric(
        "Decision Threshold (τ)",
        f"{summary['threshold']:.6f}",
        f"Margin: {summary['max_reconstruction_error'] - summary['threshold']:+.4f}"
    )
    m_col4.metric(
        "Top Divergent Sensor",
        summary["primary_contributing_sensor"],
        "Highest Channel MSE"
    )

    st.caption(
        f"⏱️ **Time Range:** `{results_df['start_time'].iloc[0]}` to `{results_df['end_time'].iloc[-1]}` "
        f"({len(input_df)} consecutive readings at 10s intervals = {len(input_df)*10/60:.1f} minutes of operational telemetry)"
    )

    # 3. Visualization
    st.subheader("📈 Telemetry Reconstruction & Sensor Dynamics Timeline")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})

    x_indices = np.arange(1, len(results_df) + 1)
    errors = results_df["reconstruction_error"].values
    tau = summary["threshold"]

    # Ax1: Reconstruction Error vs Threshold
    ax1.plot(x_indices, errors, color="#1f77b4", lw=2, marker="o", markersize=4, label="Reconstruction Error (MSE)")
    ax1.axhline(tau, color="#d62728", ls="--", lw=2, label=f"Anomaly Threshold τ ({tau:.4f})")

    # Highlight anomalous points
    anom_mask = results_df["is_anomaly"].values
    if np.any(anom_mask):
        ax1.scatter(
            x_indices[anom_mask],
            errors[anom_mask],
            color="#d62728",
            s=70,
            zorder=5,
            label="Anomalous Window"
        )
    if np.any(~anom_mask):
        ax1.scatter(
            x_indices[~anom_mask],
            errors[~anom_mask],
            color="#2ca02c",
            s=40,
            zorder=4,
            label="Normal Window"
        )

    ax1.set_ylabel("Reconstruction MSE", fontweight="bold")
    ax1.set_title("Window Reconstruction Error vs. Statistical Anomaly Threshold", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Ax2: Key Sensor Trajectories
    time_pts = np.arange(len(input_df))
    ax2_twin = ax2.twinx()

    p1 = ax2.plot(time_pts, input_df["Reservoirs"], color="#2ca02c", lw=1.5, label="Reservoirs Pressure (bar)")
    p2 = ax2.plot(time_pts, input_df["TP2"], color="#ff7f0e", lw=1.2, ls=":", label="TP2 Discharge (bar)")
    p3 = ax2_twin.plot(time_pts, input_df["Motor_current"], color="#9467bd", lw=1.5, label="Motor Current (A)")

    ax2.set_ylabel("Pressure (bar)", fontweight="bold")
    ax2_twin.set_ylabel("Current (A)", fontweight="bold", color="#9467bd")
    ax2.set_xlabel("Time Step (10-second intervals)", fontweight="bold")

    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 4. Detailed Prediction Windows Table
    with st.expander("📋 View Window-by-Window Prediction Log", expanded=False):
        st.dataframe(results_df, use_container_width=True)

    # 5. Download Button
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Detailed Prediction Results (CSV)",
        data=csv_bytes,
        file_name="metro_compressor_prediction_results.csv",
        mime="text/csv",
    )

    # 6. Industrial Engineering Context & Disclaimer
    st.info(
        "ℹ️ **Operational Anomaly Notice:** Semi-supervised autoencoder reconstruction identifies statistical and thermodynamic "
        "deviations from healthy operating patterns (e.g. continuous motor loading, rapid pressure venting, abnormal heating). "
        "An anomaly alarm indicates that the APU system is experiencing abnormal mechanical or operational stress; "
        "it serves as an actionable early warning for inspection, rather than conclusive proof of permanent mechanical failure."
    )


# -------------------------------------------------------------
# Section 1: Executive Overview & Metrics
# -------------------------------------------------------------
if sidebar_nav == "1. Executive Overview & Metrics":
    st.header("Executive Summary & Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Telemetry Records", "1,516,948")
    col2.metric("Sensor Features", "15 Features")
    col3.metric("Documented Failures", "4 Air Leaks")
    col4.metric("Chronological Window", "Feb - Aug 2020")

    st.subheader("Key Empirical Model Metrics")
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("ROC-AUC", f"{metrics['test_metrics']['roc_auc']:.4f}")
        m_col2.metric("PR-AUC", f"{metrics['test_metrics']['pr_auc']:.4f}")
        m_col3.metric("Recall (Failure Detection)", f"{metrics['test_metrics']['recall']*100:.2f}%")
        m_col4.metric("Precision", f"{metrics['test_metrics']['precision']*100:.2f}%")

        st.info(f"**Anomaly Decision Threshold (τ):** {metrics['threshold']:.6f} ({metrics['threshold_method']})")
    else:
        st.warning("Model evaluation metrics will populate once model training is completed.")

    st.markdown("---")
    st.subheader("Hardware & Physical System Context")
    st.markdown(
        """
        The **Air Production Unit (APU)** supplies compressed air for crucial metro train subsystems, including:
        * **Pneumatic Braking Systems:** Emergency and service air friction brakes.
        * **Secondary Pneumatic Suspension:** Leveling valves maintain constant carriage floor height.
        * **Automatic Passenger Doors:** Pneumatically operated sliding door actuators.
        * **Pantograph Elevation Actuation:** Ensures continuous electrical contact with overhead catenary lines.

        A mechanical air leak causes the compressor to operate under continuous load, driving motor current and lubrication oil temperatures to extreme limits and degrading system reliability.
        """
    )

# -------------------------------------------------------------
# Section 2: Ground-Truth Failure Events
# -------------------------------------------------------------
elif sidebar_nav == "2. Ground-Truth Failure Events":
    st.header("Documented Operator Failure Events & Maintenance Log")
    st.write("These 4 air-leak incidents were officially recorded by the public transit operator and serve as ground-truth benchmarks:")

    events_df = pd.DataFrame(FAILURE_EVENTS)
    st.dataframe(events_df[["id", "name", "start", "end", "failure_type", "severity", "maintenance", "notes"]], use_container_width=True)

    selected_event = st.selectbox("Select Failure Event to Inspect:", [e["name"] for e in FAILURE_EVENTS])
    event_data = next(e for e in FAILURE_EVENTS if e["name"] == selected_event)

    st.markdown(f"### {event_data['name']}: {event_data['failure_type']}")
    st.write(f"- **Time Window:** `{event_data['start']}` to `{event_data['end']}`")
    st.write(f"- **Severity:** `{event_data['severity']}`")
    st.write(f"- **Maintenance Action:** `{event_data['maintenance'] or 'Inspected / Monitored'}`")
    st.write(f"- **Diagnostic Notes:** {event_data['notes']}")

# -------------------------------------------------------------
# Section 3: Exploratory Data Analysis (12 Figures)
# -------------------------------------------------------------
elif sidebar_nav == "3. Exploratory Data Analysis (12 Figures)":
    st.header("Comprehensive Exploratory Data Analysis (12 Figures)")
    st.write("Browse the 12 publication-grade, 300-DPI visualizations generated from the MetroPT-3 dataset:")

    figure_list = [
        ("Figure 01: Telemetry Continuity & Sampling Distribution", "fig01_telemetry_continuity_missingness.png"),
        ("Figure 02: Univariate Distribution Analysis of Analog Sensors", "fig02_univariate_analog_distributions.png"),
        ("Figure 03: Digital Control Signal Duty Cycles & Frequencies", "fig03_digital_signal_frequencies.png"),
        ("Figure 04: Macro 6-Month Multi-Sensor Telemetry Overview", "fig04_macro_telemetry_timeline.png"),
        ("Figure 05: High-Resolution Microscopic APU Pumping Cycle", "fig05_micro_cycle_dynamics.png"),
        ("Figure 06: Pearson & Spearman Rank Correlation Matrices", "fig06_correlation_matrix_pearson_spearman.png"),
        ("Figure 07: Bivariate Phase-Space Couplings", "fig07_phase_space_trajectories.png"),
        ("Figure 08: Operational Regime Boxplots (Loaded vs. Offloaded)", "fig08_operational_regime_boxplots.png"),
        ("Figure 09: Diurnal & Weekly Duty Cycle Heatmap", "fig09_diurnal_weekly_duty_cycle.png"),
        ("Figure 10: Ground-Truth Failure Event #1 Deep Dive", "fig10_failure_event1_air_leak_deep_dive.png"),
        ("Figure 11: Ground-Truth Failure Event #3 Breakdown & Recovery", "fig11_failure_event3_breakdown_and_recovery.png"),
        ("Figure 12: 2D PCA Dimensionality Reduction Projection", "fig12_manifold_pca_tsne_projection.png"),
    ]

    selected_fig_title, selected_fig_file = st.selectbox("Select Figure to View:", figure_list, format_func=lambda x: x[0])
    fig_path = FIGURES_EDA_DIR / selected_fig_file

    if fig_path.exists():
        st.image(str(fig_path), caption=selected_fig_title, use_container_width=True)
    else:
        st.warning(f"Figure file `{selected_fig_file}` is currently being rendered...")

# -------------------------------------------------------------
# Section 4: Deep Learning Model Evaluation
# -------------------------------------------------------------
elif sidebar_nav == "4. Deep Learning Model Evaluation":
    st.header("Deep Learning (LSTM Autoencoder) Evaluation Suite")
    st.write("Visualizations evaluating the semi-supervised temporal anomaly detector against operator failure logs:")

    eval_fig_list = [
        ("Figure 13: LSTM Autoencoder Training & Validation Convergence", "fig13_loss_convergence_curves.png"),
        ("Figure 14: Reconstruction Error Distribution Separation (Healthy vs. Failure)", "fig14_reconstruction_error_distribution.png"),
        ("Figure 15: Continuous Anomaly Score Timeline vs. Dynamic Alarm Threshold", "fig15_anomaly_score_timeline_threshold.png"),
        ("Figure 16: Precision-Recall & Receiver Operating Characteristic Curves", "fig16_precision_recall_roc_curves.png"),
        ("Figure 17: Per-Sensor Error Attribution Breakdown (Explainability)", "fig17_sensor_attribution_breakdown.png"),
    ]

    for title, fname in eval_fig_list:
        st.subheader(title)
        p = FIGURES_EVAL_DIR / fname
        if p.exists():
            st.image(str(p), caption=title, use_container_width=True)
        else:
            st.info(f"Evaluation chart `{fname}` will appear here once model evaluation completes.")

# -------------------------------------------------------------
# Section 5: Telemetry Inspector
# -------------------------------------------------------------
elif sidebar_nav == "5. Telemetry Inspector":
    st.header("Processed Telemetry Inspector")
    resampled_p = PROCESSED_DATA_DIR / "telemetry_resampled.parquet"
    if resampled_p.exists():
        df_preview = pd.read_parquet(resampled_p)
        st.write(f"Displaying sample rows from {len(df_preview):,} resampled telemetry points:")
        st.dataframe(df_preview.head(50), use_container_width=True)
    else:
        st.warning("Processed dataset not yet generated.")


elif sidebar_nav == "6. Live Compressor Anomaly Prediction":
    st.header("⚡ Live Compressor Anomaly Prediction")
    st.markdown(
        """
        Evaluate real multivariate telemetry sequences using the **trained Deep LSTM Autoencoder**.
        The model processes temporal windows of **60 consecutive readings** (10 minutes of operational telemetry at 10-second intervals),
        computes sequence reconstruction error, and compares it against the **healthy validation threshold ($\tau = 2.589630$)**.
        """
    )

    # Load Model Artifacts
    try:
        artifacts = load_inference_artifacts("cpu")
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.stop()

    # Model Specifications Cards
    spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
    spec_col1.metric("Architecture", "LSTM Autoencoder", "15-64-32-16-32-64-15")
    spec_col2.metric("Sequence Length (W)", f"{artifacts['window_size']} steps", "10 min @ 10s")
    spec_col3.metric("Anomaly Threshold (τ)", f"{artifacts['threshold']:.6f}", "Val Quantile 99%")
    spec_col4.metric("Features Required", f"{len(artifacts['features'])} Features", "7 Analog + 8 Digital")

    st.markdown("---")

    # Mode Selection Tabs
    tab_sample, tab_upload, tab_editor, tab_template = st.tabs([
        "🔬 Historical Benchmark Sample",
        "📂 Upload Telemetry CSV",
        "✏️ Interactive Sequence Editor",
        "📥 Download CSV Template & Specs"
    ])

    # ---------------------------------------------------------
    # Tab 1: Historical Benchmark Sample (Live Demo)
    # ---------------------------------------------------------
    with tab_sample:
        st.subheader("Select a Historical Benchmark Sequence")
        st.write(
            "Choose an authentic sample extracted directly from the processed dataset to verify the genuine inference pipeline:"
        )

        sample_options = {
            "normal_baseline": "🟢 Normal Baseline — Healthy Pumping Cycle (April 16)",
            "normal_multicycle": "🟢 Normal Multi-Cycle — 20-Minute Operational Stream (April 16, 120 steps)",
            "event_1_leak": "🔴 Failure Event #1 — Active Air Leak (April 18)",
            "event_2_leak": "🔴 Failure Event #2 — Air Leak Preceding Depot Overhaul (May 30)",
            "event_3_breakdown": "🔴 Failure Event #3 — Major Pneumatic Line Breakdown (June 6)",
            "event_4_pressure_drop": "🔴 Failure Event #4 — Reservoir Pressure Cycling (July 15)",
        }

        selected_sample = st.selectbox(
            "Historical Sample Scenario:",
            options=list(sample_options.keys()),
            format_func=lambda k: sample_options[k]
        )

        raw_sample_df = get_historical_sample(selected_sample)
        st.write(f"**Loaded Telemetry:** `{raw_sample_df.shape[0]} rows × {raw_sample_df.shape[1]} columns` (100% Genuine Historical Data)")

        with st.expander("Preview Sample Telemetry Rows", expanded=False):
            st.dataframe(raw_sample_df.head(10), use_container_width=True)

        if st.button("🚀 Run Live Inference on Historical Sample", key="btn_run_hist"):
            with st.spinner("Executing PyTorch LSTM Autoencoder inference..."):
                valid, msg, clean_df = validate_input_dataframe(raw_sample_df)
                if not valid:
                    st.error(msg)
                else:
                    results_df, summary = run_inference(clean_df, artifacts, stride=1)
                    render_inference_results(results_df, summary, clean_df)

    # ---------------------------------------------------------
    # Tab 2: Upload Telemetry CSV (Method A)
    # ---------------------------------------------------------
    with tab_upload:
        st.subheader("Upload Custom Telemetry CSV")
        st.write("Upload a CSV file containing consecutive timestamped telemetry readings:")
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], key="uploader_csv")

        stride_choice = st.radio(
            "Window Sliding Resolution (Stride):",
            options=[1, 6],
            format_func=lambda s: f"Continuous Stride = {s} (Step every {s*10}s)" if s == 1 else f"Aggregated Stride = {s} (Step every 1 minute)",
            horizontal=True
        )

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.write(f"**Uploaded Telemetry:** `{user_df.shape[0]} rows × {user_df.shape[1]} columns`")

                valid, msg, clean_df = validate_input_dataframe(user_df)
                if not valid:
                    st.error(f"❌ Validation Failed: {msg}")
                else:
                    st.success(f"✅ Telemetry validated successfully: {len(clean_df)} valid timestamped readings ready for inference.")
                    with st.expander("Preview Validated Input Data", expanded=False):
                        st.dataframe(clean_df.head(10), use_container_width=True)

                    if st.button("🚀 Run Live Inference on Uploaded CSV", key="btn_run_upload"):
                        with st.spinner("Processing sequence windows and computing reconstruction errors..."):
                            results_df, summary = run_inference(clean_df, artifacts, stride=stride_choice)
                            render_inference_results(results_df, summary, clean_df)
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    # ---------------------------------------------------------
    # Tab 3: Interactive Sequence Editor & Manual Input (Method B)
    # ---------------------------------------------------------
    with tab_editor:
        st.subheader("Interactive 60-Step Sequence Editor")
        st.markdown(
            """
            **Why 60 time steps?**  
            The trained LSTM autoencoder relies on temporal sequence modeling ($W=60$) to capture compressor loading, 
            pumping ramps, cutoff governor switches, and moisture purge venting over a 10-minute cycle.
            Entering a single row cannot represent dynamic pressure rates-of-change ($\Delta P / \Delta t$).
            
            You can customize and edit individual sensor values below to test the model's reaction to simulated pneumatic leaks, 
            motor overcurrent, or oil overheating:
            """
        )

        # Session state initialization for editor
        if "editor_data" not in st.session_state:
            st.session_state["editor_data"] = get_historical_sample("normal_baseline")

        col_reset1, col_reset2 = st.columns([1, 1])
        with col_reset1:
            if st.button("🔄 Reset Editor to Nominal Baseline (Normal)", key="reset_norm"):
                st.session_state["editor_data"] = get_historical_sample("normal_baseline")
                st.rerun()
        with col_reset2:
            if st.button("⚠️ Load Active Air Leak Scenario (Failure Event #3)", key="load_leak"):
                st.session_state["editor_data"] = get_historical_sample("event_3_breakdown")
                st.rerun()

        edited_df = st.data_editor(
            st.session_state["editor_data"],
            num_rows="fixed",
            use_container_width=True,
            key="data_editor_widget"
        )

        if st.button("🚀 Run Live Inference on Edited Sequence", key="btn_run_editor"):
            valid, msg, clean_df = validate_input_dataframe(edited_df)
            if not valid:
                st.error(msg)
            else:
                with st.spinner("Running PyTorch LSTM Autoencoder inference on edited sequence..."):
                    results_df, summary = run_inference(clean_df, artifacts, stride=1)
                    render_inference_results(results_df, summary, clean_df)

    # ---------------------------------------------------------
    # Tab 4: Download CSV Template & Input Specifications
    # ---------------------------------------------------------
    with tab_template:
        st.subheader("Download Telemetry CSV Templates")
        st.write("Use these standard CSV templates to prepare operational data for batch upload:")

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("### 🟢 Nominal Baseline Template")
            st.write("Pre-filled with 60 consecutive readings from a verified healthy compressor cycle.")
            st.download_button(
                label="📥 Download Nominal Template CSV (60 Steps)",
                data=get_csv_template("nominal"),
                file_name="metropt_template_nominal_baseline.csv",
                mime="text/csv",
                key="dl_nominal"
            )
        with dcol2:
            st.markdown("### ⚙️ Neutral Resting Template")
            st.write("Pre-filled with 60 rows of neutral resting sensor values ready for custom editing.")
            st.download_button(
                label="📥 Download Neutral Template CSV (60 Steps)",
                data=get_csv_template("blank"),
                file_name="metropt_template_neutral_resting.csv",
                mime="text/csv",
                key="dl_blank"
            )

        st.markdown("---")
        st.subheader("Telemetry Schema Reference (15 Features)")

        schema_data = [
            {"Feature": "timestamp", "Type": "UTC Datetime", "Unit": "ISO-8601", "Normal Envelope": "2020-04-16 00:00:00+", "Description": "10-second regularized timestamp"},
            {"Feature": "TP2", "Type": "Analog Sensor", "Unit": "bar", "Normal Envelope": "0.0 – 10.0 bar", "Description": "Air pressure at compressor block discharge port"},
            {"Feature": "TP3", "Type": "Analog Sensor", "Unit": "bar", "Normal Envelope": "8.0 – 10.0 bar", "Description": "Pneumatic panel auxiliary supply pressure"},
            {"Feature": "H1", "Type": "Analog Sensor", "Unit": "bar", "Normal Envelope": "6.0 – 9.0 bar", "Description": "Differential pressure across cyclonic oil separator filter"},
            {"Feature": "DV_pressure", "Type": "Analog Sensor", "Unit": "bar", "Normal Envelope": "0.0 – 10.0 bar", "Description": "Pressure drop across air dryer purge discharge valve"},
            {"Feature": "Reservoirs", "Type": "Analog Sensor", "Unit": "bar", "Normal Envelope": "8.5 – 10.0 bar", "Description": "Main compressed air storage reservoir pressure"},
            {"Feature": "Motor_current", "Type": "Analog Sensor", "Unit": "A", "Normal Envelope": "0.0 – 9.5 A", "Description": "Phase current draw of 3-phase compressor drive motor"},
            {"Feature": "Oil_temperature", "Type": "Analog Sensor", "Unit": "°C", "Normal Envelope": "45.0 – 80.0 °C", "Description": "Compressor crankcase lubrication oil temperature"},
            {"Feature": "COMP", "Type": "Digital Actuator", "Unit": "0/1", "Normal Envelope": "1 (idle) / 0 (pumping)", "Description": "Compressor intake valve control signal"},
            {"Feature": "DV_eletric", "Type": "Digital Actuator", "Unit": "0/1", "Normal Envelope": "0 (idle) / 1 (pumping)", "Description": "Compressor outlet discharge valve control signal"},
            {"Feature": "Towers", "Type": "Digital Actuator", "Unit": "0/1", "Normal Envelope": "0 or 1", "Description": "Active desiccant dryer tower selector flag"},
            {"Feature": "MPG", "Type": "Digital Actuator", "Unit": "0/1", "Normal Envelope": "0 or 1", "Description": "Motor Pump Governor compressor activation signal"},
            {"Feature": "LPS", "Type": "Digital Actuator", "Unit": "0/1", "Normal Envelope": "0 (Normal) / 1 (Alarm)", "Description": "Low Pressure Switch safety brake alarm (< 7.0 bar)"},
            {"Feature": "Pressure_switch", "Type": "Digital Actuator", "Unit": "0/1", "Normal Envelope": "0 or 1", "Description": "Monitors tower purge discharge cycle"},
            {"Feature": "Oil_level", "Type": "Digital Actuator", "Unit": "0/1", "Normal Envelope": "1 (Normal) / 0 (Low)", "Description": "Crankcase oil level sensor"},
            {"Feature": "Caudal_impulses", "Type": "Digital Actuator", "Unit": "counts", "Normal Envelope": "0 or 1", "Description": "Volumetric air flow delivery pulse counter"},
        ]
        st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

