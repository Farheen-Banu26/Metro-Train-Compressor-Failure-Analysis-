"""
Exploratory Data Analysis (EDA) and Visualization module.
Generates 12 publication-grade, 300-DPI visualizations covering univariate, bivariate,
multivariate, temporal macro/micro cycles, phase-space trajectories, and failure deep dives.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from src.config import (
    PROCESSED_DATA_DIR,
    FIGURES_EDA_DIR,
    ANALOG_FEATURES,
    DIGITAL_FEATURES,
    ALL_FEATURES,
    TIMESTAMP_COL,
    FAILURE_EVENTS,
)

# Apply publication-ready visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.autolayout": True,
    "savefig.dpi": 300,
})

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "danger": "#d62728",
    "neutral": "#7f7f7f",
    "dark": "#2c3e50",
    "healthy": "#2ecc71",
    "failure": "#e74c3c"
}


def load_processed_data() -> pd.DataFrame:
    """Loads the resampled telemetry parquet file."""
    parquet_path = PROCESSED_DATA_DIR / "telemetry_resampled.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed file not found: {parquet_path}. Run preprocessing first.")
    df = pd.read_parquet(parquet_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df


def plot_fig01_telemetry_continuity(df: pd.DataFrame):
    """Figure 1: Missing Data & Telemetry Continuity Heatmap across the 6-month operational period."""
    print("[EDA] Generating Fig 01: Telemetry Continuity & Sampling Distribution...", flush=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Calculate daily sample count
    daily_counts = df.set_index(TIMESTAMP_COL).resample("D")["TP2"].count()
    expected_per_day = 24 * 3600 / 10  # 10s resampled: 8640 samples/day
    completeness_pct = (daily_counts / expected_per_day) * 100
    completeness_pct = completeness_pct.clip(upper=100.0)

    ax.plot(completeness_pct.index, completeness_pct.values, color=COLORS["primary"], lw=1.5, label="Daily Completeness (%)")
    ax.fill_between(completeness_pct.index, completeness_pct.values, alpha=0.3, color=COLORS["primary"])
    ax.axhline(100.0, color="gray", linestyle="--", alpha=0.7, label="100% Theoretical Completeness")

    # Mark the 4 failure events
    for ev in FAILURE_EVENTS:
        start_d = pd.to_datetime(ev["start"])
        end_d = pd.to_datetime(ev["end"])
        ax.axvspan(start_d, end_d, color=COLORS["danger"], alpha=0.35, label=f"{ev['name']} ({ev['failure_type']})" if ev["id"] == 1 else None)

    ax.set_title("Figure 1: MetroPT-3 Telemetry Continuity and Daily Data Completeness Rate (Feb - Aug 2020)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Data Completeness (%)")
    ax.set_ylim(-5, 110)
    ax.legend(loc="lower left", frameon=True)
    
    save_path = FIGURES_EDA_DIR / "fig01_telemetry_continuity_missingness.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig02_univariate_analog_distributions(df: pd.DataFrame):
    """Figure 2: Multi-panel KDE and Histograms for all 7 Analog Sensor Features."""
    print("[EDA] Generating Fig 02: Univariate Analog Distributions...", flush=True)
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()

    units = {
        "TP2": "bar", "TP3": "bar", "H1": "bar", "DV_pressure": "bar",
        "Reservoirs": "bar", "Motor_current": "A", "Oil_temperature": "°C"
    }

    # Sample 50k rows for instantaneous, mathematically representative KDE estimation
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)

    for i, col in enumerate(ANALOG_FEATURES):
        ax = axes[i]
        sns.histplot(sample_df[col], kde=True, ax=ax, color=COLORS["primary"], bins=50, stat="density", alpha=0.5)
        mean_val = float(df[col].mean())
        median_val = float(df[col].median())
        ax.axvline(mean_val, color=COLORS["danger"], linestyle="--", lw=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color=COLORS["accent"], linestyle="-.", lw=1.5, label=f"Median: {median_val:.2f}")
        ax.set_title(f"{col} ({units.get(col, '')}) Distribution", fontweight="bold")
        ax.set_xlabel(f"{col} [{units.get(col, '')}]")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", frameon=True)

    # Empty 8th subplot used for summary statistics table
    ax_unused = axes[7]
    ax_unused.axis("off")
    stats_df = df[ANALOG_FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
    stats_df.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
    stats_text = stats_df.round(2).to_string()
    ax_unused.text(0.05, 0.5, f"Analog Features Descriptive Statistics (Full Dataset):\n\n{stats_text}", 
                   fontsize=11, fontfamily="monospace", va="center")

    fig.suptitle("Figure 2: Univariate Distribution Analysis of APU Analog Telemetry", fontsize=16, fontweight="bold", y=0.995)
    save_path = FIGURES_EDA_DIR / "fig02_univariate_analog_distributions.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig03_digital_signal_frequencies(df: pd.DataFrame):
    """Figure 3: Duty Cycles and State Frequencies for all 8 Digital / Binary Control Signals."""
    print("[EDA] Generating Fig 03: Digital Control Signal Frequencies...", flush=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate activation percentages
    active_percentages = (df[DIGITAL_FEATURES].mean() * 100).sort_values(ascending=True)

    bars = ax.barh(active_percentages.index, active_percentages.values, color=COLORS["primary"], alpha=0.85, edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_title("Figure 3: Digital Signal & Actuator Duty Cycles (Active / On State %)", fontweight="bold")
    ax.set_xlabel("Time Spent Active (%)")
    ax.set_ylabel("Digital Signal")
    ax.set_xlim(0, max(active_percentages.values) + 12)

    save_path = FIGURES_EDA_DIR / "fig03_digital_signal_frequencies.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig04_macro_telemetry_timeline(df: pd.DataFrame):
    """Figure 4: Macro 6-Month Timeline tracking multi-sensor operational envelope and seasonal trends."""
    print("[EDA] Generating Fig 04: Macro 6-Month Telemetry Timeline...", flush=True)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # Subsample step 10 for smooth vector line rendering
    sub = df.iloc[::10]

    # 1. Pressures
    axes[0].plot(sub[TIMESTAMP_COL], sub["Reservoirs"], label="Reservoirs", color=COLORS["primary"], alpha=0.7, lw=0.8)
    axes[0].plot(sub[TIMESTAMP_COL], sub["TP3"], label="TP3 (Panel)", color=COLORS["secondary"], alpha=0.5, lw=0.6)
    axes[0].set_ylabel("Pressure (bar)")
    axes[0].set_title("Reservoir & Pneumatic Panel Pressure Dynamics", fontweight="bold")
    axes[0].legend(loc="upper right")

    # 2. Motor Current
    axes[1].plot(sub[TIMESTAMP_COL], sub["Motor_current"], label="Motor Current (A)", color=COLORS["accent"], lw=0.6, alpha=0.8)
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Electric Motor Current Consumption Envelope", fontweight="bold")
    axes[1].legend(loc="upper right")

    # 3. Oil Temperature
    axes[2].plot(sub[TIMESTAMP_COL], sub["Oil_temperature"], label="Oil Temperature (°C)", color="#d35400", lw=0.8)
    axes[2].set_ylabel("Temp (°C)")
    axes[2].set_title("Compressor Oil Temperature (Capturing Seasonal Heating Drift)", fontweight="bold")
    axes[2].legend(loc="upper right")

    # 4. Filter Pressure Drop H1
    axes[3].plot(sub[TIMESTAMP_COL], sub["H1"], label="H1 Oil Separator Drop (bar)", color="#8e44ad", lw=0.6)
    axes[3].set_ylabel("H1 Drop (bar)")
    axes[3].set_xlabel("Date")
    axes[3].set_title("Cyclonic Oil Separator Pressure Drop Telemetry", fontweight="bold")
    axes[3].legend(loc="upper right")

    # Highlight Failure Intervals on all subplots
    for ax in axes:
        for ev in FAILURE_EVENTS:
            ax.axvspan(pd.to_datetime(ev["start"]), pd.to_datetime(ev["end"]), color=COLORS["danger"], alpha=0.25)

    fig.suptitle("Figure 4: Macro 6-Month Multi-Sensor Telemetry Overview (Feb - Aug 2020)", fontsize=16, fontweight="bold", y=0.995)
    save_path = FIGURES_EDA_DIR / "fig04_macro_telemetry_timeline.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig05_micro_cycle_dynamics(df: pd.DataFrame):
    """Figure 5: High-Resolution Microscopic Thermodynamic Pumping Cycle (15-Minute Window)."""
    print("[EDA] Generating Fig 05: Micro Cycle Dynamics...", flush=True)
    sample_start = pd.to_datetime("2020-03-10 14:00:00")
    sample_end = pd.to_datetime("2020-03-10 14:20:00")
    mask = (df[TIMESTAMP_COL] >= sample_start) & (df[TIMESTAMP_COL] <= sample_end)
    sub = df[mask]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Pressures on ax1
    l1 = ax1.plot(sub[TIMESTAMP_COL], sub["TP2"], color=COLORS["primary"], lw=2.0, label="TP2 (Compressor Outlet)")
    l2 = ax1.plot(sub[TIMESTAMP_COL], sub["Reservoirs"], color=COLORS["secondary"], lw=2.0, label="Reservoirs Pressure")
    l3 = ax1.plot(sub[TIMESTAMP_COL], sub["DV_pressure"], color="purple", linestyle=":", lw=1.5, label="DV_pressure (Discharge)")
    ax1.set_ylabel("Pressure (bar)", fontsize=12)
    ax1.set_xlabel("Time (HH:MM)", fontsize=12)

    # Motor current on ax2
    l4 = ax2.plot(sub[TIMESTAMP_COL], sub["Motor_current"], color=COLORS["accent"], lw=1.5, linestyle="--", label="Motor Current (A)")
    ax2.set_ylabel("Motor Current (A)", color=COLORS["accent"], fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COLORS["accent"])

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    ax1.set_title("Figure 5: High-Resolution Microscopic APU Pumping Cycle (Load, Pumping, and Purge Phases)", fontweight="bold")
    save_path = FIGURES_EDA_DIR / "fig05_micro_cycle_dynamics.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig06_correlation_matrices(df: pd.DataFrame):
    """Figure 6: Side-by-Side Pearson (Linear) and Spearman Rank (Monotonic) Correlation Matrices."""
    print("[EDA] Generating Fig 06: Correlation Matrices...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sample_corr = df[ALL_FEATURES].sample(n=min(50000, len(df)), random_state=42)
    corr_pearson = sample_corr.corr(method="pearson")
    corr_spearman = sample_corr.corr(method="spearman")

    sns.heatmap(corr_pearson, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Pearson Linear Correlation Matrix", fontweight="bold")

    sns.heatmap(corr_spearman, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Spearman Rank Monotonic Correlation Matrix", fontweight="bold")

    fig.suptitle("Figure 6: Inter-Sensor Correlation Analysis across All 15 Telemetry Features", fontsize=16, fontweight="bold", y=0.98)
    save_path = FIGURES_EDA_DIR / "fig06_correlation_matrix_pearson_spearman.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig07_phase_space_trajectories(df: pd.DataFrame):
    """Figure 7: Phase-Space Bivariate Trajectories (TP2 vs TP3 and Motor Current vs DV_pressure)."""
    print("[EDA] Generating Fig 07: Phase-Space Bivariate Couplings...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sample_df = df.sample(n=min(50000, len(df)), random_state=42)

    # 1. TP2 vs TP3
    hb1 = axes[0].hexbin(sample_df["TP2"], sample_df["TP3"], gridsize=40, cmap="viridis", mincnt=1, bins="log")
    axes[0].set_title("Phase Space: TP2 (Compressor) vs. TP3 (Panel)", fontweight="bold")
    axes[0].set_xlabel("TP2 Pressure (bar)")
    axes[0].set_ylabel("TP3 Pressure (bar)")
    cb1 = fig.colorbar(hb1, ax=axes[0])
    cb1.set_label("Log Count Density")

    # 2. Motor Current vs DV Pressure
    hb2 = axes[1].hexbin(sample_df["DV_pressure"], sample_df["Motor_current"], gridsize=40, cmap="plasma", mincnt=1, bins="log")
    axes[1].set_title("Phase Space: DV Pressure vs. Motor Current", fontweight="bold")
    axes[1].set_xlabel("DV Pressure (bar)")
    axes[1].set_ylabel("Motor Current (A)")
    cb2 = fig.colorbar(hb2, ax=axes[1])
    cb2.set_label("Log Count Density")

    fig.suptitle("Figure 7: Bivariate Phase-Space Couplings Demonstrating Mechanical Regimes", fontsize=16, fontweight="bold")
    save_path = FIGURES_EDA_DIR / "fig07_phase_space_trajectories.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig08_operational_regime_boxplots(df: pd.DataFrame):
    """Figure 8: Operational Regime Boxplots (Motor Current & Oil Temp partitioned by COMP status)."""
    print("[EDA] Generating Fig 08: Operational Regime Boxplots...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sample_df = df.sample(n=min(50000, len(df)), random_state=42).copy()
    sample_df["Compressor_State"] = sample_df["COMP"].map({0.0: "Loaded / Pumping", 1.0: "Offloaded / Idle"})

    palette = {"Loaded / Pumping": COLORS["danger"], "Offloaded / Idle": COLORS["primary"]}

    sns.boxplot(data=sample_df, x="Compressor_State", y="Motor_current", ax=axes[0], palette=palette)
    axes[0].set_title("Motor Current by Operating State", fontweight="bold")
    axes[0].set_xlabel("Operational State (COMP Valve)")
    axes[0].set_ylabel("Motor Current (A)")

    sns.boxplot(data=sample_df, x="Compressor_State", y="Oil_temperature", ax=axes[1], palette=palette)
    axes[1].set_title("Compressor Oil Temperature by Operating State", fontweight="bold")
    axes[1].set_xlabel("Operational State (COMP Valve)")
    axes[1].set_ylabel("Oil Temperature (°C)")

    fig.suptitle("Figure 8: Operational State Separation (Loaded vs. Offloaded Regimes)", fontsize=16, fontweight="bold")
    save_path = FIGURES_EDA_DIR / "fig08_operational_regime_boxplots.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig09_diurnal_weekly_duty_cycle(df: pd.DataFrame):
    """Figure 9: Heatmap of Compressor Activation Duty Cycle by Hour-of-Day and Day-of-Week."""
    print("[EDA] Generating Fig 09: Diurnal & Weekly Duty Cycle Heatmap...", flush=True)
    df_time = df.copy()
    df_time["hour"] = df_time[TIMESTAMP_COL].dt.hour
    df_time["day_of_week"] = df_time[TIMESTAMP_COL].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    df_time["is_pumping"] = (df_time["Motor_current"] > 1.0).astype(float)
    pivot = df_time.pivot_table(index="day_of_week", columns="hour", values="is_pumping", aggfunc="mean") * 100
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "Pumping Duty Cycle (%)"})
    ax.set_title("Figure 9: APU Compressor Duty Cycle (%) Heatmap by Day of Week & Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour of Day (0 - 23)")
    ax.set_ylabel("Day of Week")

    save_path = FIGURES_EDA_DIR / "fig09_diurnal_weekly_duty_cycle.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig10_failure_event1_deep_dive(df: pd.DataFrame):
    """Figure 10: Deep Dive into Ground-Truth Failure Event #1 (April 18, 2020 Air Leak)."""
    print("[EDA] Generating Fig 10: Failure Event #1 Deep Dive...", flush=True)
    start_window = pd.to_datetime("2020-04-17 12:00:00")
    end_window   = pd.to_datetime("2020-04-19 12:00:00")
    sub = df[(df[TIMESTAMP_COL] >= start_window) & (df[TIMESTAMP_COL] <= end_window)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(sub[TIMESTAMP_COL], sub["Reservoirs"], color=COLORS["primary"], lw=1.2, label="Reservoirs Pressure")
    axes[0].plot(sub[TIMESTAMP_COL], sub["TP2"], color=COLORS["secondary"], alpha=0.6, lw=0.8, label="TP2 (Compressor)")
    axes[0].set_ylabel("Pressure (bar)")
    axes[0].set_title("Pneumatic Pressure Collapse and Rapid Cycling", fontweight="bold")
    axes[0].legend(loc="upper right")

    axes[1].plot(sub[TIMESTAMP_COL], sub["Motor_current"], color=COLORS["danger"], lw=1.0, label="Motor Current (A)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Continuous Electric Motor Overdrive", fontweight="bold")
    axes[1].legend(loc="upper right")

    axes[2].plot(sub[TIMESTAMP_COL], sub["Oil_temperature"], color="#d35400", lw=1.2, label="Oil Temp (°C)")
    axes[2].set_ylabel("Oil Temp (°C)")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_title("Lubrication Oil Thermal Surge", fontweight="bold")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.axvspan(pd.to_datetime("2020-04-18 00:00:00"), pd.to_datetime("2020-04-18 23:59:00"), 
                   color="red", alpha=0.2, label="Official Air Leak Incident Window")

    fig.suptitle("Figure 10: Ground-Truth Failure Event #1 Deep Dive: Pneumatic Telemetry During April 18 Air Leak", fontsize=15, fontweight="bold")
    save_path = FIGURES_EDA_DIR / "fig10_failure_event1_air_leak_deep_dive.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig11_failure_event3_breakdown_recovery(df: pd.DataFrame):
    """Figure 11: Multi-Day Breakdown, Failure, and Maintenance Recovery (June 5–9, 2020)."""
    print("[EDA] Generating Fig 11: Failure Event #3 Breakdown & Recovery...", flush=True)
    start_window = pd.to_datetime("2020-06-04 00:00:00")
    end_window   = pd.to_datetime("2020-06-10 00:00:00")
    sub = df[(df[TIMESTAMP_COL] >= start_window) & (df[TIMESTAMP_COL] <= end_window)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(sub[TIMESTAMP_COL], sub["Reservoirs"], color=COLORS["primary"], lw=1.0, label="Reservoirs Pressure")
    axes[0].set_ylabel("Pressure (bar)")
    axes[0].set_title("Reservoir Pressure Behavior Across Breakdown and Maintenance", fontweight="bold")
    axes[0].legend(loc="upper right")

    axes[1].plot(sub[TIMESTAMP_COL], sub["Motor_current"], color=COLORS["accent"], lw=0.8, label="Motor Current (A)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Motor Duty Cycle (Extreme Strain during Leak vs. Restabilization Post-Maintenance)", fontweight="bold")
    axes[1].legend(loc="upper right")

    axes[2].plot(sub[TIMESTAMP_COL], sub["Oil_temperature"], color="#c0392b", lw=1.0, label="Oil Temp (°C)")
    axes[2].set_ylabel("Oil Temp (°C)")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Thermal Trajectory Pre-, During-, and Post-Failure", fontweight="bold")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.axvspan(pd.to_datetime("2020-06-05 10:00:00"), pd.to_datetime("2020-06-07 14:30:00"), 
                   color="red", alpha=0.25, label="Documented Failure Period")
        ax.axvline(pd.to_datetime("2020-06-08 16:00:00"), color="green", linestyle="--", lw=2, label="Depot Maintenance Intervention")

    axes[0].legend(loc="upper right")
    fig.suptitle("Figure 11: Failure Event #3 Telemetry: Air Leak Breakdown and Subsequent Depot Maintenance Recovery", fontsize=15, fontweight="bold")
    save_path = FIGURES_EDA_DIR / "fig11_failure_event3_breakdown_and_recovery.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def plot_fig12_manifold_projection(df: pd.DataFrame):
    """Figure 12: 2D PCA Dimensionality Reduction Projection (Healthy Manifold vs. Failure Regimes)."""
    print("[EDA] Generating Fig 12: 2D Manifold Projection...", flush=True)
    failure_mask = df["is_failure_period"] == 1
    df_fail = df[failure_mask]
    df_norm = df[~failure_mask].sample(n=min(len(df_fail) * 3, 20000), random_state=42)

    sample_combined = pd.concat([df_norm, df_fail])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(sample_combined[ALL_FEATURES])

    fig, ax = plt.subplots(figsize=(11, 8))
    norm_idx = sample_combined["is_failure_period"] == 0
    fail_idx = sample_combined["is_failure_period"] == 1

    ax.scatter(coords[norm_idx, 0], coords[norm_idx, 1], c=COLORS["primary"], alpha=0.3, s=12, label="Normal Operation Manifold")
    ax.scatter(coords[fail_idx, 0], coords[fail_idx, 1], c=COLORS["danger"], alpha=0.7, s=20, label="Documented Air Leak Failure Events")

    ax.set_title("Figure 12: 2D Principal Component Analysis (PCA) Projection of APU Telemetry Manifold", fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.legend(loc="upper right", frameon=True)

    save_path = FIGURES_EDA_DIR / "fig12_manifold_pca_tsne_projection.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  --> Saved: {save_path}", flush=True)


def run_all_eda_visualizations():
    """Generates and saves the full suite of 12 publication-ready EDA figures."""
    FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()
    print(f"[EDA] Starting visualization pipeline on {len(df):,} resampled telemetry rows...", flush=True)
    
    plot_fig01_telemetry_continuity(df)
    plot_fig02_univariate_analog_distributions(df)
    plot_fig03_digital_signal_frequencies(df)
    plot_fig04_macro_telemetry_timeline(df)
    plot_fig05_micro_cycle_dynamics(df)
    plot_fig06_correlation_matrices(df)
    plot_fig07_phase_space_trajectories(df)
    plot_fig08_operational_regime_boxplots(df)
    plot_fig09_diurnal_weekly_duty_cycle(df)
    plot_fig10_failure_event1_deep_dive(df)
    plot_fig11_failure_event3_breakdown_recovery(df)
    plot_fig12_manifold_projection(df)
    
    print("[EDA] All 12 figures successfully generated and saved to figures/eda/!", flush=True)


if __name__ == "__main__":
    run_all_eda_visualizations()
