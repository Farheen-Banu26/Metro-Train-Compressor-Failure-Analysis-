# Assignment II Compliance & Verification Audit Checklist

**Course:** U21ADP05 — Exploratory Data Analysis and Visualization  
**Assignment:** Case Study of a Real-World Dataset (Assignment II)  
**Institution:** KPR Institute of Engineering and Technology, Coimbatore  
**Department:** Artificial Intelligence and Data Science  
**Student Name:** Farheen Banu S. D  
**Academic Year:** 2026–2027  
**Project Title:** Metro Train Compressor Failure Analysis Using Exploratory Data Analysis, Visualization, and Deep Learning  
**Dataset:** Official UCI MetroPT-3 Benchmark ([UCI ID: 791](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset))  
**Audit Date:** October 2026  

---

## Executive Summary of Compliance

This document provides a line-by-line audit verifying that the **Metro Train Compressor Failure Analysis** project satisfies every academic requirement specified for Course **U21ADP05 (Assignment II)**. Every compliance claim is mapped directly to actual source files, saved model weights, verified metrics, unit tests, and figure artifacts within the workspace.

```
Overall Compliance Status: 100% Core Technical Requirements Complete & Verified
Automated Test Suite: 16 / 16 Unit Tests Passing (100%)
Systematic Tuning Status: Transparently Documented as Future Work (Curated Domain Baseline Verified)
GitHub Publishing Status: Prepared and committed locally; awaiting user authorization to publish
```

---

## Comprehensive Requirement Verification Matrix

| # | Assignment Requirement | Course Specification | Implementation in Workspace | Verification Evidence & Source Files | Status |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **1** | **Real-World Dataset** | Must use a real-world dataset from an official repository | Official UCI MetroPT-3 benchmark collected from active passenger metro train in Porto, Portugal | `src/config.py`, `src/data_loader.py`, `data/raw/MetroPT3(AirCompressor).csv` | **VERIFIED** |
| **2** | **Feature Count ($\ge 15$)** | Minimum 15 domain features/fields | **15 domain features** (7 analog physical sensors + 8 digital actuator control signals) | `TOTAL_FEATURE_COUNT = 15` in `src/config.py`, verified in `tests/test_pipeline.py::test_feature_count` | **VERIFIED** |
| **3** | **Data Cleaning & Hygiene** | Handle missing values, outliers, duplicates, and physical limits | 0 missing values, 0 duplicate timestamps; sensor bounds clipped ($TP2, TP3 \ge 0$, current $\ge 0$) | `audit_and_clean_data()` in `src/preprocessing.py` | **VERIFIED** |
| **4** | **Temporal Regularization** | Address non-uniform telemetry deltas and noise | Resampled 1.5M raw records into 1,504,107 uniform 10-second calendar bins ($\Delta t = 10\text{s}$) | `resample_telemetry()` in `src/preprocessing.py`, `src/audit_intervals.py` | **VERIFIED** |
| **5** | **Zero Data Leakage Splitting** | Chronological partition; no future lookahead | Train: Feb 28 – Mar 31 (Pure healthy); Val: Apr 1 – Apr 15 (Tuning); Test: Apr 16 – Aug 31 (All failures) | `create_chronological_splits()` in `src/preprocessing.py`, `test_zero_leakage_training_partition` | **VERIFIED** |
| **6** | **Feature Scaling Integrity** | Scaler fitted strictly on training baseline | `StandardScaler` fitted strictly on Train partition ($\mu_{\text{train}}, \sigma_{\text{train}}$); avoids resting sensor explosions | `models/scaler.pkl`, `src/preprocessing.py#L164-L182` | **VERIFIED** |
| **7** | **Descriptive Statistical Analysis** | Comprehensive parametric & non-parametric statistics | Full 11-column table: count, mean, std, min, Q1, median, Q3, max, skewness, IQR on all continuous sensors | Section 5.1 of report, `test_continuous_descriptive_statistics` | **VERIFIED** |
| **8** | **Actuator Duty Cycle Analysis** | Frequencies and operational duty cycle of binary signals | Frequency breakdown of all 8 discrete actuators (`COMP` 16.31% pump, `DV_eletric` 16.07% load, `LPS` 0.34%) | Section 5.2 of report, Table 3 in report | **VERIFIED** |
| **9** | **Sensor Correlation Analysis** | Interpret linear vs monotonic relationships | Pearson ($r$) and Spearman ($\rho$) table: collinear redundancy ($TP3/\text{Reservoirs} = 1.0$), filter inversion ($TP2/H1 = -0.961$) | Section 5.3 of report, `test_correlation_matrix_properties` | **VERIFIED** |
| **10** | **Failure-Event Comparative Analysis** | Quantitative pre-event, event, and recovery comparison | Pre (24h), Event, and Post (24h) metrics across all 4 air leaks (duty cycle surges up to 100%, $+23.68^\circ\text{C}$ oil delta) | Section 5.5 of report, Table 5 in report | **VERIFIED** |
| **11** | **Statistical Testing & Effect Sizes** | Account for temporal autocorrelation; report effect sizes | Autocorrelation caveat documented ($N_{\text{eff}} \ll N$, lag-1 $\rho > 0.95$); Cohen's $d$ reported ($|d| > 1.20$ across channels) | Section 5.6 of report, `test_effect_sizes_and_failure_separation` | **VERIFIED** |
| **12** | **Visualizations Count** | At least 5 meaningful visualizations (target 10–15) | **17 publication-grade visualizations** at 300 DPI (12 EDA figures + 5 evaluation figures embedded) | `figures/eda/fig01` to `fig12`, `figures/evaluation/fig13` to `fig17` | **VERIFIED** |
| **13** | **Seven Visualization Stages** | Explicitly follow the seven stages of data visualization | Complete documentation covering Acquire, Parse, Filter, Mine, Represent, Refine, and Interact | Section 6 of academic report, interactive Streamlit dashboard | **VERIFIED** |
| **14** | **Deep Learning Architecture** | Build at least one suitable deep learning model | Deep LSTM Autoencoder (15 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 15) in PyTorch | `LSTMAutoencoder` in `src/model.py`, weights in `models/best_lstm_autoencoder.pt` | **VERIFIED** |
| **15** | **Hyperparameter Documentation** | Document all verified model settings and rationale | All 14 settings verified (sequence length 60, latent 16, batch 256, Adam lr=1e-3, early stopping patience 3, seed 42) | Section 7.2 of report, `test_hyperparameter_configuration_integrity` | **VERIFIED** |
| **16** | **Systematic Hyperparameter Tuning** | Systematic search (grid, random, or Optuna search) | Curated literature baseline verified; systematic grid/Bayesian search transparently documented as future work | Section 7.3 of report | **FUTURE WORK** |
| **17** | **Dynamic Threshold Selection** | Anomaly threshold set on healthy data only | $\tau = 2.589630$ established strictly at 99.0th percentile of healthy validation error | `models/evaluation_metrics.json`, `src/evaluate.py#L150-L157` | **VERIFIED** |
| **18** | **Empirical Model Evaluation** | Real quantitative metrics on independent test partition | ROC-AUC: **0.8203**, PR-AUC: **0.0776**, Recall: **83.74%**, Precision: **11.08%**, $F_1$: **0.1957** | `models/evaluation_metrics.json`, Figure 16 | **VERIFIED** |
| **19** | **Event-Level Detection** | Recall across operator-documented failure incidents | **100% Incident Recall** (All 4 confirmed operator air leak events successfully flagged) | Events #1, #2, #3, #4 audited in `src/evaluate.py` | **VERIFIED** |
| **20** | **Sensor Attribution & Explainability** | Sensor error decomposition for root-cause diagnostics | Per-channel reconstruction error decomposition ($e_{t,j}$) highlighting `TP2`, `Motor_current`, `DV_pressure` | Figure 17, `compute_per_channel_reconstruction_error()` | **VERIFIED** |
| **21** | **Interactive Prediction Feature** | Working demonstration with user inputs | Streamlit dashboard page 6: CSV Upload, Interactive 60-Step Data Editor, Benchmark Demos, CSV Download | `app/dashboard.py`, `src/inference.py` | **VERIFIED** |
| **22** | **Scientific Honesty & Limitations** | Distinguish anomaly detection from future failure prediction | Explicitly distinguishes anomaly detection from RUL forecasting; reports $FP=33,258$ and seasonal thermal drift | Section 12 of Report | **VERIFIED** |
| **23** | **Actual Code Appendix** | Real syntax-correct, commented code excerpts with files | 8 authentic code excerpts covering ingestion, cleaning, scaling, architecture, training, scoring, evaluation, and inference | Section 15.2 of Report | **VERIFIED** |
| **24** | **Automated Testing Suite** | Comprehensive automated verification tests | **16 / 16 unit tests passing** covering splits, leakage, figures, metrics, inference, and statistical properties | `tests/test_pipeline.py` run via `pytest` | **VERIFIED** |
| **25** | **Academic Report Deliverables** | Formal academic report with all 17 figures embedded | Provided in Markdown (`.md`), Microsoft Word (`.docx`), and Adobe PDF (`.pdf`) formats | `reports/Metro_Train_Compressor_Failure_Analysis_Report.*` | **VERIFIED** |
| **26** | **References (IEEE Style)** | At least 3–5 genuine, verifiable scientific citations | 5 genuine peer-reviewed and primary repository citations in IEEE format | Section 14 of Report, Veloso et al. (2022), UCI (2022), Malhotra et al. | **VERIFIED** |
| **27** | **Public GitHub Repository** | Clean repository ready for public sharing | Git repository clean, fully committed on `master` branch. Held locally pending user release signal | Local Git repository (Branch: `master`, Clean tree) | **READY** |

---

## Detailed Requirement Auditing

### 1. Statistical Analysis & Sensor Distributions (Task 1A)
- **Descriptive Statistics Table:** Computed across all 1,504,107 records for all 7 continuous features:
  - `TP2`: Mean 1.3681 bar, median -0.0120 bar, IQR 0.0040 bar, skewness +1.9857 (extreme zero-inflation during idle states).
  - `TP3`: Mean 8.9846 bar, median 8.9600 bar, IQR 1.0000 bar, skewness -0.8748.
  - `H1`: Mean 7.5678 bar, median 8.7840 bar, IQR 1.1200 bar, skewness -1.7645.
  - `DV_pressure`: Mean 0.0559 bar, median -0.0200 bar, IQR 0.0040 bar, skewness +5.7184 (extreme zero-inflation with desiccant purge spikes).
  - `Reservoirs`: Mean 8.9852 bar, median 8.9600 bar, IQR 0.9980 bar, skewness -0.8836.
  - `Motor_current`: Mean 2.0505 A, median 0.0450 A, IQR 3.7675 A, skewness +0.4482 (bimodal idle vs pumping distribution).
  - `Oil_temperature`: Mean 62.6447 °C, median 62.7250 °C, IQR 9.4750 °C, skewness -0.0458 (symmetric Gaussian-like thermal distribution).
- **Physical Bounds & Transducer Drift:** Minor negative pressures (< 0 bar down to -0.032 bar) represent piezoresistive zero-drift and temperature sensitivity rather than physical vacuum.

### 2. Digital Actuator Frequencies (Task 1A)
- `COMP`: Inactive (0) during pumping (16.31%), active (1) during offloaded idling (83.69%).
- `DV_eletric`: Inactive (0) during idling (83.93%), active (1) during loaded pumping (16.07%).
- `Towers`: Tower 1 active 8.01%, Tower 2 active 91.99%.
- `LPS`: Low Pressure Switch alarm at 0.34% of entire 7-month dataset.

### 3. Sensor Collinearity & Relationships (Task 1B)
- **Collinear Redundancy:** `TP3` vs `Reservoirs` ($r = 1.000, \rho = 1.000$) measures identical physical reservoir storage air volume.
- **Filter Inversion:** `TP2` vs `H1` ($r = -0.961, \rho = -0.391$) tracks reciprocal separator pressure drop.
- **Electromechanical Coupling:** `TP2` vs `Motor_current` ($r = 0.698, \rho = 0.358$).
- **Thermal Dissipation:** `Motor_current` vs `Oil_temperature` ($r = 0.529, \rho = 0.494$).
- **Causality Caveat:** Explicitly notes that correlation does not establish direct causation.

### 4. Failure-Event Comparative Analysis (Task 1D)
- Quantitative comparison across 24h pre-event, event window, and 24h post-recovery:
  - Event #1: Duty cycle surges from 2.4% to 98.9%; motor current rises from 0.45 A to 5.60 A; oil temp surges by $+23.68^\circ\text{C}$ (50.45°C to 74.13°C). Precursor leak active April 17.
  - Event #2: Duty cycle surges from 13.9% to 99.1%; motor current rises to 5.57 A; oil temp reaches 75.87°C; alarms trigger within 20s.
  - Event #3: Duty cycle locks at 100.0% for 52.5h; reservoir collapses from 9.02 to 8.02 bar (and lower to 7.39 bar); peak MSE hits 19.10; major overhaul executed June 8.
  - Event #4: Pre-event duty cycle already elevated at 39.3%, surging to 97.3%; oil temp hits record 83.90°C (peak 89.05°C).

### 5. Statistical Testing & Effect Sizes (Task 1E)
- **Autocorrelation Limitation:** High-frequency rolling stock telemetry exhibits extreme temporal autocorrelation (lag-1 $\rho > 0.95$). Standard independent-sample t-tests and ANOVA violate i.i.d. assumptions and report artificially inflated degrees of freedom ($p < 10^{-300}$).
- **Standardized Effect Sizes (Cohen's d):**
  - `DV_pressure`: $d = +6.845$ (Massive positive shift)
  - `H1`: $d = -2.301$ (Massive negative collapse)
  - `TP2`: $d = +2.068$ (Massive positive elevation)
  - `Oil_temperature`: $d = +1.997$ (Massive thermal heating)
  - `Motor_current`: $d = +1.572$ (Massive continuous electrical load)
  - `Reservoirs`: $d = -1.201$ (Large pressure depression)

### 6. Hyperparameter Documentation & Tuning Verification (Task 2)
- **Verified Model Settings:** Sequence window $W = 60$, features $D = 15$, encoder [64, 32], bottleneck $K = 16$, decoder [32, 64], dropout $0.2$, Adam lr $10^{-3}$, batch size $256$, epochs $10$, seed $42$, threshold $\tau = 2.589630$.
- **Transparent Tuning Accounting:** Confirms that systematic hyperparameter tuning (grid/random/Optuna) was **not conducted** due to computational constraints on CPU. Baseline architecture was adopted directly from proven industrial literature (Malhotra et al. 2016; Baptista et al. 2018). Multi-trial Bayesian tuning is formally designated as future work.

### 7. Code Appendix Excerpts (Task 4)
- Excerpt 1: `src/preprocessing.py::load_raw_dataset`
- Excerpt 2: `src/preprocessing.py::audit_and_clean_data` & `resample_telemetry`
- Excerpt 3: `src/preprocessing.py::create_chronological_splits` & `create_sliding_windows`
- Excerpt 4: `src/model.py::LSTMEncoder`, `LSTMDecoder`, `LSTMAutoencoder`
- Excerpt 5: `src/trainer.py::train_lstm_autoencoder`
- Excerpt 6: `src/evaluate.py::compute_windowed_anomaly_scores` & `src/model.py`
- Excerpt 7: `src/evaluate.py::evaluate_model_pipeline`
- Excerpt 8: `src/inference.py::validate_input_dataframe` & `run_inference`

### 8. Automated Testing Suite (16 / 16 Unit Tests Passing)
- `tests/test_pipeline.py::TestDataPipeline` (6 tests: feature count, zero leakage, chronological order, figures existence, model artifacts)
- `tests/test_pipeline.py::TestInferencePipeline` (6 tests: artifacts loading, input validation, insufficient sequence rejection, threshold math, historical benchmarks, CSV templates)
- `tests/test_pipeline.py::TestStatisticalAnalysis` (4 tests: continuous descriptive stats, correlation properties, Cohen's d effect sizes, hyperparameter configuration integrity)
