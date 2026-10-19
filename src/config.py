"""
Configuration module for Metro Train Compressor Failure Analysis.
Defines paths, feature column specifications, ground-truth failure intervals,
chronological partition dates, and deep learning model hyperparameters.
"""

from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_EDA_DIR = FIGURES_DIR / "eda"
FIGURES_EVAL_DIR = FIGURES_DIR / "evaluation"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# UCI Dataset Details
UCI_DATASET_ID = 791
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
RAW_ZIP_FILE = RAW_DATA_DIR / "metropt_3_dataset.zip"
RAW_CSV_FILE = RAW_DATA_DIR / "MetroPT3(AirCompressor).csv"

# Column Definitions
INDEX_COL = "index"
TIMESTAMP_COL = "timestamp"

ANALOG_FEATURES = [
    "TP2",              # Compressor pressure (bar)
    "TP3",              # Pneumatic panel pressure (bar)
    "H1",               # Oil separator filter pressure drop (bar)
    "DV_pressure",      # Air dryer discharge pressure drop (bar)
    "Reservoirs",       # Main air reservoirs pressure (bar)
    "Motor_current",    # Electric motor 3-phase current (Amps)
    "Oil_temperature",  # Compressor oil temperature (°C)
]

DIGITAL_FEATURES = [
    "COMP",             # Intake valve electrical signal (active when offloaded)
    "DV_eletric",       # Outlet valve control signal (active under load)
    "Towers",           # Dryer tower active selection (0 = Tower 1, 1 = Tower 2)
    "MPG",              # Motor Pump Governor signal (triggers when APU < 8.2 bar)
    "LPS",              # Low Pressure Switch signal (triggers when system < 7.0 bar)
    "Pressure_switch",  # Dryer discharge cycle detection switch
    "Oil_level",        # Low oil level alarm
    "Caudal_impulses",  # Volumetric air flow pulse count
]

ALL_FEATURES = ANALOG_FEATURES + DIGITAL_FEATURES
TOTAL_FEATURE_COUNT = len(ALL_FEATURES)  # Exactly 15 features

# Official Ground-Truth Failure Intervals (from operator maintenance records)
FAILURE_EVENTS = [
    {
        "id": 1,
        "name": "Failure #1",
        "start": "2020-04-18 00:00:00",
        "end": "2020-04-18 23:59:00",
        "failure_type": "Air leak",
        "severity": "High stress",
        "maintenance": None,
        "notes": "Severe air leak causing continuous compressor stress"
    },
    {
        "id": 2,
        "name": "Failure #2",
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
        "failure_type": "Air leak",
        "severity": "High stress",
        "maintenance": "2020-05-30 12:00:00",
        "notes": "Air leak resulting in emergency maintenance at 12:00 on May 30"
    },
    {
        "id": 3,
        "name": "Failure #3",
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
        "failure_type": "Air leak",
        "severity": "High stress",
        "maintenance": "2020-06-08 16:00:00",
        "notes": "Extended major air leak spanning 2+ days before scheduled maintenance"
    },
    {
        "id": 4,
        "name": "Failure #4",
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
        "failure_type": "Air leak",
        "severity": "High stress",
        "maintenance": "2020-07-16 00:00:00",
        "notes": "Air leak episode followed by midnight maintenance intervention"
    }
]

# Strict Chronological Splitting Boundaries
# Train: Pure healthy baseline operation before any failure incident
TRAIN_START = "2020-02-28 00:00:00"
TRAIN_END   = "2020-03-31 23:59:59"

# Validation: Healthy operating period used to tune threshold tau and monitor loss
VAL_START   = "2020-04-01 00:00:00"
VAL_END     = "2020-04-15 23:59:59"

# Test: Period containing degradation, all 4 failure incidents, and post-repair recoveries
TEST_START  = "2020-04-16 00:00:00"
TEST_END    = "2020-08-31 23:59:59"

# Preprocessing & Resampling
# Downsample aggregation frequency for noise reduction and training efficiency
RESAMPLE_FREQ = "10s"  # 10-second averages preserve operational cycles

# Deep Learning Model Hyperparameters (LSTM Autoencoder)
WINDOW_SIZE = 60          # 60 timesteps (10 mins at 10s resolution, or 1 min at 1s)
STRIDE_TRAIN = 10         # Stride for training sequence generation
STRIDE_TEST = 1           # Continuous stride for inference / anomaly scoring
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
PATIENCE = 3              # Early stopping patience
LATENT_DIM = 16
ENCODER_DIMS = [64, 32]
DECODER_DIMS = [32, 64]
DROPOUT_RATE = 0.2
RANDOM_SEED = 42

# Thresholding Parameters
ANOMALY_PERCENTILE = 99.0  # 99th percentile of validation reconstruction error
