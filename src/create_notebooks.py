"""
Generates publication-ready, fully executable Jupyter notebooks for all 5 project modules.
"""

import json
from pathlib import Path

from src.config import NOTEBOOKS_DIR


def make_notebook(cells, metadata=None):
    if metadata is None:
        metadata = {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        }
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5
    }


def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def generate_all_notebooks():
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Notebook 1: Data Acquisition and Verification
    # -----------------------------------------------------------------
    nb1_cells = [
        make_md_cell(
            "# Module 01: Official Dataset Acquisition & Ground-Truth Verification\n"
            "**Course:** U21ADP05 — Exploratory Data Analysis and Visualization\n"
            "**Dataset:** Official UCI MetroPT-3 Dataset (ID: 791)\n"
            "**Domain:** Metro Train Air Production Unit (APU) Predictive Maintenance\n\n"
            "This notebook automates downloading the official dataset from the UCI Machine Learning Repository, "
            "extracting the archives, and verifying column schemas, data types, and row completeness."
        ),
        make_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "# Ensure project root is in python path\n"
            "sys.path.append(str(Path.cwd().parent))\n\n"
            "from src.config import RAW_CSV_FILE, UCI_DATASET_URL, ALL_FEATURES\n"
            "from src.data_loader import download_uci_dataset, extract_zip_dataset, verify_dataset_integrity\n\n"
            "print('Configuration and modules loaded successfully!')"
        ),
        make_md_cell("### Step 1: Download & Extract Dataset"),
        make_code_cell(
            "zip_path = download_uci_dataset()\n"
            "csv_path = extract_zip_dataset(zip_path)\n"
            "print(f'CSV dataset ready at: {csv_path}')"
        ),
        make_md_cell("### Step 2: Integrity Verification & Feature Inspection"),
        make_code_cell(
            "info = verify_dataset_integrity(csv_path)\n"
            "print('Verification Summary:')\n"
            "for k, v in info.items():\n"
            "    print(f' - {k}: {v}')"
        ),
        make_md_cell("### Step 3: Preview First 10 Records"),
        make_code_cell(
            "import pandas as pd\n"
            "df_preview = pd.read_csv(csv_path, nrows=10)\n"
            "df_preview"
        )
    ]
    with open(NOTEBOOKS_DIR / "01_data_acquisition_and_inspection.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # -----------------------------------------------------------------
    # Notebook 2: Exploratory Data Analysis & Visualizations
    # -----------------------------------------------------------------
    nb2_cells = [
        make_md_cell(
            "# Module 02: Comprehensive Exploratory Data Analysis (EDA)\n"
            "**Course:** U21ADP05 — Exploratory Data Analysis and Visualization\n\n"
            "This notebook loads the 10-second resampled operational telemetry (~1.5M records), "
            "computes univariate, bivariate, and multivariate statistics, and renders 12 publication-ready figures."
        ),
        make_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.append(str(Path.cwd().parent))\n\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from src.config import PROCESSED_DATA_DIR, FIGURES_EDA_DIR, ALL_FEATURES\n"
            "from src.eda_visualizations import load_processed_data, run_all_eda_visualizations\n\n"
            "df = load_processed_data()\n"
            "print(f'Loaded {len(df):,} resampled telemetry rows.')\n"
            "df.head()"
        ),
        make_md_cell("### Generate & Render All 12 Publication EDA Visualizations"),
        make_code_cell(
            "run_all_eda_visualizations()\n"
            "print('All 12 figures successfully generated and saved to figures/eda/!')"
        ),
        make_md_cell("### Visual Inspection of Generated Figures"),
        make_code_cell(
            "from IPython.display import Image, display\n"
            "eda_figures = sorted(list(FIGURES_EDA_DIR.glob('*.png')))\n"
            "for fig_p in eda_figures:\n"
            "    print(f'Displaying: {fig_p.name}')\n"
            "    display(Image(filename=str(fig_p), width=900))\n"
            "    print('-' * 80)"
        )
    ]
    with open(NOTEBOOKS_DIR / "02_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # -----------------------------------------------------------------
    # Notebook 3: Preprocessing & Leak-Free Pipeline
    # -----------------------------------------------------------------
    nb3_cells = [
        make_md_cell(
            "# Module 03: Data Preprocessing, Hygiene & Strict Chronological Splitting\n"
            "**Course:** U21ADP05 — Exploratory Data Analysis and Visualization\n\n"
            "This notebook details data cleaning, null checks, duplicate timestamp audits, physical limit checks, "
            "resampling, and strictly partitioned chronological train/val/test splits to eliminate temporal data leakage."
        ),
        make_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.append(str(Path.cwd().parent))\n\n"
            "from src.preprocessing import run_preprocessing_pipeline\n"
            "df_train, df_val, df_test = run_preprocessing_pipeline()\n"
            "print('Preprocessing completed successfully!')"
        ),
        make_md_cell("### Partition Statistics & Distribution Verification"),
        make_code_cell(
            "print(f'Train set: {len(df_train):,} rows | Failure rate: {df_train[\"is_failure_period\"].mean():.4f}')\n"
            "print(f'Val set:   {len(df_val):,} rows | Failure rate: {df_val[\"is_failure_period\"].mean():.4f}')\n"
            "print(f'Test set:  {len(df_test):,} rows | Failure rate: {df_test[\"is_failure_period\"].mean():.4f}')\n"
            "assert df_train['is_failure_period'].sum() == 0, 'Leakage detected! Train set contains failure data!'\n"
            "print('Verified: Train partition contains 100% normal operational telemetry.')"
        )
    ]
    with open(NOTEBOOKS_DIR / "03_preprocessing_and_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    # -----------------------------------------------------------------
    # Notebook 4: Deep Learning Model Architecture & Training
    # -----------------------------------------------------------------
    nb4_cells = [
        make_md_cell(
            "# Module 04: Deep LSTM Autoencoder Architecture & Training\n"
            "**Course:** U21ADP05 — Exploratory Data Analysis and Visualization\n\n"
            "This notebook implements the PyTorch Deep LSTM Autoencoder, generates sliding temporal sequence windows, "
            "and trains the network on healthy baseline APU telemetry with early stopping and loss tracking."
        ),
        make_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.append(str(Path.cwd().parent))\n\n"
            "import torch\n"
            "from src.model import LSTMAutoencoder\n"
            "from src.train_runner import run_training\n\n"
            "print(f'PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
        ),
        make_md_cell("### Model Architecture Summary"),
        make_code_cell(
            "model = LSTMAutoencoder(input_dim=15, seq_len=60)\n"
            "print(model)"
        ),
        make_md_cell("### Execute Model Training"),
        make_code_cell(
            "model, history = run_training()\n"
            "print('Training complete! Best weights saved.')"
        )
    ]
    with open(NOTEBOOKS_DIR / "04_deep_learning_lstm_autoencoder.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb4_cells), f, indent=2)

    # -----------------------------------------------------------------
    # Notebook 5: Model Evaluation & Failure Event Analysis
    # -----------------------------------------------------------------
    nb5_cells = [
        make_md_cell(
            "# Module 05: Model Evaluation, Real Metrics & Failure Event Benchmark\n"
            "**Course:** U21ADP05 — Exploratory Data Analysis and Visualization\n\n"
            "This notebook evaluates the trained LSTM Autoencoder against the 4 documented operator failure logs, "
            "computes ROC-AUC, PR-AUC, F1, event-level detection recall, and renders the 5 evaluation figures."
        ),
        make_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.append(str(Path.cwd().parent))\n\n"
            "from src.evaluate import evaluate_model_pipeline\n"
            "metrics = evaluate_model_pipeline()\n"
            "print('Model evaluation complete!')"
        ),
        make_md_cell("### Evaluation Metrics Summary"),
        make_code_cell(
            "import json\n"
            "from src.config import MODELS_DIR\n"
            "with open(MODELS_DIR / 'evaluation_metrics.json') as f:\n"
            "    res = json.load(f)\n"
            "print(json.dumps(res, indent=2))"
        ),
        make_md_cell("### Display Evaluation Visualizations (Figures 13 to 17)"),
        make_code_cell(
            "from IPython.display import Image, display\n"
            "from src.config import FIGURES_EVAL_DIR\n"
            "eval_figures = sorted(list(FIGURES_EVAL_DIR.glob('*.png')))\n"
            "for fig_p in eval_figures:\n"
            "    print(f'Displaying: {fig_p.name}')\n"
            "    display(Image(filename=str(fig_p), width=900))\n"
            "    print('-' * 80)"
        )
    ]
    with open(NOTEBOOKS_DIR / "05_model_evaluation_and_failure_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb5_cells), f, indent=2)

    print(f"[NOTEBOOKS] All 5 Jupyter notebooks generated successfully in {NOTEBOOKS_DIR}!")


if __name__ == "__main__":
    generate_all_notebooks()
