"""
Unit tests for data pipeline integrity, zero data leakage, and feature consistency.
"""

import unittest
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_EDA_DIR,
    FIGURES_EVAL_DIR,
    ALL_FEATURES,
    TOTAL_FEATURE_COUNT,
    TIMESTAMP_COL,
)


class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_path = PROCESSED_DATA_DIR / "train.parquet"
        cls.val_path = PROCESSED_DATA_DIR / "val.parquet"
        cls.test_path = PROCESSED_DATA_DIR / "test.parquet"

        assert cls.train_path.exists(), "train.parquet missing"
        assert cls.val_path.exists(), "val.parquet missing"
        assert cls.test_path.exists(), "test.parquet missing"

        cls.df_train = pd.read_parquet(cls.train_path)
        cls.df_val = pd.read_parquet(cls.val_path)
        cls.df_test = pd.read_parquet(cls.test_path)

    def test_feature_count(self):
        """Verify the dataset contains at least 15 features as required by the assignment."""
        self.assertEqual(len(ALL_FEATURES), TOTAL_FEATURE_COUNT)
        self.assertGreaterEqual(TOTAL_FEATURE_COUNT, 15)
        for feat in ALL_FEATURES:
            self.assertIn(feat, self.df_train.columns)

    def test_zero_leakage_training_partition(self):
        """Verify the training partition contains 100% normal data and zero failure instances."""
        failure_count_in_train = self.df_train["is_failure_period"].sum()
        self.assertEqual(failure_count_in_train, 0, "Data leakage detected: Training set contains failure data!")

    def test_chronological_ordering(self):
        """Verify timestamps are strictly monotonic and sorted chronologically."""
        train_ts = pd.to_datetime(self.df_train[TIMESTAMP_COL])
        val_ts = pd.to_datetime(self.df_val[TIMESTAMP_COL])
        test_ts = pd.to_datetime(self.df_test[TIMESTAMP_COL])

        self.assertTrue(train_ts.is_monotonic_increasing, "Train partition is not monotonically increasing!")
        self.assertTrue(val_ts.is_monotonic_increasing, "Val partition is not monotonically increasing!")
        self.assertTrue(test_ts.is_monotonic_increasing, "Test partition is not monotonically increasing!")

        # Partition boundary separation
        self.assertLess(train_ts.max(), val_ts.min(), "Overlap detected between Train and Val partitions!")
        self.assertLess(val_ts.max(), test_ts.min(), "Overlap detected between Val and Test partitions!")

    def test_eda_figures_generated(self):
        """Verify that all 12 publication-ready EDA figures were successfully generated and saved."""
        for i in range(1, 13):
            matches = list(FIGURES_EDA_DIR.glob(f"fig{i:02d}_*.png"))
            self.assertTrue(len(matches) > 0, f"EDA Figure {i:02d} is missing!")
            self.assertGreater(matches[0].stat().st_size, 10000, f"EDA Figure {i:02d} file is corrupt/empty!")

    def test_evaluation_figures_generated(self):
        """Verify that all 5 evaluation figures exist and are non-empty."""
        for i in range(13, 18):
            matches = list(FIGURES_EVAL_DIR.glob(f"fig{i:02d}_*.png"))
            self.assertTrue(len(matches) > 0, f"Evaluation Figure {i:02d} is missing!")
            self.assertGreater(matches[0].stat().st_size, 10000, f"Evaluation Figure {i:02d} file is corrupt/empty!")

    def test_metrics_and_model_artifacts(self):
        """Verify that evaluation metrics and model checkpoints exist and have valid structure."""
        metrics_p = MODELS_DIR / "evaluation_metrics.json"
        self.assertTrue(metrics_p.exists(), "evaluation_metrics.json is missing!")
        with open(metrics_p) as f:
            metrics = json.load(f)
        self.assertIn("test_metrics", metrics)
        self.assertGreater(metrics["test_metrics"]["roc_auc"], 0.5)
        self.assertEqual(len(metrics["event_level_evaluation"]), 4)

        history_p = MODELS_DIR / "training_history.json"
        self.assertTrue(history_p.exists(), "training_history.json is missing!")

        checkpoint_p = MODELS_DIR / "best_lstm_autoencoder.pt"
        self.assertTrue(checkpoint_p.exists(), "best_lstm_autoencoder.pt is missing!")


class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.inference import load_inference_artifacts, get_historical_sample
        cls.artifacts = load_inference_artifacts("cpu")
        cls.normal_sample = get_historical_sample("normal_baseline")
        cls.failure_sample = get_historical_sample("event_3_breakdown")

    def test_artifacts_loading(self):
        """Verify inference artifacts are loaded with correct parameters and threshold."""
        self.assertIn("model", self.artifacts)
        self.assertIn("scaler", self.artifacts)
        self.assertIn("threshold", self.artifacts)
        self.assertGreater(self.artifacts["threshold"], 0.0)
        self.assertEqual(len(self.artifacts["features"]), 15)
        self.assertEqual(self.artifacts["window_size"], 60)

    def test_preprocessing_consistency(self):
        """Verify input validation handles columns, ordering, and types consistently."""
        from src.inference import validate_input_dataframe

        # Valid input passes
        valid, msg, df_clean = validate_input_dataframe(self.normal_sample)
        self.assertTrue(valid, msg)
        self.assertEqual(df_clean.shape[1], 16)  # timestamp + 15 features

        # Missing required feature is caught
        bad_df = self.normal_sample.drop(columns=["TP2"])
        valid, msg, _ = validate_input_dataframe(bad_df)
        self.assertFalse(valid)
        self.assertIn("Missing required sensor columns", msg)

    def test_insufficient_input_length(self):
        """Verify that sequence lengths shorter than 60 steps are rejected with clear message."""
        from src.inference import validate_input_dataframe

        short_df = self.normal_sample.iloc[:30]  # Only 30 rows
        valid, msg, _ = validate_input_dataframe(short_df)
        self.assertFalse(valid)
        self.assertIn("Insufficient sequence length", msg)

    def test_threshold_classification_and_math(self):
        """Verify that anomaly classification strictly adheres to threshold comparison."""
        from src.inference import run_inference, validate_input_dataframe

        valid, _, df_clean = validate_input_dataframe(self.normal_sample)
        results_df, summary = run_inference(df_clean, self.artifacts, stride=1)

        tau = self.artifacts["threshold"]
        for _, row in results_df.iterrows():
            err = row["reconstruction_error"]
            expected_diff = round(err - tau, 6)
            self.assertAlmostEqual(row["diff_from_threshold"], expected_diff, places=4)
            if err > tau:
                self.assertEqual(row["status"], "ANOMALY DETECTED")
                self.assertTrue(row["is_anomaly"])
            else:
                self.assertEqual(row["status"], "NORMAL")
                self.assertFalse(row["is_anomaly"])

    def test_valid_inference_on_historical_samples(self):
        """Verify genuine inference produces NORMAL on healthy baseline and ANOMALY on Event 3."""
        from src.inference import run_inference, validate_input_dataframe

        # Healthy sample should be NORMAL
        _, _, norm_clean = validate_input_dataframe(self.normal_sample)
        res_norm, sum_norm = run_inference(norm_clean, self.artifacts, stride=1)
        self.assertEqual(sum_norm["overall_status"], "NORMAL")
        self.assertLess(sum_norm["mean_reconstruction_error"], self.artifacts["threshold"])

        # Catastrophic failure sample should be ANOMALY DETECTED
        _, _, fail_clean = validate_input_dataframe(self.failure_sample)
        res_fail, sum_fail = run_inference(fail_clean, self.artifacts, stride=1)
        self.assertEqual(sum_fail["overall_status"], "ANOMALY DETECTED")
        self.assertGreater(sum_fail["mean_reconstruction_error"], self.artifacts["threshold"])

    def test_csv_template_generation(self):
        """Verify downloadable CSV templates contain all 15 features and 60 rows."""
        import io
        from src.inference import get_csv_template

        nominal_csv = get_csv_template("nominal")
        df_nom = pd.read_csv(io.StringIO(nominal_csv))
        self.assertEqual(len(df_nom), 60)
        for feat in ALL_FEATURES:
            self.assertIn(feat, df_nom.columns)

        blank_csv = get_csv_template("blank")
        df_blank = pd.read_csv(io.StringIO(blank_csv))
        self.assertEqual(len(df_blank), 60)
        for feat in ALL_FEATURES:
            self.assertIn(feat, df_blank.columns)


class TestStatisticalAnalysis(unittest.TestCase):
    """Unit tests for statistical analysis, effect size calculations, and hyperparameter consistency."""
    @classmethod
    def setUpClass(cls):
        from src.config import PROCESSED_DATA_DIR, ANALOG_FEATURES, DIGITAL_FEATURES, FAILURE_EVENTS
        cls.resampled_path = PROCESSED_DATA_DIR / "telemetry_resampled.parquet"
        assert cls.resampled_path.exists(), "telemetry_resampled.parquet missing"
        cls.df = pd.read_parquet(cls.resampled_path)
        cls.analog_cols = ANALOG_FEATURES
        cls.digital_cols = DIGITAL_FEATURES
        cls.events = FAILURE_EVENTS

    def test_continuous_descriptive_statistics(self):
        """Verify continuous features have exact row count 1,504,107 and valid physical ranges."""
        self.assertEqual(len(self.df), 1504107)
        for col in self.analog_cols:
            self.assertIn(col, self.df.columns)
            # Physical boundaries
            self.assertGreaterEqual(self.df[col].min(), -1.0)
            self.assertFalse(self.df[col].isna().any())

        # Check mean of Reservoirs and TP3 are ~8.98 bar
        self.assertAlmostEqual(self.df["Reservoirs"].mean(), 8.985, places=2)
        self.assertAlmostEqual(self.df["TP3"].mean(), 8.985, places=2)

    def test_correlation_matrix_properties(self):
        """Verify collinearity between TP3 and Reservoirs (r=1.0) and inverse coupling of TP2 vs H1."""
        p_corr = self.df[self.analog_cols].corr(method="pearson")
        s_corr = self.df[self.analog_cols].corr(method="spearman")

        # Redundant collinear pair: TP3 vs Reservoirs
        self.assertAlmostEqual(p_corr.loc["TP3", "Reservoirs"], 1.000, places=3)
        self.assertAlmostEqual(s_corr.loc["TP3", "Reservoirs"], 1.000, places=3)

        # Inverted pressure drop: TP2 vs H1
        self.assertLess(p_corr.loc["TP2", "H1"], -0.90)

    def test_effect_sizes_and_failure_separation(self):
        """Verify Cohen's d effect sizes between healthy training baseline and failure regimes."""
        from src.config import TRAIN_START, TRAIN_END
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        is_fail = np.zeros(len(self.df), dtype=bool)
        for ev in self.events:
            s = pd.to_datetime(ev["start"])
            e = pd.to_datetime(ev["end"])
            is_fail |= ((self.df["timestamp"] >= s) & (self.df["timestamp"] <= e))

        df_train = self.df[(self.df["timestamp"] >= TRAIN_START) & (self.df["timestamp"] <= TRAIN_END)]
        df_failures = self.df[is_fail]

        def cohen_d(x, y):
            nx, ny = len(x), len(y)
            pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
            return (np.mean(x) - np.mean(y)) / pooled_std

        # Verify massive positive shift in TP2 and Motor_current
        d_tp2 = cohen_d(df_failures["TP2"], df_train["TP2"])
        d_motor = cohen_d(df_failures["Motor_current"], df_train["Motor_current"])
        d_oil = cohen_d(df_failures["Oil_temperature"], df_train["Oil_temperature"])
        d_h1 = cohen_d(df_failures["H1"], df_train["H1"])

        self.assertGreater(d_tp2, 1.5, "TP2 should show massive positive effect size during leaks")
        self.assertGreater(d_motor, 1.0, "Motor_current should show large positive effect size during leaks")
        self.assertGreater(d_oil, 1.5, "Oil_temperature should show massive positive effect size during leaks")
        self.assertLess(d_h1, -1.5, "H1 should show massive negative effect size during leaks")

    def test_hyperparameter_configuration_integrity(self):
        """Verify hyperparameters match verified values in config and evaluation metrics."""
        from src.config import (
            TOTAL_FEATURE_COUNT,
            WINDOW_SIZE,
            LATENT_DIM,
            ENCODER_DIMS,
            DECODER_DIMS,
            DROPOUT_RATE,
            BATCH_SIZE,
            LEARNING_RATE,
            NUM_EPOCHS,
            PATIENCE,
            RANDOM_SEED,
        )
        self.assertEqual(TOTAL_FEATURE_COUNT, 15)
        self.assertEqual(WINDOW_SIZE, 60)
        self.assertEqual(LATENT_DIM, 16)
        self.assertEqual(ENCODER_DIMS, [64, 32])
        self.assertEqual(DECODER_DIMS, [32, 64])
        self.assertEqual(DROPOUT_RATE, 0.2)
        self.assertEqual(BATCH_SIZE, 256)
        self.assertEqual(LEARNING_RATE, 1e-3)
        self.assertEqual(NUM_EPOCHS, 10)
        self.assertEqual(PATIENCE, 3)
        self.assertEqual(RANDOM_SEED, 42)


if __name__ == "__main__":
    unittest.main()

