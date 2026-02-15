#!/usr/bin/env python3
"""
train.py
Training pipeline for Customer Personality Analysis (target = 'Response').

Usage:
    python train.py
Outputs saved to ./models/:
 - scaler.pkl
 - feature_names.pkl
 - logistic_regression.pkl
 - decision_tree.pkl
 - knn.pkl
 - naive_bayes.pkl
 - random_forest.pkl
 - xgboost.pkl
 - model_comparison_results.csv
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    DATASET_FILE = "dataset.csv"
    TARGET_COLUMN = "Response"

    # Columns specific to your dataset
    ID_COL = "ID"
    DATE_COL = "Dt_Customer"  # will drop as requested

    BINARY_COLUMNS = [
        "Complain",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
    ]
    CATEGORICAL_COLUMNS = ["Education", "Marital_Status"]

    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    MODELS_REQUIRING_SCALING = ["Logistic Regression", "kNN"]

    METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def map_to_binary(series: pd.Series) -> pd.Series:
    """Map various representations of binary values to 0/1; unknown -> np.nan"""
    def mapper(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in {"1", "yes", "y", "true", "t", "1.0"}:
            return 1
        if s in {"0", "no", "n", "false", "f", "0.0"}:
            return 0
        try:
            # numeric-like
            v = float(s)
            return 1 if v == 1.0 else 0 if v == 0.0 else np.nan
        except Exception:
            return np.nan

    return series.map(mapper)


class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        path = self.cfg.DATA_DIR / self.cfg.DATASET_FILE
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pd.read_csv(path)
        print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[2/6] Preprocessing data...")

        # Drop ID if present
        if self.cfg.ID_COL in df.columns:
            df = df.drop(self.cfg.ID_COL, axis=1)
            print("   ✓ Dropped ID column")

        # Drop date column (Dt_Customer) as requested
        if self.cfg.DATE_COL in df.columns:
            df = df.drop(self.cfg.DATE_COL, axis=1)
            print("   ✓ Dropped Dt_Customer column")

        # Income -> numeric, fill missing with median (use assignment to avoid chained-assignment warning)
        if "Income" in df.columns:
            df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
            df["Income"] = df["Income"].fillna(df["Income"].median())
            print("   ✓ Converted Income to numeric and filled missing values")

        # Map binary columns robustly
        for col in self.cfg.BINARY_COLUMNS:
            if col in df.columns:
                df[col] = map_to_binary(df[col])
                # fill missing binary values with 0 (assume no if unknown). change if you prefer median/other.
                df[col] = df[col].fillna(0).astype(int)
        print(f"   ✓ Processed {len([c for c in self.cfg.BINARY_COLUMNS if c in df.columns])} binary cols")

        # Target mapping: support 1/0 or 'Yes'/'No'
        if self.cfg.TARGET_COLUMN in df.columns:
            df[self.cfg.TARGET_COLUMN] = map_to_binary(df[self.cfg.TARGET_COLUMN])
            # if still NaN convert to 0 (safe fallback); convert to int
            df[self.cfg.TARGET_COLUMN] = df[self.cfg.TARGET_COLUMN].fillna(0).astype(int)
            print(f"   ✓ Converted target ({self.cfg.TARGET_COLUMN}) to numeric")

        # One-hot encode categorical columns if present
        present_cat_cols = [c for c in self.cfg.CATEGORICAL_COLUMNS if c in df.columns]
        if present_cat_cols:
            df = pd.get_dummies(df, columns=present_cat_cols, drop_first=True)
            print(f"   ✓ One-hot encoded: {present_cat_cols}")

        # Remove any remaining non-numeric columns (defensive)
        non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
        if non_numeric:
            print(f"   ⚠ Dropping non-numeric columns (unexpected): {non_numeric}")
            df = df.drop(columns=non_numeric)

        print(f"   ✓ Final shape after preprocessing: {df.shape}")
        return df

    def prepare_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.Series, pd.Series]:
        print("\n[3/6] Preparing train/test split and scaling...")

        if self.cfg.TARGET_COLUMN not in df.columns:
            raise KeyError(f"Target column '{self.cfg.TARGET_COLUMN}' not found after preprocessing")

        X = df.drop(self.cfg.TARGET_COLUMN, axis=1)
        y = df[self.cfg.TARGET_COLUMN]

        print(f"   ✓ Features X: {X.shape}, Target y: {y.shape}")
        print(f"   ✓ Class distribution: {dict(y.value_counts())}")

        # Train/test split with stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.TEST_SIZE,
            random_state=self.cfg.RANDOM_STATE,
            stratify=y
        )

        # Fill missing numeric features with training medians (defensive)
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

        # Convert all to numeric dtype (float) for scaler and models
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        # Fit scaler on training numeric features and transform both
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save artifacts: scaler and feature names
        with open(self.cfg.MODEL_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.cfg.MODEL_DIR / "feature_names.pkl", "wb") as f:
            pickle.dump(list(X.columns), f)
        print("   ✓ Saved scaler and feature names to models/")

        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


class ModelManager:
    @staticmethod
    def get_models() -> Dict[str, Any]:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(random_state=Config.RANDOM_STATE, max_depth=10),
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, eval_metric="logloss", use_label_encoder=False)
        }
        return models


class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        # AUC may fail if y_proba is constant or single-class; handle exceptions
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = float("nan")
        metrics = {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "AUC": float(auc),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "MCC": float(matthews_corrcoef(y_true, y_pred))
        }
        # round for readability
        return {k: round(v, 6) if not (isinstance(v, float) and np.isnan(v)) else v for k, v in metrics.items()}


class ModelTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.evaluator = ModelEvaluator()
        self.results: List[Dict[str, Any]] = []

    def train_and_evaluate(self, models: Dict[str, Any],
                           X_train, X_test, X_train_scaled, X_test_scaled,
                           y_train, y_test) -> List[Dict[str, Any]]:
        print("\n[5/6] Training & evaluating models...")
        for name, model in models.items():
            print(f"\n-> Training {name} ...")

            # choose scaled or raw
            if name in self.cfg.MODELS_REQUIRING_SCALING:
                Xtr, Xte = X_train_scaled, X_test_scaled
            else:
                Xtr, Xte = X_train.values, X_test.values

            model.fit(Xtr, y_train)
            # Predictions and probabilities
            y_pred = model.predict(Xte)
            # handle predict_proba absence (most classifiers have it, but be defensive)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(Xte)[:, 1]
            else:
                # fallback to decision_function or use predicted labels
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(Xte)
                    # convert to 0-1 via min-max
                    smin, smax = scores.min(), scores.max()
                    if smax - smin != 0:
                        y_proba = (scores - smin) / (smax - smin)
                    else:
                        y_proba = np.zeros_like(scores)
                else:
                    y_proba = y_pred.astype(float)

            metrics = self.evaluator.calculate_metrics(y_test.values, y_pred, y_proba)
            self._print_metrics(metrics)

            # Save model
            model_file = self.cfg.MODEL_DIR / f"{name.replace(' ', '_').lower()}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            print(f"   ✓ Saved model -> {model_file.name}")

            # Save confusion matrix (printed)
            cm = confusion_matrix(y_test, y_pred)
            print(f"   Confusion Matrix:\n   {cm}")

            # Store results row
            row = {"Model": name, **metrics}
            self.results.append(row)

        return self.results

    @staticmethod
    def _print_metrics(metrics: Dict[str, float]):
        print("   Metrics:")
        for k, v in metrics.items():
            print(f"    - {k}: {v}")


class ResultsAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def create_results_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        # ensure metric columns order
        cols = ["Model"] + self.cfg.METRICS
        for c in self.cfg.METRICS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
        print("\n[6/6] Results summary:")
        print(df.to_string(index=False))
        return df

    def save_results(self, results_df: pd.DataFrame):
        out = self.cfg.MODEL_DIR / "model_comparison_results.csv"
        results_df.to_csv(out, index=False)
        print(f"   ✓ Saved results -> {out.name}")


class TrainingPipeline:
    def __init__(self):
        self.cfg = Config()
        self.loader = DataLoader(self.cfg)
        self.manager = ModelManager()
        self.trainer = ModelTrainer(self.cfg)
        self.analyzer = ResultsAnalyzer(self.cfg)

    def run(self):
        print("\n[1/6] Loading dataset...")
        df = self.loader.load_data()
        df_processed = self.loader.preprocess_data(df)
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = self.loader.prepare_train_test_split(df_processed)
        models = self.manager.get_models()
        results = self.trainer.train_and_evaluate(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
        results_df = self.analyzer.create_results_table(results)
        self.analyzer.save_results(results_df)
        print("\nTraining pipeline completed.")


if __name__ == "__main__":
    TrainingPipeline().run()