from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_VERSION = "1.0.0"

DATASET_ID = 336
DATASET_NAME = "Chronic Kidney Disease"
DATASET_SOURCE = (
    "UCI Machine Learning Repository - Chronic Kidney Disease "
    "(Dataset ID 336)"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "ckd_dataset.csv"
MODEL_PATH = MODEL_DIR / "ckd_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "task1_metrics.json"
RESULTS_PATH = REPORT_DIR / "task1_results.txt"


# ============================================================
# DATASET FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "age",
    "bp",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
]

CATEGORICAL_FEATURES = [
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "class"

TARGET_LABELS = [
    "notckd",
    "ckd",
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create required Task 1 directories."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():
    """Load the CKD dataset from the UCI Machine Learning Repository."""

    print("\n[1] Loading UCI Chronic Kidney Disease dataset...")

    dataset = fetch_ucirepo(id=DATASET_ID)

    X_raw = dataset.data.features.copy()
    y_raw = dataset.data.targets.squeeze().copy()

    print(f"    Raw rows: {len(X_raw)}")
    print(f"    Raw features: {X_raw.shape[1]}")

    return X_raw, y_raw


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(X_raw, y_raw):
    """
    Clean dataset representation.

    Missing values are represented as NaN and are handled later
    by the fitted scikit-learn preprocessing pipeline.
    """

    print("\n[2] Cleaning dataset...")

    X = X_raw.copy()
    y = y_raw.copy()

    # Normalize column names.
    X.columns = [
        str(column).strip().lower()
        for column in X.columns
    ]

    y.name = TARGET_COLUMN

    # Verify the expected input columns.
    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "Expected dataset columns are missing: "
            + ", ".join(missing_features)
        )

    # Keep exactly the 24 expected input features.
    X = X[FEATURE_COLUMNS].copy()

    # Clean every value:
    # - blank values -> NaN
    # - '?' -> NaN
    # - text -> stripped and lowercase
    for column in X.columns:
        X[column] = X[column].map(
            lambda value: (
                np.nan
                if pd.isna(value)
                or str(value).strip() in {"", "?"}
                else str(value).strip().lower()
            )
        )

    # Convert numeric columns to numeric values.
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Clean target values.
    y = y.map(
        lambda value: (
            np.nan
            if pd.isna(value)
            else str(value).strip().lower()
        )
    )

    # Keep only valid CKD target classes.
    valid_rows = y.isin(TARGET_LABELS)

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    if X.empty:
        raise ValueError(
            "No valid records remain after cleaning."
        )

    if y.nunique() != 2:
        raise ValueError(
            "Expected exactly two target classes, found: "
            f"{sorted(y.dropna().unique())}"
        )

    missing_values = int(
        X.isna().sum().sum()
    )

    print(f"    Clean rows: {len(X)}")
    print(f"    Features used: {X.shape[1]}")
    print(
        "    Missing values before pipeline preprocessing: "
        f"{missing_values}"
    )

    print("\n    Class distribution:")
    print(y.value_counts())

    return X, y, missing_values


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def build_pipeline():
    """
    Build one complete preprocessing + model pipeline.

    Numeric:
        median imputation
        standard scaling

    Categorical:
        most-frequent imputation
        one-hot encoding

    Model:
        Logistic Regression
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


# ============================================================
# TRAIN / TEST SPLIT AND TRAINING
# ============================================================

def train_model(X, y):
    """Split the dataset and train the complete ML pipeline."""

    print("\n[3] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"    Training samples: {len(X_train)}"
    )
    print(
        f"    Testing samples:  {len(X_test)}"
    )

    print(
        "\n[4] Building and training ML pipeline..."
    )

    pipeline = build_pipeline()

    # All preprocessing is fitted only on training data.
    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "    Training completed successfully."
    )

    return (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    pipeline,
    X_test,
    y_test,
):
    """Evaluate the trained model."""

    print("\n[5] Evaluating model...")

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)

    classifier = pipeline.named_steps["classifier"]

    class_names = list(
        classifier.classes_
    )

    if "ckd" not in class_names:
        raise ValueError(
            "The trained model does not contain the "
            f"'ckd' class. Classes: {class_names}"
        )

    ckd_index = class_names.index("ckd")

    y_test_binary = (
        y_test == "ckd"
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="ckd",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="ckd",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="ckd",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test_binary,
        y_probability[:, ckd_index],
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=TARGET_LABELS,
        target_names=[
            "Not CKD",
            "CKD",
        ],
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=TARGET_LABELS,
    )

    print(
        f"    Accuracy : {accuracy:.4f}"
    )
    print(
        f"    Precision: {precision:.4f}"
    )
    print(
        f"    Recall   : {recall:.4f}"
    )
    print(
        f"    F1-score : {f1:.4f}"
    )
    print(
        f"    ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        "\n    Classification Report:"
    )
    print(report)

    print(
        "    Confusion Matrix"
    )
    print(
        "    Class order: notckd, ckd"
    )
    print(confusion)

    metrics = {
        "accuracy": float(accuracy),
        "precision_ckd": float(precision),
        "recall_ckd": float(recall),
        "f1_ckd": float(f1),
        "roc_auc": float(roc_auc),
    }

    return (
        metrics,
        report,
        confusion,
        y_pred,
    )


# ============================================================
# MODEL SAVING AND RELOAD VERIFICATION
# ============================================================

def save_and_verify_model(
    pipeline,
    X_test,
    y_pred,
):
    """
    Save the complete preprocessing + model pipeline and verify
    that reloading it produces identical predictions.
    """

    print(
        "\n[6] Saving trained pipeline..."
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
        compress=3,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file was not created: "
            f"{MODEL_PATH}"
        )

    file_size_kb = (
        MODEL_PATH.stat().st_size / 1024
    )

    print(
        f"    Saved to: {MODEL_PATH}"
    )
    print(
        f"    File size: {file_size_kb:.2f} KB"
    )

    print(
        "\n[7] Reloading saved pipeline..."
    )

    loaded_pipeline = joblib.load(
        MODEL_PATH
    )

    reloaded_predictions = (
        loaded_pipeline.predict(X_test)
    )

    predictions_match = np.array_equal(
        y_pred,
        reloaded_predictions,
    )

    if not predictions_match:
        raise RuntimeError(
            "Predictions from the reloaded model "
            "do not match the original predictions."
        )

    # Perform one actual inference after reload.
    sample = X_test.iloc[[0]]

    sample_prediction = (
        loaded_pipeline.predict(sample)[0]
    )

    sample_probabilities = (
        loaded_pipeline.predict_proba(sample)[0]
    )

    loaded_classifier = (
        loaded_pipeline.named_steps["classifier"]
    )

    loaded_classes = list(
        loaded_classifier.classes_
    )

    probability_map = {
        class_name: float(probability)
        for class_name, probability in zip(
            loaded_classes,
            sample_probabilities,
        )
    }

    print(
        "    Model reload: SUCCESS"
    )
    print(
        "    Prediction consistency: SUCCESS"
    )
    print(
        "    Sample prediction after reload: "
        f"{sample_prediction}"
    )
    print(
        "    Sample model probabilities: "
        f"{probability_map}"
    )

    return (
        sample_prediction,
        probability_map,
    )


# ============================================================
# REPORT GENERATION
# ============================================================

def save_reports(
    X,
    X_train,
    X_test,
    metrics,
    report,
    confusion,
    missing_values,
):
    """Generate lab-record-friendly and JSON reports."""

    metrics_data = {
        "experiment": (
            "MLOps Experiment 5 - Task 1"
        ),
        "dataset": {
            "name": DATASET_NAME,
            "uci_dataset_id": DATASET_ID,
            "source": DATASET_SOURCE,
            "rows_after_cleaning": int(
                len(X)
            ),
            "input_features": int(
                X.shape[1]
            ),
            "target": TARGET_COLUMN,
            "classes": TARGET_LABELS,
        },
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "categorical_imputation": (
                "most_frequent"
            ),
            "categorical_encoding": (
                "OneHotEncoder"
            ),
            "unknown_categories": "ignore",
            "missing_values_before_pipeline": (
                missing_values
            ),
            "preprocessing_inside_pipeline": True,
        },
        "split": {
            "test_size": TEST_SIZE,
            "train_samples": int(
                len(X_train)
            ),
            "test_samples": int(
                len(X_test)
            ),
            "stratified": True,
            "random_state": RANDOM_STATE,
        },
        "model": {
            "algorithm": "Logistic Regression",
            "model_version": MODEL_VERSION,
        },
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix_class_order": (
            TARGET_LABELS
        ),
        "confusion_matrix": (
            confusion.tolist()
        ),
        "saved_model": str(
            MODEL_PATH
        ),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_data,
            file,
            indent=2,
        )

    results_text = f"""
============================================================
MLOps Experiment 5 - Task 1
CKD Model Preparation
============================================================

DATASET
------------------------------------------------------------
Name              : {DATASET_NAME}
UCI Dataset ID    : {DATASET_ID}
Source            : {DATASET_SOURCE}
Rows after clean  : {len(X)}
Input features    : {X.shape[1]}
Target            : {TARGET_COLUMN}
Classes           : notckd, ckd

PREPROCESSING
------------------------------------------------------------
Numeric imputation     : Median
Numeric scaling        : StandardScaler
Categorical imputation : Most Frequent
Categorical encoding   : OneHotEncoder
Unknown categories     : Ignored
Pipeline preprocessing : Yes

MISSING VALUES
------------------------------------------------------------
Missing values before pipeline:
{missing_values}

TRAIN / TEST SPLIT
------------------------------------------------------------
Training samples : {len(X_train)}
Testing samples  : {len(X_test)}
Test size        : {TEST_SIZE}
Stratified       : Yes
Random state     : {RANDOM_STATE}

MODEL
------------------------------------------------------------
Algorithm     : Logistic Regression
Model version : {MODEL_VERSION}

EVALUATION
------------------------------------------------------------
Accuracy        : {metrics["accuracy"]:.4f}
Precision (CKD) : {metrics["precision_ckd"]:.4f}
Recall (CKD)    : {metrics["recall_ckd"]:.4f}
F1-score (CKD)  : {metrics["f1_ckd"]:.4f}
ROC-AUC         : {metrics["roc_auc"]:.4f}

CLASSIFICATION REPORT
------------------------------------------------------------
{report}

CONFUSION MATRIX
------------------------------------------------------------
Class order: notckd, ckd

{confusion}

SAVED MODEL
------------------------------------------------------------
{MODEL_PATH}

MODEL RELOAD VERIFICATION
------------------------------------------------------------
SUCCESS - Saved pipeline loaded successfully.
SUCCESS - Reloaded pipeline produced identical
predictions on the test set.

============================================================
"""

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            results_text.strip()
            + "\n"
        )

    print(
        "\n[8] Reports saved:"
    )
    print(
        f"    {METRICS_PATH}"
    )
    print(
        f"    {RESULTS_PATH}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print(
        "MLOps Experiment 5 - Task 1"
    )
    print(
        "CKD Dataset + ML Model Preparation"
    )
    print("=" * 60)

    create_directories()

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    X_raw, y_raw = load_dataset()

    # Save original fetched dataset snapshot.
    raw_snapshot = X_raw.copy()
    raw_snapshot[TARGET_COLUMN] = (
        y_raw.values
    )

    raw_snapshot.to_csv(
        RAW_DATA_PATH,
        index=False,
    )

    print(
        "\n    Raw dataset snapshot saved to:"
    )
    print(
        f"    {RAW_DATA_PATH}"
    )

    # --------------------------------------------------------
    # 2. CLEAN DATA
    # --------------------------------------------------------

    (
        X,
        y,
        missing_values,
    ) = clean_dataset(
        X_raw,
        y_raw,
    )

    # --------------------------------------------------------
    # 3. TRAIN MODEL
    # --------------------------------------------------------

    (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_model(
        X,
        y,
    )

    # --------------------------------------------------------
    # 4. EVALUATE MODEL
    # --------------------------------------------------------

    (
        metrics,
        report,
        confusion,
        y_pred,
    ) = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # 5. SAVE + RELOAD + VERIFY MODEL
    # --------------------------------------------------------

    save_and_verify_model(
        pipeline,
        X_test,
        y_pred,
    )

    # --------------------------------------------------------
    # 6. SAVE REPORTS
    # --------------------------------------------------------

    save_reports(
        X=X,
        X_train=X_train,
        X_test=X_test,
        metrics=metrics,
        report=report,
        confusion=confusion,
        missing_values=missing_values,
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "TASK 1 COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)

    print("\nGenerated artifacts:")
    print(
        f"1. Dataset : {RAW_DATA_PATH}"
    )
    print(
        f"2. Model   : {MODEL_PATH}"
    )
    print(
        f"3. Metrics : {METRICS_PATH}"
    )
    print(
        f"4. Report  : {RESULTS_PATH}"
    )

    print(
        "\nThe trained Joblib pipeline is ready "
        "for Task 2."
    )


if __name__ == "__main__":
    main()