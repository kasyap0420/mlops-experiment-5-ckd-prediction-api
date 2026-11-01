import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .schemas import CKDPredictionRequest


# Features expected by the trained model.
EXPECTED_FEATURES = {
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
}


class ModelService:
    """Loads the trained model and performs predictions."""

    def __init__(self, model_path: Path, metrics_path: Path):
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)

        self.model = None
        self.metadata = None
        self.feature_order = None

    def load(self) -> None:
        """Load and validate the saved model and Task 1 metadata."""

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        if not self.metrics_path.is_file():
            raise FileNotFoundError(
                f"Metrics file not found: {self.metrics_path}"
            )

        # Load the serialized model.
        loaded_model = joblib.load(self.model_path)

        # The saved artifact must be a complete scikit-learn pipeline.
        if not isinstance(loaded_model, Pipeline):
            raise TypeError(
                "The saved model is not a scikit-learn Pipeline."
            )

        # Verify that the expected pipeline steps exist.
        required_steps = {"preprocessor", "classifier"}
        actual_steps = set(loaded_model.named_steps.keys())

        if not required_steps.issubset(actual_steps):
            raise ValueError(
                "Saved pipeline does not contain the required "
                "'preprocessor' and 'classifier' steps."
            )

        # Verify that the trained model contains feature-name metadata.
        if not hasattr(loaded_model, "feature_names_in_"):
            raise ValueError(
                "Saved pipeline does not contain feature_names_in_."
            )

        model_features = [
            str(feature)
            for feature in loaded_model.feature_names_in_
        ]

        # The API must use exactly the same 24 features as the model.
        if len(model_features) != 24:
            raise ValueError(
                f"Expected 24 model features, found {len(model_features)}."
            )

        if set(model_features) != EXPECTED_FEATURES:
            raise ValueError(
                "Saved model features do not match the expected CKD "
                "features."
            )

        # Verify the classifier classes.
        classifier = loaded_model.named_steps["classifier"]

        if not hasattr(classifier, "classes_"):
            raise ValueError(
                "The saved classifier does not contain class metadata."
            )

        model_classes = {
            str(value)
            for value in classifier.classes_
        }

        expected_classes = {"ckd", "notckd"}

        if model_classes != expected_classes:
            raise ValueError(
                "Unexpected model classes. "
                f"Expected: {expected_classes}, "
                f"Found: {model_classes}"
            )

        # Load the evaluation and model metadata generated in Task 1.
        with self.metrics_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        required_metadata = {"dataset", "model", "metrics"}

        if not required_metadata.issubset(metadata.keys()):
            raise ValueError(
                "Task 1 metrics metadata is incomplete."
            )

        # Store everything only after all validation checks pass.
        self.model = loaded_model
        self.metadata = metadata
        self.feature_order = model_features

    def is_ready(self) -> bool:
        """Return True when the model has been loaded successfully."""

        return (
            self.model is not None
            and self.metadata is not None
            and self.feature_order is not None
        )

    def get_model_info(self) -> dict:
        """Return model information for the API."""

        if not self.is_ready():
            raise RuntimeError("Model service is not ready.")

        return {
            "model_name": "CKD Classification Model",
            "model_version": self.metadata["model"]["model_version"],
            "algorithm": self.metadata["model"]["algorithm"],
            "number_of_features": len(self.feature_order),
            "features": self.feature_order,
            "evaluation_metrics": {
                key: float(value)
                for key, value in self.metadata["metrics"].items()
            },
        }

    def predict(self, request_data: CKDPredictionRequest) -> dict:
        """Run a prediction using the saved preprocessing/model pipeline."""

        if not self.is_ready():
            raise RuntimeError("Model service is not ready.")

        # Convert the validated Pydantic request into a dictionary.
        input_data = request_data.model_dump()

        # Create a DataFrame using the exact feature order from training.
        dataframe = pd.DataFrame(
            [input_data],
            columns=self.feature_order,
        )

        # Confirm the API data matches the trained model's feature order.
        if list(dataframe.columns) != self.feature_order:
            raise ValueError(
                "API feature order does not match the trained model."
            )

        # The saved pipeline performs both preprocessing and prediction.
        prediction = self.model.predict(dataframe)[0]
        probabilities = self.model.predict_proba(dataframe)[0]

        classifier = self.model.named_steps["classifier"]

        classes = [
            str(value)
            for value in classifier.classes_
        ]

        probability_map = {
            class_name: float(probability)
            for class_name, probability in zip(
                classes,
                probabilities,
            )
        }

        # Convert internal dataset labels into API-friendly labels.
        if str(prediction) == "ckd":
            prediction_label = "CKD"
        elif str(prediction) == "notckd":
            prediction_label = "Not CKD"
        else:
            raise ValueError(
                f"Unexpected model prediction: {prediction}"
            )

        return {
            "prediction": prediction_label,
            "model_version": self.metadata["model"]["model_version"],
            "probabilities": {
                "CKD": probability_map.get("ckd", 0.0),
                "Not CKD": probability_map.get("notckd", 0.0),
            },
            "note": (
                "Model output/probability only; "
                "not a medical diagnosis."
            ),
        }