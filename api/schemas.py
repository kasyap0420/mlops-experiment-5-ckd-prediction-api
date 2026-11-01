from math import isfinite
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# ALLOWED DATASET VALUES
# ============================================================

ALLOWED_SPECIFIC_GRAVITY = {
    "1.005",
    "1.010",
    "1.015",
    "1.020",
    "1.025",
}

ALLOWED_LEVELS = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
}

ALLOWED_NORMALITY = {
    "normal",
    "abnormal",
}

ALLOWED_PRESENCE = {
    "present",
    "notpresent",
}

ALLOWED_YES_NO = {
    "yes",
    "no",
}

ALLOWED_APPETITE = {
    "good",
    "poor",
}


# ============================================================
# NORMALIZATION / VALIDATION HELPERS
# ============================================================

def normalize_specific_gravity(value):
    """Validate specific gravity and match the original training categories."""
    if isinstance(value, bool):
        raise ValueError("Specific gravity must be a numeric value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Specific gravity must be 1.005, 1.010, 1.015, 1.020, or 1.025.") from exc
    allowed = {1.005, 1.010, 1.015, 1.020, 1.025}
    if not isfinite(numeric) or numeric not in allowed:
        raise ValueError("Specific gravity must be 1.005, 1.010, 1.015, 1.020, or 1.025.")
    # The original model was trained on float-to-string values, e.g. '1.02', not '1.020'.
    return str(numeric)


def normalize_level(value, field_name):
    """
    Normalize albumin/sugar levels to strings matching
    the categorical values used during model training.
    """

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be an integer level from 0 to 5."
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be an integer level from 0 to 5."
        )

    if not isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(
            f"{field_name} must be an integer level from 0 to 5."
        )

    normalized = str(int(numeric_value))

    if normalized not in ALLOWED_LEVELS:
        raise ValueError(
            f"Invalid {field_name}. Allowed values: 0, 1, 2, 3, 4, 5."
        )

    # The existing model was trained on numeric-string categories like '0.0', not '0'.
    return f"{int(numeric_value):.1f}"


def normalize_choice(value, field_name, allowed_values):
    """Normalize and validate a categorical value."""

    normalized = str(value).strip().lower()

    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))

        raise ValueError(
            f"Invalid {field_name}. Allowed values: {allowed}."
        )

    return normalized


# ============================================================
# PREDICTION REQUEST
# ============================================================

class CKDPredictionRequest(BaseModel):
    """
    Input schema for POST /predict.

    Contains exactly the 24 features used by the trained
    CKD classification pipeline.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    # --------------------------
    # Numeric features
    # --------------------------

    age: int = Field(ge=0, le=120,
        description="Age in years."
    )

    bp: int = Field(gt=0,
        description="Blood pressure in mm/Hg."
    )

    bgr: float = Field(gt=0,
        description="Random blood glucose."
    )

    bu: float = Field(gt=0,
        description="Blood urea."
    )

    sc: float = Field(gt=0,
        description="Serum creatinine."
    )

    sod: float = Field(gt=0,
        description="Sodium."
    )

    pot: float = Field(gt=0,
        description="Potassium."
    )

    hemo: float = Field(gt=0,
        description="Hemoglobin."
    )

    pcv: float = Field(gt=0,
        description="Packed cell volume."
    )

    wbcc: float = Field(gt=0,
        description="White blood cell count."
    )

    rbcc: float = Field(gt=0,
        description="Red blood cell count."
    )

    # --------------------------
    # Categorical features
    # --------------------------

    sg: str = Field(
        description=(
            "Specific gravity: "
            "1.005, 1.010, 1.015, 1.020, or 1.025."
        )
    )

    al: str = Field(
        description="Albumin level: 0, 1, 2, 3, 4, or 5."
    )

    su: str = Field(
        description="Sugar level: 0, 1, 2, 3, 4, or 5."
    )

    rbc: str = Field(
        description="Red blood cells: normal or abnormal."
    )

    pc: str = Field(
        description="Pus cell: normal or abnormal."
    )

    pcc: str = Field(
        description="Pus cell clumps: present or notpresent."
    )

    ba: str = Field(
        description="Bacteria: present or notpresent."
    )

    htn: str = Field(
        description="Hypertension: yes or no."
    )

    dm: str = Field(
        description="Diabetes mellitus: yes or no."
    )

    cad: str = Field(
        description="Coronary artery disease: yes or no."
    )

    appet: str = Field(
        description="Appetite: good or poor."
    )

    pe: str = Field(
        description="Pedal edema: yes or no."
    )

    ane: str = Field(
        description="Anemia: yes or no."
    )

    # ========================================================
    # FIELD VALIDATORS
    # ========================================================

    @field_validator("sg", mode="before")
    @classmethod
    def validate_sg(cls, value):
        return normalize_specific_gravity(value)

    @field_validator("al", mode="before")
    @classmethod
    def validate_al(cls, value):
        return normalize_level(value, "albumin")

    @field_validator("su", mode="before")
    @classmethod
    def validate_su(cls, value):
        return normalize_level(value, "sugar")

    @field_validator("rbc", mode="before")
    @classmethod
    def validate_rbc(cls, value):
        return normalize_choice(
            value,
            "rbc",
            ALLOWED_NORMALITY,
        )

    @field_validator("pc", mode="before")
    @classmethod
    def validate_pc(cls, value):
        return normalize_choice(
            value,
            "pc",
            ALLOWED_NORMALITY,
        )

    @field_validator("pcc", mode="before")
    @classmethod
    def validate_pcc(cls, value):
        return normalize_choice(
            value,
            "pcc",
            ALLOWED_PRESENCE,
        )

    @field_validator("ba", mode="before")
    @classmethod
    def validate_ba(cls, value):
        return normalize_choice(
            value,
            "ba",
            ALLOWED_PRESENCE,
        )

    @field_validator(
        "htn",
        "dm",
        "cad",
        "pe",
        "ane",
        mode="before",
    )
    @classmethod
    def validate_yes_no(cls, value, info):
        return normalize_choice(
            value,
            info.field_name,
            ALLOWED_YES_NO,
        )

    @field_validator("appet", mode="before")
    @classmethod
    def validate_appetite(cls, value):
        return normalize_choice(
            value,
            "appet",
            ALLOWED_APPETITE,
        )


# ============================================================
# PREDICTION RESPONSE
# ============================================================

class PredictionResponse(BaseModel):

    prediction: str
    model_version: str
    probabilities: dict[str, float]
    note: str


# ============================================================
# MODEL INFORMATION RESPONSE
# ============================================================

class ModelInfoResponse(BaseModel):

    model_name: str
    model_version: str
    algorithm: str
    number_of_features: int
    features: list[str]
    evaluation_metrics: dict[str, float]


# ============================================================
# OPERATIONAL METRICS RESPONSE
# ============================================================

class MetricsResponse(BaseModel):

    total_requests: int
    successful_predictions: int
    failed_requests: int
    average_response_time_ms: float
