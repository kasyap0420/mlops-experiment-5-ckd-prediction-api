import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .metrics import APIMetrics
from .model_service import ModelService
from .schemas import (
    CKDPredictionRequest,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
)


# Get the project root from the location of this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "ckd_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "task1_metrics.json"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


# Configure basic application logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ckd_prediction_api")


# Create the services used by the API.
model_service = ModelService(
    model_path=MODEL_PATH,
    metrics_path=METRICS_PATH,
)

api_metrics = APIMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and validate the ML model when the API starts."""

    logger.info("Loading model from: %s", MODEL_PATH)

    try:
        model_service.load()
    except Exception:
        logger.exception("Model loading failed.")
        raise

    logger.info("Model loaded successfully.")

    yield

    logger.info("API application shutting down.")


app = FastAPI(
    title="CKD Prediction API",
    description=(
        "MLOps Experiment 5 - Model Deployment as a REST API. "
        "Provides CKD classification using a trained machine learning pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_metrics_middleware(
    request: Request,
    call_next,
):
    """Record request count, failures, and response time."""

    start_time = perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        elapsed_ms = (perf_counter() - start_time) * 1000

        api_metrics.record_request(
            status_code=500,
            response_time_ms=elapsed_ms,
        )

        raise

    elapsed_ms = (perf_counter() - start_time) * 1000

    api_metrics.record_request(
        status_code=response.status_code,
        response_time_ms=elapsed_ms,
    )

    return response


# Serve the dashboard with FastAPI; no separate frontend server or CORS setup.
app.mount(
    "/dashboard-assets",
    StaticFiles(directory=DASHBOARD_DIR),
    name="dashboard-assets",
)


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/", tags=["Service"])
async def root():
    """Return basic API information."""

    return {
        "service": "CKD Prediction API",
        "status": "running",
        "api_version": app.version,
        "model_version": (
            model_service.metadata["model"]["model_version"]
            if model_service.is_ready()
            else None
        ),
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "model_info": "/model-info",
        "prediction": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["Service"])
async def health():
    """Return the current health status of the service."""

    if not model_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service is not ready because the model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/ready", tags=["Service"])
async def ready():
    """Return whether the service is ready to make predictions."""

    if not model_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    return {
        "ready": True,
        "model_loaded": True,
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
async def model_info():
    """Return model metadata and Task 1 evaluation metrics."""

    if not model_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    return model_service.get_model_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict(
    request_data: CKDPredictionRequest,
):
    """Return a CKD or Not CKD prediction."""

    if not model_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    try:
        result = model_service.predict(request_data)

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed because the model "
                "could not process the supplied data."
            ),
        ) from exc

    api_metrics.record_successful_prediction()

    return result


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Monitoring"],
)
async def metrics():
    """Return basic operational API metrics."""

    return api_metrics.snapshot()
