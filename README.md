# MLOps Experiment 5 - CKD Model Deployment as a REST API

## Aim
Train a machine-learning classification model and serve predictions through a REST API. This experiment uses a Chronic Kidney Disease (CKD) classification dataset.

## Work performed
- Loaded the UCI Chronic Kidney Disease dataset (dataset ID 336): 400 records and 24 input features.
- Cleaned input values, split the data into 320 training and 80 test records using a stratified split, and fitted preprocessing only on the training set.
- Used median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- Trained a Logistic Regression classifier and saved the preprocessing/model pipeline to `models/ckd_pipeline.joblib`.
- Implemented a FastAPI service that loads the saved pipeline at startup.
- Built a browser dashboard with a dark theme for input, predictions, model details, and API metrics.
- Verified two demonstration predictions (CKD and Not CKD), six invalid-input requests returning HTTP 422, operational metrics, and the existing pytest suite.
- Created Docker deployment configuration. Docker runtime/build and public API hosting were **not** verified.

## Recorded model results
The saved `reports/task1_metrics.json` contains these results on the 80-record test set:

| Metric | Recorded value |
| --- | ---: |
| Accuracy | 98.75% |
| CKD precision | 100.00% |
| CKD recall | 98.00% |
| CKD F1-score | 98.99% |
| ROC-AUC | 99.87% |

These are results for this small held-out dataset, **not clinical validation**.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Service information |
| GET | `/health` | API/model health |
| GET | `/ready` | Model readiness (optional bonus endpoint) |
| GET | `/model-info` | Model metadata and recorded evaluation metrics |
| POST | `/predict` | CKD / Not CKD model prediction (24 fields) |
| GET | `/metrics` | In-memory request and prediction counters |
| GET | `/dashboard` | Web dashboard |
| GET | `/docs` | Interactive FastAPI API documentation |

## Run locally (Windows PowerShell)
Open a terminal in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard` for the dashboard or `http://127.0.0.1:8000/docs` for Swagger UI. The saved `.joblib` model is loaded at startup. **Only load model files from trusted sources**; Joblib deserialization can execute code.

To run the existing automated tests in another terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

For a basic health check:

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Model predictions require the 24 features defined in `api/schemas.py`; use the dashboard's Example A / Example B controls for sample requests.

## Deployment status
The project includes `Dockerfile` and `.dockerignore` as a **deployment-ready configuration**. The final verification report records that Docker build/run were not performed. The running API was verified locally at `127.0.0.1:8000`; no public server deployment is claimed.

## Test evidence and report
- [Consolidated final experiment report](reports/Experiment5_Final_Report.pdf) (includes all three test-case PDFs as appendices).
- [CKD prediction PDF](reports/test_cases/01_CKD_Prediction.pdf)
- [Not CKD prediction PDF](reports/test_cases/02_Not_CKD_Prediction.pdf)
- [Invalid-input HTTP 422 PDF](reports/test_cases/03_Invalid_Input_422.pdf)
- [Recorded model metrics](reports/task1_metrics.json)

Repository: https://github.com/kasyap0420/mlops-experiment-5-ckd-prediction-api

**Limitations:** This is a lab demonstration using a small public dataset. Probabilities are model outputs, not medical diagnoses. Operational metrics are stored in memory and reset when the API process restarts.