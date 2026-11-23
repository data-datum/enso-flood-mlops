# ENSO & Paraná River 15-Day Probabilistic Flood Early-Warning MLOps Pipeline

[![Daily ENSO Flood Forecast Pipeline](https://github.com/data-datum/enso-flood-mlops/actions/workflows/daily_forecast.yml/badge.svg)](https://github.com/data-datum/enso-flood-mlops/actions/workflows/daily_forecast.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Quantile-success.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-Feature%20Store-FFF000.svg)
![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

An end-to-end **MLOps & Operational Hydrology** system delivering **15-day multi-horizon probabilistic river-level forecasts** (`P10`, `P50`, `P90`) for key gauging stations along the Upper and Middle **Paraná River basin** in Argentina, dynamically coupled with the **El Niño–Southern Oscillation (ENSO / ONI)** climate regime.

---

## Problem Statement & Domain Context

Flooding along the Paraná River basin exhibits strong non-linear memory driven by upstream rainfall accumulation and low-frequency ocean-atmosphere teleconnections (**ENSO**). Deterministic point forecasts fail to capture tail risk during **El Niño** phases, while static alert thresholds cause either late evacuations or excessive false alarms during **Neutral / La Niña** years.

This system bridges operational hydrology and production MLOps by:
1. Predicting full **15-day predictive distributions** (`P10`, `P50`, `P90`) and threshold exceedance probabilities (`flood_prob`) for each station and lead time ($h = +1 \dots +15$ days).
2. Adapting decision thresholds dynamically according to the active **Oceanic Niño Index (ONI)** phase.
3. Automating daily ingestion, feature engineering, shadow model evaluation, data drift monitoring, REST serving, and alerting.

---

## System Architecture

```text
[INA Hydrometric API + 15d Precipitation Forecast + NOAA ONI Index]
                                │
                                ▼
               src/ingestion/ingest_daily.py
        (Daily API pull + operational forward-fill fallback)
                                │
                                ▼
          DuckDB Feature Store (feature_store_daily)
           ├── River stage lags (1d, 3d, 7d, 14d, 30d)
           ├── Accumulated rainfall (3d, 7d, 15d, 30d)
           └── ENSO regime features (oni_index, oni_trend_90d, is_el_nino)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
[Shadow Eval & Drift Monitor]               [15-Day Quantile Inference]
 MLflow Registry (@champion / @shadow)       src/inference/predict_15d.py
 Retrospective MAE & Pinball Loss            LightGBM Quantile (P10/P50/P90)
 30d PSI Feature Drift Check                          │
          │                                           ▼
          │                                  Cascade Alerting Engine
          └─────────────────────┬──────────── src/inference/alerts.py
                                ▼           (ENSO Regime + PNA Stages)
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       Telegram Alert Bot           FastAPI Service & Metrics
    (Automated 15d Bulletins)        ├── GET /health
                                     ├── GET /forecast/{station_id}
                                     ├── GET /alerts/latest
                                     └── GET /metrics (Prometheus)


```


## Repository Structure

```text
enso-flood-mlops/
├── .github/
│   └── workflows/
│       └── daily_forecast.yml      # Automated daily cron execution (16:00 UTC)
├── data/
│   ├── parquet_backup/             # Partitioned historical backups
│   ├── processed/                  # Processed training & feature datasets
│   └── quarantine/                 # Data contract validation rejects
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI service (/health, /forecast, /alerts/latest, /metrics)
│   ├── features/
│   │   └── build_features.py       # Polars/DuckDB feature engineering & 15d inference matrix
│   ├── flows/
│   │   └── daily_pipeline.py       # Unified daily end-to-end MLOps orchestrator
│   ├── inference/
│   │   ├── alerts.py               # ENSO-adaptive threshold logic & Telegram alert formatter
│   │   └── predict_15d.py          # Multi-horizon P10/P50/P90 inference & flood probability
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── fetch_ina.py            # Instituto Nacional del Agua (INA) API connector
│   │   └── ingest_daily.py         # Daily ingestion with operational forward-fill fallback
│   └── models/
│       ├── evaluate_shadow.py      # Retrospective evaluation & @shadow -> @champion promotion
│       └── train.py                # Expanding-window LightGBM quantile model training
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md

```


## Quickstart & Deployment

### 1. Local Installation

```text
git clone [https://github.com/data-datum/enso-flood-mlops.git](https://github.com/data-datum/enso-flood-mlops.git)
cd enso-flood-mlops
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Daily End-to-End Pipeline
Executes daily ingestion, feature store update, retrospective shadow scoring, 30-day PSI drift monitoring, 15-day quantile inference, and alert evaluation:


```text
python -m src.flows.daily_pipeline
```

### 3. Launch the FastAPI Service



```text
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```


Swagger UI: http://127.0.0.1:8000/docs

Latest Alerts JSON: http://127.0.0.1:8000/alerts/latest

Prometheus Metrics: http://127.0.0.1:8000/metrics



