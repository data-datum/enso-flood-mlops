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
