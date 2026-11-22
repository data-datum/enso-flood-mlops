from pathlib import Path
import duckdb
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import polars as pl
from src.models.backtest import (
    compute_flood_probability,
    prepare_pandas_xy,
    run_expanding_backtest,
    train_quantile_ensemble,
)
from src.models.features_config import CATEGORICAL_COLS, FEATURE_COLS

DB_PATH = Path("data/flood_monitor.duckdb")
TRAIN_PATH = Path("data/processed/train_multihorizon.parquet")
MODEL_NAME = "enso_flood_quantile_lgbm"


class MultiHorizonFloodWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models: dict, thresholds_map: dict[str, float]):
        self.models = models
        self.thresholds_map = thresholds_map

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        X = model_input[FEATURE_COLS].copy()
        for col in CATEGORICAL_COLS:
            X[col] = X[col].astype("category")

        p10 = self.models[0.10].predict(X)
        p50 = self.models[0.50].predict(X)
        p90 = self.models[0.90].predict(X)

        p10_clean = np.minimum(p10, p50)
        p90_clean = np.maximum(p90, p50)

        thresholds = model_input["station_id"].map(self.thresholds_map).fillna(5.0).to_numpy()
        flood_prob = compute_flood_probability(p10_clean, p50, p90_clean, thresholds)

        return pd.DataFrame(
            {
                "pred_p10": np.round(p10_clean, 4),
                "pred_p50": np.round(p50, 4),
                "pred_p90": np.round(p90_clean, 4),
                "flood_prob": np.round(flood_prob, 4),
            }
        )


def train_and_register_champion() -> None:
    # 1. Correr backtesting primero para registrar métricas de validación[cite: 1]
    cv_metrics = run_expanding_backtest(n_splits=3, test_days=60)

    # 2. Cargar dataset completo y cotas de alerta por estación
    train_df = pl.read_parquet(TRAIN_PATH)
    con = duckdb.connect(str(DB_PATH))
    stations = con.sql("SELECT station_id, CAST(alert_threshold_m AS DOUBLE) FROM stations").fetchall()
    con.close()
    thresholds_map = {row[0]: row[1] for row in stations}

    X_full, y_full = prepare_pandas_xy(train_df)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("enso-flood-15d-production")

    with mlflow.start_run(run_name="train_full_quantile_lgbm") as run:
        mlflow.log_metrics(cv_metrics)
        mlflow.log_param("train_rows", len(X_full))
        mlflow.log_param("features", ",".join(FEATURE_COLS))

        models = train_quantile_ensemble(X_full, y_full)
        wrapper = MultiHorizonFloodWrapper(models=models, thresholds_map=thresholds_map)

        sample_input = X_full.head(5)
        sample_output = wrapper.predict(None, sample_input)
        signature = mlflow.models.infer_signature(sample_input, sample_output)

        mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        # 3. Asignar alias @champion a la versión recién registrada[cite: 1]
        client = MlflowClient()
        latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(int(v.version) for v in latest_versions)
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=str(latest_version),
        )

        print(f"\nModelo '{MODEL_NAME}' v{latest_version} registrado y promovido a @champion (Run ID: {run.info.run_id}).")


if __name__ == "__main__":
    train_and_register_champion()