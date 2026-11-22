from datetime import datetime, timezone
from pathlib import Path
import duckdb
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import pandas as pd
import polars as pl
from src.features.build_features import build_inference_matrix
from src.models.features_config import FEATURE_COLS

DB_PATH = Path("data/flood_monitor.duckdb")
MODEL_NAME = "enso_flood_quantile_lgbm"
ALIASES_TO_RUN = ["champion", "shadow"]


def ensure_prediction_logs_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            forecast_issued_at TIMESTAMP,
            issued_date DATE,
            target_date DATE,
            station_id VARCHAR,
            horizon_day INTEGER,
            pred_p10 DOUBLE,
            pred_p50 DOUBLE,
            pred_p90 DOUBLE,
            flood_prob DOUBLE,
            model_version VARCHAR,
            model_alias VARCHAR
        )
        """
    )


def run_daily_inference(issued_date: str | None = None) -> pd.DataFrame:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    con = duckdb.connect(str(DB_PATH))
    try:
        ensure_prediction_logs_table(con)

        if issued_date is None:
            issued_date = str(con.sql("SELECT MAX(issued_date) FROM forecast_inputs").fetchall()[0][0])

        fs_df = pl.from_arrow(con.sql("SELECT * FROM feature_store_daily").to_arrow_table())
        inf_pl = build_inference_matrix(con, fs_df, issued_date=issued_date)

        if inf_pl.is_empty():
            raise ValueError(f"No se pudo construir matriz de inferencia para issued_date={issued_date}.")

        inf_pdf = inf_pl.to_pandas()
        X_inf = inf_pdf[FEATURE_COLS].copy()
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        logged_frames: list[pd.DataFrame] = []

        for alias in ALIASES_TO_RUN:
            try:
                mv = client.get_model_version_by_alias(name=MODEL_NAME, alias=alias)
                model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
            except MlflowException:
                # Si no existe @shadow todavía, continúa solo con @champion
                continue

            preds = model.predict(X_inf)


            log_df = pd.DataFrame(
                {
                    "forecast_issued_at": now_utc,
                    "target_date": pd.to_datetime(inf_pdf["target_date"]).dt.date,
                    "station_id": inf_pdf["station_id"],
                    "horizon_day": inf_pdf["horizon_day"].astype(int),
                    "pred_p10": preds["pred_p10"].astype(float),
                    "pred_p50": preds["pred_p50"].astype(float),
                    "pred_p90": preds["pred_p90"].astype(float),
                    "flood_prob": preds["flood_prob"].astype(float),
                    "model_version": str(mv.version),
                    "model_alias": f"@{alias}",
                }
            )

            # Evitar duplicados si se re-ejecuta el mismo día para el mismo alias
            con.execute(
                "DELETE FROM prediction_logs WHERE CAST(forecast_issued_at AS DATE) = CURRENT_DATE AND model_alias = ?",
                [f"@{alias}"],
            )
            con.register("temp_preds", log_df)
            con.execute(
                """
                INSERT INTO prediction_logs (
                    forecast_issued_at, target_date, station_id, horizon_day,
                    pred_p10, pred_p50, pred_p90, flood_prob, model_version, model_alias
                )
                SELECT
                    forecast_issued_at, target_date, station_id, horizon_day,
                    pred_p10, pred_p50, pred_p90, flood_prob, model_version, model_alias
                FROM temp_preds
                """
            )

        return pd.concat(logged_frames, ignore_index=True) if logged_frames else pd.DataFrame()
    finally:
        con.close()


if __name__ == "__main__":
    run_daily_inference()