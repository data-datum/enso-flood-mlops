from pathlib import Path
import duckdb
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

DB_PATH = Path("data/flood_monitor.duckdb")
MODEL_NAME = "enso_flood_quantile_lgbm"
MIN_EVAL_SAMPLES = 30  # Mínimo de pares predicción-realidad para promover @shadow


def evaluate_and_promote_shadow() -> dict[str, float]:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    con = duckdb.connect(str(DB_PATH))
    try:
        # Cruzar prediction_logs contra realidad observada en raw_observations
        eval_pdf = con.sql(
            """
            SELECT
                CAST(p.forecast_issued_at AS DATE) AS issued_date,
                p.target_date,
                p.station_id,
                p.horizon_day,
                p.pred_p10,
                p.pred_p50,
                p.pred_p90,
                p.flood_prob,
                p.model_version,
                p.model_alias,
                CAST(r.river_level_m AS DOUBLE) AS y_true
            FROM prediction_logs p
            INNER JOIN raw_observations r
                ON p.station_id = r.station_id
               AND p.target_date = CAST(r.event_date AS DATE)
            WHERE r.river_level_m IS NOT NULL
            """
        ).to_df()
    finally:
        con.close()

    if eval_pdf.empty:
        print("Aún no hay predicciones vencidas (target_date <= hoy) para evaluar retrospectivamente.")
        return {}

    metrics_by_alias: dict[str, dict[str, float]] = {}

    for alias, grp in eval_pdf.groupby("model_alias"):
        yt = grp["y_true"].to_numpy()
        mae = float(mean_absolute_error(yt, grp["pred_p50"].to_numpy()))
        pb10 = float(mean_pinball_loss(yt, grp["pred_p10"].to_numpy(), alpha=0.10))
        pb50 = float(mean_pinball_loss(yt, grp["pred_p50"].to_numpy(), alpha=0.50))
        pb90 = float(mean_pinball_loss(yt, grp["pred_p90"].to_numpy(), alpha=0.90))
        pinball_avg = (pb10 + pb50 + pb90) / 3.0

        metrics_by_alias[str(alias)] = {
            "n_samples": float(len(grp)),
            "mae": mae,
            "pinball_avg": pinball_avg,
        }
        print(
            f"[Retrospectiva {alias}] muestras={len(grp)} | "
            f"MAE={mae:.4f} m | Pinball_avg={pinball_avg:.4f}"
        )

    # Lógica de promoción @shadow -> @champion si supera estadísticamente al campeón
    if "@champion" in metrics_by_alias and "@shadow" in metrics_by_alias:
        champ = metrics_by_alias["@champion"]
        shad = metrics_by_alias["@shadow"]

        if shad["n_samples"] >= MIN_EVAL_SAMPLES:
            improves_pinball = shad["pinball_avg"] < champ["pinball_avg"] * 0.99
            improves_mae = shad["mae"] <= champ["mae"]

            if improves_pinball and improves_mae:
                try:
                    shadow_mv = client.get_model_version_by_alias(name=MODEL_NAME, alias="shadow")
                    client.set_registered_model_alias(
                        name=MODEL_NAME,
                        alias="champion",
                        version=str(shadow_mv.version),
                    )
                    client.delete_registered_model_alias(name=MODEL_NAME, alias="shadow")
                    print(
                        f"PROMOCIÓN AUTOMÁTICA: @shadow (v{shadow_mv.version}) superó a @champion "
                        f"y fue promovido a @champion."
                    )
                except MlflowException as exc:
                    print(f"No se pudo completar la promoción en MLflow: {exc}")

    return metrics_by_alias.get("@champion", {})


if __name__ == "__main__":
    evaluate_and_promote_shadow()