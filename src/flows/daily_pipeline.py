from datetime import datetime, timezone
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from src.features.build_features import run_feature_pipeline
from src.inference.alerts import evaluate_alerts
from src.inference.predict_15d import run_daily_inference
from src.ingestion.ingest_daily import run_daily_ingestion
from src.models.evaluate_shadow import evaluate_and_promote_shadow

DB_PATH = Path("data/flood_monitor.duckdb")


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calcula el Population Stability Index (PSI) para detectar data drift."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    exp_counts = np.histogram(expected, bins=breakpoints)[0] / max(len(expected), 1)
    act_counts = np.histogram(actual, bins=breakpoints)[0] / max(len(actual), 1)

    exp_counts = np.clip(exp_counts, 1e-4, 1.0)
    act_counts = np.clip(act_counts, 1e-4, 1.0)
    return float(np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts)))


def check_feature_drift() -> dict[str, float]:
    """Compara los últimos 30 días de variables clave contra la ventana histórica previa."""
    con = duckdb.connect(str(DB_PATH))
    try:
        df = con.sql(
            """
            SELECT event_date, river_level_m, precip_acc_7d, oni_index
            FROM feature_store_daily
            ORDER BY event_date
            """
        ).df()
    finally:
        con.close()

    if len(df) < 120:
        return {}

    baseline = df.iloc[:-30]
    recent = df.iloc[-30:]

    drift_report: dict[str, float] = {}
    for col in ["river_level_m", "precip_acc_7d", "oni_index"]:
        psi = compute_psi(
            baseline[col].dropna().to_numpy(),
            recent[col].dropna().to_numpy(),
        )
        drift_report[col] = round(psi, 4)

    print(f"[Drift Monitor PSI 30d]: {drift_report}")
    return drift_report


def run_end_to_end_pipeline() -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"=== INICIANDO PIPELINE DIARIO ENSO-FLOOD ({started_at}) ===")

    # 1. Ingesta diaria de observaciones + pronóstico 15d + ONI
    run_daily_ingestion()

    # 2. Actualización del Feature Store sin data leakage
    run_feature_pipeline()

    # 3. Evaluación retrospectiva y promoción automática @shadow -> @champion
    retro_metrics = evaluate_and_promote_shadow()

    # 4. Chequeo de data drift (PSI)
    drift_metrics = check_feature_drift()

    # 5. Inferencia multi-horizonte 1..15d
    preds_df = run_daily_inference()

    # 6. Evaluación de alertas ENSO + cotas locales
    alerts = evaluate_alerts()

    print("=== PIPELINE DIARIO FINALIZADO CON ÉXITO ===")
    return {
        "started_at": started_at,
        "predictions_generated": len(preds_df),
        "retro_metrics": retro_metrics,
        "drift_psi": drift_metrics,
        "alerts_count": len(alerts),
    }


if __name__ == "__main__":
    run_end_to_end_pipeline()