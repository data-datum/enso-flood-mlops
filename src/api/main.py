from pathlib import Path
import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from src.inference.alerts import classify_enso_regime, evaluate_alerts

DB_PATH = Path("data/flood_monitor.duckdb")

app = FastAPI(
    title="ENSO & Paraná-Uruguay 15-Day Flood Forecasting API",
    version="1.0.0",
    description="Pronóstico hidrológico cuantílico (P10/P50/P90) acoplado al índice ENSO (ONI).",
)


@app.get("/health")
def health_check() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        latest_obs = con.sql("SELECT MAX(event_date) FROM feature_store_daily").fetchall()[0][0]
        latest_fc = con.sql(
            "SELECT MAX(CAST(forecast_issued_at AS DATE)) FROM prediction_logs"
        ).fetchall()[0][0]
    finally:
        con.close()

    return {
        "status": "ok",
        "latest_observation_date": str(latest_obs),
        "latest_forecast_date": str(latest_fc),
    }


@app.get("/forecast/{station_id}")
def get_station_forecast(station_id: str, alias: str = "@champion") -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        latest_date = con.sql(
            "SELECT MAX(CAST(forecast_issued_at AS DATE)) FROM prediction_logs"
        ).fetchall()[0][0]

        df = con.execute(
            """
            SELECT
                p.station_id,
                p.horizon_day,
                CAST(p.target_date AS VARCHAR) AS target_date,
                p.pred_p10,
                p.pred_p50,
                p.pred_p90,
                p.flood_prob,
                p.model_version,
                p.model_alias,
                CAST(s.alert_threshold_m AS DOUBLE) AS alert_threshold_m,
                CAST(s.evacuation_threshold_m AS DOUBLE) AS evacuation_threshold_m
            FROM prediction_logs p
            INNER JOIN stations s ON p.station_id = s.station_id
            WHERE p.station_id = ?
              AND p.model_alias = ?
              AND CAST(p.forecast_issued_at AS DATE) = ?
            ORDER BY p.horizon_day
            """,
            [station_id.upper(), alias, latest_date],
        ).df()
    finally:
        con.close()

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin pronósticos para estación '{station_id}'.")

    return {
        "station_id": station_id.upper(),
        "issued_date": str(latest_date),
        "model_alias": alias,
        "model_version": df["model_version"].iloc[0],
        "alert_threshold_m": float(df["alert_threshold_m"].iloc[0]),
        "evacuation_threshold_m": float(df["evacuation_threshold_m"].iloc[0]),
        "forecast_15d": df[
            ["horizon_day", "target_date", "pred_p10", "pred_p50", "pred_p90", "flood_prob"]
        ].to_dict(orient="records"),
    }


@app.get("/alerts/latest")
def get_latest_alerts() -> dict:
    messages = evaluate_alerts()
    return {"alerts": messages}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        oni_rows = con.sql(
            "SELECT CAST(oni_index AS DOUBLE) FROM feature_store_daily WHERE oni_index IS NOT NULL ORDER BY event_date DESC LIMIT 1"
        ).fetchall()
        oni_val = float(oni_rows[0][0]) if oni_rows else 0.0

        latest_date = con.sql(
            "SELECT MAX(CAST(forecast_issued_at AS DATE)) FROM prediction_logs"
        ).fetchall()[0][0]

        max_probs = con.execute(
            """
            SELECT station_id, MAX(flood_prob) AS max_flood_prob, MAX(pred_p90) AS peak_p90_m
            FROM prediction_logs
            WHERE CAST(forecast_issued_at AS DATE) = ?
              AND model_alias = '@champion'
            GROUP BY station_id
            ORDER BY station_id
            """,
            [latest_date],
        ).fetchall()
    finally:
        con.close()

    regime = classify_enso_regime(oni_val)
    lines = [
        "# HELP enso_oni_index Current Oceanic Nino Index (Celsius)",
        "# TYPE enso_oni_index gauge",
        f'enso_oni_index{{regime="{regime}"}} {oni_val:.2f}',
        "# HELP station_max_flood_prob_15d Maximum 15-day flood exceedance probability",
        "# TYPE station_max_flood_prob_15d gauge",
    ]
    for st_id, max_prob, peak_p90 in max_probs:
        lines.append(f'station_max_flood_prob_15d{{station_id="{st_id}"}} {float(max_prob):.4f}')

    lines.append("# HELP station_peak_p90_meters Peak 15-day P90 river level forecast in meters")
    lines.append("# TYPE station_peak_p90_meters gauge")
    for st_id, _, peak_p90 in max_probs:
        lines.append(f'station_peak_p90_meters{{station_id="{st_id}"}} {float(peak_p90):.4f}')

    return "\n".join(lines) + "\n"
