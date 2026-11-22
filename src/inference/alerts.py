import os
from pathlib import Path
import urllib.parse
import urllib.request
import duckdb
import pandas as pd

DB_PATH = Path("data/flood_monitor.duckdb")


def classify_enso_regime(oni_index: float) -> str:
    if oni_index >= 1.5:
        return "EL_NINO_STRONG"
    if oni_index >= 0.5:
        return "EL_NINO_MODERATE"
    if oni_index <= -0.5:
        return "LA_NINA"
    return "NEUTRAL"


def send_telegram_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status == 200


def evaluate_alerts(issued_date: str | None = None) -> list[str]:
    con = duckdb.connect(str(DB_PATH))
    try:
        if issued_date is None:
            rows = con.sql(
                "SELECT MAX(CAST(forecast_issued_at AS DATE)) FROM prediction_logs"
            ).fetchall()
            issued_date = str(rows[0][0])

        # 1. Estado macro ENSO más reciente
        oni_rows = con.sql(
            """
            SELECT event_date, CAST(oni_index AS DOUBLE) AS oni_index
            FROM feature_store_daily
            WHERE oni_index IS NOT NULL
            ORDER BY event_date DESC
            LIMIT 1
            """
        ).fetchall()

        oni_val = float(oni_rows[0][1]) if oni_rows else 0.0
        enso_state = classify_enso_regime(oni_val)
        is_el_nino = oni_val >= 0.5

    # 2. Cruzar pronóstico @champion con cotas de alerta/evacuación de cada estación
        stations_df = con.sql("SELECT * FROM stations").df()
        label_col = next(
            (c for c in ["station_name", "name", "location", "river_name"] if c in stations_df.columns),
            "station_id",
        )

        df = con.execute(
            f"""
            SELECT
                p.station_id,
                s.{label_col} AS station_label,
                p.horizon_day,
                p.target_date,
                p.pred_p10,
                p.pred_p50,
                p.pred_p90,
                p.flood_prob,
                CAST(s.alert_threshold_m AS DOUBLE) AS alert_threshold_m,
                CAST(s.evacuation_threshold_m AS DOUBLE) AS evacuation_threshold_m
            FROM prediction_logs p
            INNER JOIN stations s ON p.station_id = s.station_id
            WHERE CAST(p.forecast_issued_at AS DATE) = ?
              AND p.model_alias = '@champion'
            ORDER BY p.station_id, p.horizon_day
            """,
            [issued_date],
        ).df()
    finally:
        con.close()

    alerts: list[str] = []
    header = (
        f"*REPORTE HIDROMETEOROLÓGICO 15D ({issued_date})*\n"
        f"• Régimen ENSO: *{enso_state}* (ONI = {oni_val:+.2f} °C)\n"
        f"• Criterio activo: {'Preventivo P90 (El Niño)' if is_el_nino else 'Estándar P50 (Neutral)'}"
    )
    alerts.append(header)

    # En régimen El Niño se vigila P90; en Neutral se vigila P50[cite: 1]
    eval_col = "pred_p90" if is_el_nino else "pred_p50"

    for station_id, grp in df.groupby("station_id"):
        st_name = grp["station_label"].iloc[0]
        alert_th = grp["alert_threshold_m"].iloc[0]
        evac_th = grp["evacuation_threshold_m"].iloc[0]

        evac_hits = grp[grp[eval_col] >= evac_th]
        alert_hits = grp[grp[eval_col] >= alert_th]

        if not evac_hits.empty:
            first = evac_hits.iloc[0]
            alerts.append(
                f"🚨 *[EVACUACIÓN] {station_id} ({st_name})*: {eval_col}={first[eval_col]:.2f} m "
                f"(cota={evac_th:.2f} m) proyectado para *{first['target_date']}* (h=+{first['horizon_day']}d, "
                f"prob={first['flood_prob']*100:.1f}%)."
            )
        elif not alert_hits.empty:
            first = alert_hits.iloc[0]
            alerts.append(
                f"⚠️ *[ALERTA] {station_id} ({st_name})*: {eval_col}={first[eval_col]:.2f} m "
                f"(cota={alert_th:.2f} m) proyectado para *{first['target_date']}* (h=+{first['horizon_day']}d, "
                f"prob={first['flood_prob']*100:.1f}%)."
            )
        else:
            peak = grp.loc[grp[eval_col].idxmax()]
            alerts.append(
                f"✅ *[NORMAL] {station_id} ({st_name})*: pico proyectado {eval_col}={peak[eval_col]:.2f} m "
                f"(cota alerta={alert_th:.2f} m) el {peak['target_date']} (h=+{peak['horizon_day']}d)."
            )

    full_report = "\n".join(alerts)
    print("\n" + full_report + "\n")

    if send_telegram_message(full_report):
        print("Notificación enviada por Telegram correctamente.")

    return alerts


if __name__ == "__main__":
    evaluate_alerts()