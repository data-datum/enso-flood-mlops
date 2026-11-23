from datetime import date, timedelta
from pathlib import Path
import duckdb
import pandas as pd
from src.ingestion.fetch_ina import fetch_ina_river_level

DB_PATH = Path("data/flood_monitor.duckdb")


def run_daily_ingestion(days_back: int = 3) -> None:
    """Ingesta observaciones recientes del INA y actualiza forecast_inputs para hoy."""
    today = date.today()
    start_date = today - timedelta(days=days_back)

    con = duckdb.connect(str(DB_PATH))
    try:
        stations_df = con.sql("SELECT * FROM stations").df()
        raw_cols = con.sql("SELECT * FROM raw_observations LIMIT 0").df().columns.tolist()

        # 1. Actualizar observaciones hidrométricas recientes desde el INA
        for _, row in stations_df.iterrows():
            st_id = row["station_id"]
            ina_id = row.get("ina_station_id")
            if pd.notna(ina_id):
                try:
                    obs_df = fetch_ina_river_level(
                        ina_station_id=int(ina_id),
                        start_date=start_date,
                        end_date=today,
                    )
                    if obs_df is not None and not obs_df.empty:
                        obs_df["station_id"] = st_id
                        con.execute(
                            "DELETE FROM raw_observations WHERE station_id = ? AND CAST(event_date AS DATE) >= ?",
                            [st_id, start_date],
                        )
                        con.register("temp_obs", obs_df)
                        con.execute("INSERT INTO raw_observations BY NAME SELECT * FROM temp_obs")
                except Exception as exc:
                    print(f"[Ingesta INA] Aviso para {st_id} (ina_id={ina_id}): {exc}")

        # 2. Fallback: si hoy aún no hay registro en raw_observations, propagar último estado conocido
        for st_id in stations_df["station_id"]:
            has_today = con.execute(
                "SELECT COUNT(*) FROM raw_observations WHERE station_id = ? AND CAST(event_date AS DATE) = ?",
                [st_id, today],
            ).fetchall()[0][0]

            if has_today == 0:
                last_row = con.execute(
                    """
                    SELECT *
                    FROM raw_observations
                    WHERE station_id = ?
                    ORDER BY event_date DESC
                    LIMIT 1
                    """,
                    [st_id],
                ).df()

                if not last_row.empty:
                    last_row["event_date"] = today
                    if "precip_mm" in raw_cols:
                        last_row["precip_mm"] = 0.0
                    con.register("temp_fallback", last_row)
                    con.execute("INSERT INTO raw_observations BY NAME SELECT * FROM temp_fallback")

        # 3. Asegurar filas de pronóstico meteorológico 15d para issued_date = hoy
        existing_fc = con.execute(
            "SELECT COUNT(*) FROM forecast_inputs WHERE CAST(issued_date AS DATE) = ?",
            [today],
        ).fetchall()[0][0]

        if existing_fc == 0:
            last_issued = con.sql("SELECT MAX(issued_date) FROM forecast_inputs").fetchall()[0][0]
            if last_issued is not None:
                template_fc = con.execute(
                    """
                    SELECT
                        CAST(? AS DATE) AS issued_date,
                        CAST(? AS DATE) + CAST(DATE_DIFF('day', CAST(issued_date AS DATE), CAST(target_date AS DATE)) AS INTEGER) AS target_date,
                        station_id,
                        forecast_precip_mm
                    FROM forecast_inputs
                    WHERE issued_date = ?
                    """,
                    [today, today, last_issued],
                ).df()

                if not template_fc.empty:
                    con.register("temp_fc", template_fc)
                    con.execute("INSERT INTO forecast_inputs BY NAME SELECT * FROM temp_fc")
                    print(f"[Ingesta Forecast] Generadas {len(template_fc)} filas de pronóstico para {today}.")
    finally:
        con.close()

    print(f"Ingesta diaria completada para {today}.")


if __name__ == "__main__":
    run_daily_ingestion()