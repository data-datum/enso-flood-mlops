from datetime import date, timedelta
import duckdb
from datetime import date, timedelta
import duckdb
import httpx
import polars as pl
from loguru import logger

from src.db.init_db import DB_PATH
from src.ingestion.contracts import (
    ForecastInputSchema,
    RawObservationSchema,
    validate_and_quarantine,
)
from src.ingestion.fetch_enso import get_current_oni_value
from src.ingestion.fetch_ina import fetch_ina_river_level

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_station_weather(lat: float, lon: float, past_days: int = 3) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "past_days": past_days,
        "forecast_days": 16,  # Hoy (t) + 15 días futuros (t+1..t+15)
        "timezone": "America/Argentina/Buenos_Aires",
    }
    resp = httpx.get(OPEN_METEO_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["daily"]


def run_daily_ingestion(db_path=DB_PATH, past_days: int = 3) -> None:
    con = duckdb.connect(str(db_path))
    stations = con.execute(
        "SELECT station_id, ina_station_id, lat, lon FROM stations"
    ).fetchall()
    current_oni = get_current_oni_value()
    today = date.today()
    start_obs = today - timedelta(days=past_days)

    obs_dfs = []
    fcst_rows = []

    for station_id, ina_station_id, lat, lon in stations:
        # 1. Descargar lluvias y pronóstico
        daily = fetch_station_weather(lat, lon, past_days=past_days)
        dates = [date.fromisoformat(d) for d in daily["time"]]
        precips = daily["precipitation_sum"]

        weather_obs = []
        for d, p in zip(dates, precips):
            p_val = float(p) if p is not None else 0.0
            if d <= today:
                weather_obs.append({
                    "event_date": d,
                    "station_id": station_id,
                    "precip_mm": p_val,
                    "oni_index": current_oni,
                })
            else:
                fcst_rows.append({
                    "issued_date": today,
                    "target_date": d,
                    "station_id": station_id,
                    "forecast_precip_mm": p_val,
                })

        df_weather = pl.DataFrame(weather_obs)

        # 2. Descargar nivel del río desde el INA
        df_river = fetch_ina_river_level(
            ina_station_id=ina_station_id,
            station_id=station_id,
            start_date=start_obs,
            end_date=today,
        )

        # 3. Unir lluvia + nivel del río por fecha y estación
        df_joined = df_weather.join(
            df_river, on=["event_date", "station_id"], how="left"
        ).select(["event_date", "station_id", "precip_mm", "river_level_m", "oni_index"])

        obs_dfs.append(df_joined)

    # Validación con Pandera
    obs_df = validate_and_quarantine(
        pl.concat(obs_dfs), RawObservationSchema, "raw_observations"
    )
    fcst_df = validate_and_quarantine(
        pl.DataFrame(fcst_rows), ForecastInputSchema, "forecast_inputs"
    )

    # Inserción idempotente en DuckDB
    con.register("tmp_obs", obs_df.to_arrow())
    con.execute("""
        INSERT OR REPLACE INTO raw_observations (event_date, station_id, precip_mm, river_level_m, oni_index)
        SELECT event_date, station_id, precip_mm, river_level_m, oni_index FROM tmp_obs;
    """)

    con.register("tmp_fcst", fcst_df.to_arrow())
    con.execute("""
        INSERT OR REPLACE INTO forecast_inputs (issued_date, target_date, station_id, forecast_precip_mm)
        SELECT issued_date, target_date, station_id, forecast_precip_mm FROM tmp_fcst;
    """)

    con.close()
    logger.success(
        f"Ingesta diaria completada: {len(obs_df)} observaciones y {len(fcst_df)} pronósticos guardados."
    )


if __name__ == "__main__":
    run_daily_ingestion()