from datetime import date, timedelta
import duckdb
import httpx
import polars as pl
from loguru import logger

from src.db.init_db import DB_PATH
from src.ingestion.contracts import RawObservationSchema, validate_and_quarantine
from src.ingestion.fetch_enso import fetch_latest_oni
from src.ingestion.fetch_ina import fetch_ina_river_level

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical_precip(lat: float, lon: float, start_date: date, end_date: date) -> pl.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "America/Argentina/Buenos_Aires",
    }
    resp = httpx.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60.0)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    return pl.DataFrame({
        "event_date": [date.fromisoformat(d) for d in daily["time"]],
        "precip_mm": [float(p) if p is not None else 0.0 for p in daily["precipitation_sum"]],
    })


def run_historical_backfill(years_back: int = 3, db_path=DB_PATH) -> None:
    """Descarga el historial diario de lluvias, niveles INA e índice ONI para entrenamiento."""
    con = duckdb.connect(str(db_path))
    stations = con.execute(
        "SELECT station_id, ina_station_id, lat, lon FROM stations"
    ).fetchall()

    # Open-Meteo Archive tiene un desfase de ~5 días respecto a hoy
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=365 * years_back)

    logger.info(f"Iniciando backfill histórico desde {start_date} hasta {end_date}...")

    # 1. Histórico mensual ONI de NOAA
    df_oni = fetch_latest_oni()

    all_stations_dfs = []

    for station_id, ina_station_id, lat, lon in stations:
        logger.info(f"Procesando historial de estación {station_id}...")

        # Lluvias históricas ERA5
        df_precip = fetch_historical_precip(lat, lon, start_date, end_date).with_columns(
            pl.lit(station_id).alias("station_id")
        )

        # Niveles históricos del río (INA)
        df_river = fetch_ina_river_level(
            ina_station_id=ina_station_id,
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
        )

        # Cruzar lluvia + río + índice ONI (por año y mes)
        df_merged = (
            df_precip
            .join(df_river, on=["event_date", "station_id"], how="left")
            .with_columns([
                pl.col("event_date").dt.year().alias("year"),
                pl.col("event_date").dt.month().alias("month"),
            ])
            .join(df_oni, on=["year", "month"], how="left")
            .with_columns(pl.col("oni_index").forward_fill())
            .select(["event_date", "station_id", "precip_mm", "river_level_m", "oni_index"])
        )

        all_stations_dfs.append(df_merged)

    full_history = pl.concat(all_stations_dfs)
    clean_history = validate_and_quarantine(
        full_history, RawObservationSchema, "historical_backfill"
    )

    con.register("tmp_hist", clean_history.to_arrow())
    con.execute("""
        INSERT OR REPLACE INTO raw_observations (event_date, station_id, precip_mm, river_level_m, oni_index)
        SELECT event_date, station_id, precip_mm, river_level_m, oni_index FROM tmp_hist;
    """)

    con.close()
    logger.success(f"Backfill completado: {len(clean_history)} registros históricos insertados en DuckDB.")


if __name__ == "__main__":
    run_historical_backfill(years_back=3)