import httpx
import polars as pl
from loguru import logger

NOAA_ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"


def fetch_latest_oni() -> pl.DataFrame:
    """Descarga la tabla histórica mensual del índice ONI desde NOAA PSL."""
    logger.info("Descargando índice ONI desde NOAA...")
    response = httpx.get(NOAA_ONI_URL, timeout=30.0)
    response.raise_for_status()

    lines = response.text.strip().splitlines()
    records = []

    for line in lines[1:]:
        parts = line.split()
        # Filtrar filas de años válidos (ej. 1950 a 2026)
        if len(parts) == 13 and parts[0].isdigit() and len(parts[0]) == 4:
            year = int(parts[0])
            for month_idx, val_str in enumerate(parts[1:], start=1):
                val = float(val_str)
                if val > -90.0:  # Ignorar sentinel -99.90 de NOAA
                    records.append({"year": year, "month": month_idx, "oni_index": val})

    df = pl.DataFrame(records).sort(["year", "month"])
    logger.info(f"Último ONI registrado: {df.tail(1).to_dicts()[0]}")
    return df


def get_current_oni_value() -> float:
    df = fetch_latest_oni()
    return float(df["oni_index"][-1])