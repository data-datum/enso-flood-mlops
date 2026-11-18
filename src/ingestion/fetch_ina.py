from datetime import date, timedelta
import httpx
import polars as pl
from loguru import logger

INA_BASE_URL = "https://alerta.ina.gob.ar/a5"
RIVER_LEVEL_VAR_ID = 2  # Altura / Nivel hidrométrico
HEADERS = {"User-Agent": "enso-flood-mlops/0.1 (research-monitoring)"}


def _get_ina_series_id(client: httpx.Client, ina_station_id: int) -> int | None:
    """Busca el series_id activo de altura (var_id=2) para una estación del INA."""
    url = f"{INA_BASE_URL}/obs/puntual/series"
    params = {"estacion_id": ina_station_id, "var_id": RIVER_LEVEL_VAR_ID, "format": "json"}
    try:
        resp = client.get(url, params=params, timeout=20.0)
        resp.raise_for_status()
        payload = resp.json()
        series_list = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if series_list and isinstance(series_list, list):
            return int(series_list[0]["id"])
    except Exception as exc:
        logger.warning(f"No se pudo resolver series_id INA para estacion_id={ina_station_id}: {exc}")
    return None


def fetch_ina_river_level(
    ina_station_id: int,
    station_id: str,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Descarga alturas observadas desde la API del INA en bloques de 90 días y agrega a escala diaria."""
    empty_df = pl.DataFrame(
        schema={"event_date": pl.Date, "station_id": pl.String, "river_level_m": pl.Float64}
    )

    with httpx.Client(headers=HEADERS) as client:
        series_id = _get_ina_series_id(client, ina_station_id)
        if series_id is None:
            logger.warning(f"Sin series_id en INA para {station_id} (estacion_id={ina_station_id})")
            return empty_df

        rows = []
        chunk_start = start_date
        chunk_days = 90  # Ventana segura para evitar 400 Bad Request en INA

        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end_date)
            params = {
                "series_id": series_id,
                "timestart": f"{chunk_start.isoformat()}T00:00:00.000Z",
                "timeend": f"{chunk_end.isoformat()}T23:59:59.000Z",
                "format": "json",
            }
            try:
                resp = client.get(f"{INA_BASE_URL}/obs/puntual/observaciones", params=params, timeout=30.0)
                resp.raise_for_status()
                payload = resp.json()
                obs_list = payload.get("rows", payload) if isinstance(payload, dict) else payload

                for item in obs_list:
                    ts_str = item.get("timestart")
                    val = item.get("valor")
                    if ts_str is not None and val is not None:
                        rows.append({
                            "event_date": date.fromisoformat(ts_str[:10]),
                            "station_id": station_id,
                            "valor": float(val),
                        })
            except Exception as exc:
                logger.warning(
                    f"Error en tramo {chunk_start}..{chunk_end} INA ({station_id}): {exc}"
                )

            chunk_start = chunk_end + timedelta(days=1)

    if not rows:
        logger.warning(f"Sin registros hidrométricos devueltos por INA para {station_id}")
        return empty_df

    df_daily = (
        pl.DataFrame(rows)
        .group_by(["event_date", "station_id"])
        .agg(pl.col("valor").max().round(3).alias("river_level_m"))
        .sort("event_date")
    )
    return df_daily