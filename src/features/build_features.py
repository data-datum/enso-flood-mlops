from pathlib import Path
import duckdb
import numpy as np
import polars as pl
from src.ingestion.contracts import feature_store_schema
from datetime import date

DB_PATH = Path("data/flood_monitor.duckdb")


def build_antecedent_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Calcula variables de estado de cuenca hasta hoy (t) sin data leakage."""
    raw_df = pl.from_arrow(
        con.execute(
            """
            SELECT 
                CAST(event_date AS DATE) AS event_date,
                station_id,
                CAST(precip_mm AS DOUBLE) AS precip_mm,
                CAST(river_level_m AS DOUBLE) AS river_level_m,
                CAST(oni_index AS DOUBLE) AS oni_index
            FROM raw_observations
            ORDER BY station_id, event_date
            """
        ).to_arrow_table()
    )

    if raw_df.is_empty():
        raise ValueError("La tabla raw_observations está vacía.")

    # Rellenar fechas faltantes por estación para que rolling_sum(window_size="7d") sea exacto
    df = (
        raw_df.sort(["station_id", "event_date"])
        .upsample(time_column="event_date", every="1d", group_by="station_id")
        .with_columns(
            pl.col("precip_mm").fill_null(0.0),
            pl.col("oni_index").forward_fill().over("station_id"),
            pl.col("river_level_m").forward_fill(limit=2).over("station_id"),
        )
    )

# 1. Acumulados de lluvia (proxy de saturación del suelo, evitando ruido flotante negativo)
    df = df.with_columns(
        pl.col("precip_mm")
        .rolling_sum(window_size=3, min_samples=1)
        .clip(lower_bound=0.0)
        .round(4)
        .over("station_id")
        .alias("precip_acc_3d"),
        pl.col("precip_mm")
        .rolling_sum(window_size=7, min_samples=1)
        .clip(lower_bound=0.0)
        .round(4)
        .over("station_id")
        .alias("precip_acc_7d"),
        pl.col("precip_mm")
        .rolling_sum(window_size=15, min_samples=1)
        .clip(lower_bound=0.0)
        .round(4)
        .over("station_id")
        .alias("precip_acc_15d"),
        pl.col("precip_mm")
        .rolling_sum(window_size=30, min_samples=1)
        .clip(lower_bound=0.0)
        .round(4)
        .over("station_id")
        .alias("precip_acc_30d"),
    )

# 2. Dinámica del nivel del río (tasas de crecida/bajante y media móvil)
    df = df.with_columns(
        (pl.col("river_level_m") - pl.col("river_level_m").shift(1))
        .over("station_id")
        .alias("river_diff_24h"),
        (pl.col("river_level_m") - pl.col("river_level_m").shift(2))
        .over("station_id")
        .alias("river_diff_48h"),
        (pl.col("river_level_m") - pl.col("river_level_m").shift(7))
        .over("station_id")
        .alias("river_diff_7d"),
        pl.col("river_level_m")
        .rolling_mean(window_size=7, min_samples=1)
        .over("station_id")
        .alias("river_roll_mean_7d"),
    )

    # 3. Señal ENSO (tendencia trimestral y bandera de régimen El Niño >= +0.5 °C)
    df = df.with_columns(
        (pl.col("oni_index") - pl.col("oni_index").shift(90))
        .over("station_id")
        .alias("oni_trend_90d"),
        (pl.col("oni_index").fill_null(0.0) >= 0.5).cast(pl.Int8).alias("is_el_nino"),
    )

    # Validar con Pandera antes de persistir
    validated_df = feature_store_schema.validate(df)
    return validated_df

def build_inference_matrix(
    con: duckdb.DuckDBPyConnection, features_df: pl.DataFrame, issued_date: str | None = None
) -> pl.DataFrame:
    """Construye la matriz (station_id x horizon 1..15) para inferencia diaria."""

    if issued_date is None:
        issued_date_val = con.sql("SELECT MAX(issued_date) FROM forecast_inputs").fetchall()[0][0]
    elif isinstance(issued_date, str):
        issued_date_val = date.fromisoformat(issued_date)
    else:
        issued_date_val = issued_date


    fc_df = pl.from_arrow(
        con.execute(
            """
            SELECT 
                CAST(issued_date AS DATE) AS issued_date,
                CAST(target_date AS DATE) AS target_date,
                station_id,
                CAST(forecast_precip_mm AS DOUBLE) AS forecast_precip_mm
            FROM forecast_inputs
            WHERE issued_date = ?
            ORDER BY station_id, target_date
            """,
            [issued_date_val],
        ).to_arrow_table()
    )

    # Calcular horizonte h = (target_date - issued_date) en días y lluvia futura acumulada hasta t+h
    fc_horizons = (
        fc_df.with_columns(
            (pl.col("target_date") - pl.col("issued_date")).dt.total_days().cast(pl.Int32).alias("horizon_day")
        )
        .filter((pl.col("horizon_day") >= 1) & (pl.col("horizon_day") <= 15))
        .sort(["station_id", "horizon_day"])
        .with_columns(
            pl.col("forecast_precip_mm").cum_sum().over("station_id").alias("future_precip_acc_h"),
            pl.col("forecast_precip_mm").alias("future_precip_day_h"),
        )
    )

    # Unir el estado de la cuenca en issued_date (t) con los 15 horizontes futuros
    state_t = features_df.filter(pl.col("event_date") == pl.lit(issued_date_val))
    inference_df = fc_horizons.join(state_t, on="station_id", how="inner")

    return inference_df


def build_training_dataset(
    features_df: pl.DataFrame, max_horizon: int = 15, seed: int = 42
) -> pl.DataFrame:
    """
    Genera dataset supervisado histórico (Direct Multi-Step) simulando incertidumbre 
    de pronóstico meteorológico creciente con el horizonte h.
    """
    rng = np.random.default_rng(seed)
    base = features_df.sort(["station_id", "event_date"]).filter(pl.col("river_level_m").is_not_null())

    horizon_frames = []
    for h in range(1, max_horizon + 1):
        # Target real en t+h y suma exacta de lluvia real entre t+1 y t+h
        future_rain_expr = sum(
            pl.col("precip_mm").shift(-step).over("station_id") for step in range(1, h + 1)
        )
        day_h_rain_expr = pl.col("precip_mm").shift(-h).over("station_id")
        target_level_expr = pl.col("river_level_m").shift(-h).over("station_id")

        df_h = base.with_columns(
            pl.lit(h).cast(pl.Int32).alias("horizon_day"),
            future_rain_expr.alias("true_future_precip_acc"),
            day_h_rain_expr.alias("true_future_precip_day"),
            target_level_expr.alias("target_river_level_m"),
            (pl.col("event_date") + pl.duration(days=h)).alias("target_date"),
        ).filter(pl.col("target_river_level_m").is_not_null())

        # Inyección de ruido calibrado: sigma crece ~3% por día de horizonte (ej. 45% en día 15)
        n = len(df_h)
        sigma = 0.03 * h
        noise_acc = np.clip(rng.normal(loc=1.0, scale=sigma, size=n), 0.2, 2.5)
        noise_day = np.clip(rng.normal(loc=1.0, scale=sigma * 1.2, size=n), 0.0, 3.0)

        df_h = df_h.with_columns(
            (pl.col("true_future_precip_acc") * pl.Series(noise_acc)).alias("future_precip_acc_h"),
            (pl.col("true_future_precip_day") * pl.Series(noise_day)).alias("future_precip_day_h"),
        ).drop(["true_future_precip_acc", "true_future_precip_day"])

        horizon_frames.append(df_h)

    train_df = pl.concat(horizon_frames).sort(["station_id", "event_date", "horizon_day"])
    return train_df


def run_feature_pipeline() -> None:
    con = duckdb.connect(str(DB_PATH))
    try:
        features_df = build_antecedent_features(con)

        # Persistir feature_store_daily en DuckDB y respaldo Parquet
        con.register("features_view", features_df.to_arrow())
        con.execute("CREATE OR REPLACE TABLE feature_store_daily AS SELECT * FROM features_view")

        train_df = build_training_dataset(features_df, max_horizon=15)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        train_df.write_parquet("data/processed/train_multihorizon.parquet")

        print(f"feature_store_daily guardada: {len(features_df):,} filas.")
        print(f"Dataset de entrenamiento generado: {len(train_df):,} filas (horizontes 1..15).")
    finally:
        con.close()


if __name__ == "__main__":
    run_feature_pipeline()