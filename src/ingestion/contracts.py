from datetime import date, datetime
from pathlib import Path
import polars as pl
import pandera.polars as pa
from pandera.errors import SchemaErrors
from loguru import logger

QUARANTINE_DIR = Path("data/quarantine")


class RawObservationSchema(pa.DataFrameModel):
    event_date: pl.Date = pa.Field(nullable=False)
    station_id: pl.String = pa.Field(str_matches=r"^AR_[A-Z]{3}_\d{3}$", nullable=False)
    precip_mm: pl.Float64 = pa.Field(ge=0.0, le=500.0, nullable=True)
    river_level_m: pl.Float64 = pa.Field(ge=-2.0, le=25.0, nullable=True)
    oni_index: pl.Float64 = pa.Field(ge=-3.5, le=3.5, nullable=True)


class ForecastInputSchema(pa.DataFrameModel):
    issued_date: pl.Date = pa.Field(nullable=False)
    target_date: pl.Date = pa.Field(nullable=False)
    station_id: pl.String = pa.Field(str_matches=r"^AR_[A-Z]{3}_\d{3}$", nullable=False)
    forecast_precip_mm: pl.Float64 = pa.Field(ge=0.0, le=500.0, nullable=False)

    @pa.dataframe_check
    def target_after_issued(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col("target_date") >= pl.col("issued_date"))


def validate_and_quarantine(
    df: pl.DataFrame,
    schema: type[pa.DataFrameModel],
    source_name: str,
) -> pl.DataFrame:
    """Valida el DataFrame con Pandera; guarda filas inválidas en cuarentena y retorna las limpias."""
    try:
        validated = schema.validate(df, lazy=True)
        logger.info(f"[{source_name}] Validación exitosa: {len(validated)} filas válidas.")
        return validated
    except SchemaErrors as err:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_file = QUARANTINE_DIR / f"{source_name}_errors_{ts}.parquet"

        failure_cases = err.failure_cases
        failure_cases.write_parquet(quarantine_file)
        logger.warning(
            f"[{source_name}] Se detectaron anomalías. Detalle guardado en {quarantine_file}"
        )

        # Filtrado de filas válidas mediante los índices reportados por Pandera
        bad_indices = failure_cases["index"].drop_nulls().unique().to_list()
        clean_df = (
            df.with_row_index("_row_idx")
            .filter(~pl.col("_row_idx").is_in(bad_indices))
            .drop("_row_idx")
        )
        return schema.validate(clean_df)