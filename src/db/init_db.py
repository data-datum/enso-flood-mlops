from pathlib import Path
import duckdb
from loguru import logger

DB_PATH = Path("data/flood_monitor.duckdb")

STATIONS_SEED = [
    # station_id, ina_station_id, name, river, lat, lon, alert_threshold_m, evacuation_threshold_m
    ("AR_COR_001", 29, "Puerto Corrientes", "Paraná", -27.4667, -58.8333, 6.50, 7.00),
    ("AR_COR_002", 28, "Paso de la Patria", "Paraná", -27.3167, -58.5667, 6.50, 7.00),
    ("AR_COR_003", 33, "Goya", "Paraná", -29.1333, -59.2667, 5.20, 5.70),
    ("AR_SFE_001", 34, "Reconquista", "Paraná", -29.1500, -59.6500, 5.10, 5.30),
]


def init_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    logger.info("Creando tablas core en DuckDB...")

    con.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            station_id VARCHAR PRIMARY KEY,
            ina_station_id INTEGER NOT NULL,
            name VARCHAR NOT NULL,
            river VARCHAR NOT NULL,
            lat DOUBLE NOT NULL,
            lon DOUBLE NOT NULL,
            alert_threshold_m DOUBLE NOT NULL,
            evacuation_threshold_m DOUBLE NOT NULL
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_observations (
            event_date DATE NOT NULL,
            station_id VARCHAR NOT NULL,
            precip_mm DOUBLE,
            river_level_m DOUBLE,
            oni_index DOUBLE,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (event_date, station_id),
            FOREIGN KEY (station_id) REFERENCES stations(station_id)
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_inputs (
            issued_date DATE NOT NULL,
            target_date DATE NOT NULL,
            station_id VARCHAR NOT NULL,
            forecast_precip_mm DOUBLE NOT NULL,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (issued_date, target_date, station_id),
            FOREIGN KEY (station_id) REFERENCES stations(station_id)
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            forecast_issued_at TIMESTAMP NOT NULL,
            target_date DATE NOT NULL,
            station_id VARCHAR NOT NULL,
            horizon_day INTEGER NOT NULL,
            pred_p10 DOUBLE NOT NULL,
            pred_p50 DOUBLE NOT NULL,
            pred_p90 DOUBLE NOT NULL,
            flood_prob DOUBLE NOT NULL,
            model_version VARCHAR NOT NULL,
            model_alias VARCHAR NOT NULL,
            PRIMARY KEY (forecast_issued_at, target_date, station_id, model_alias)
        );
    """)

    con.executemany("""
        INSERT OR IGNORE INTO stations VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, STATIONS_SEED)

    con.close()
    logger.success(f"Base de datos inicializada correctamente en {db_path}")


if __name__ == "__main__":
    init_database()