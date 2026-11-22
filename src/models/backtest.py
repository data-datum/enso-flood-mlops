from datetime import timedelta
from pathlib import Path
import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_pinball_loss
from src.models.features_config import CATEGORICAL_COLS, FEATURE_COLS, QUANTILES, TARGET_COL

DB_PATH = Path("data/flood_monitor.duckdb")
TRAIN_PATH = Path("data/processed/train_multihorizon.parquet")


def prepare_pandas_xy(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    pdf = df.select(FEATURE_COLS + [TARGET_COL]).to_pandas()
    for col in CATEGORICAL_COLS:
        pdf[col] = pdf[col].astype("category")
    X = pdf[FEATURE_COLS]
    y = pdf[TARGET_COL]
    return X, y


def train_quantile_ensemble(X_train: pd.DataFrame, y_train: pd.Series) -> dict[float, lgb.LGBMRegressor]:
    models: dict[float, lgb.LGBMRegressor] = {}
    for alpha in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            min_child_samples=30,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)
        models[alpha] = model
    return models


def compute_flood_probability(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Estima P(nivel > cota_alerta) aproximando la desviación estándar desde el rango intercuantílico P10-P90."""
    # En una normal, P90 - P10 ≈ 2.563 * sigma
    sigma = np.maximum((p90 - p10) / 2.563, 0.05)
    z = (threshold - p50) / sigma
    # Aproximación logística rápida de 1 - Phi(z)
    prob = 1.0 / (1.0 + np.exp(1.702 * z))
    return np.clip(prob, 0.0, 1.0)


def run_expanding_backtest(n_splits: int = 3, test_days: int = 60) -> dict[str, float]:
    train_df = pl.read_parquet(TRAIN_PATH)

    # Traer cotas de alerta desde DuckDB para evaluar PR-AUC de desborde
    con = duckdb.connect(str(DB_PATH))
    stations_df = pl.from_arrow(
        con.sql("SELECT station_id, CAST(alert_threshold_m AS DOUBLE) AS alert_threshold_m FROM stations").to_arrow_table()
    )
    con.close()

    train_df = train_df.join(stations_df, on="station_id", how="left")

    max_date = train_df["event_date"].max()
    min_date = train_df["event_date"].min()
    print(f"Rango histórico para backtesting: {min_date} -> {max_date}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("enso-flood-15d-backtest")

    all_preds = []

    with mlflow.start_run(run_name="expanding_window_cv"):
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("test_days", test_days)
        mlflow.log_param("quantiles", str(QUANTILES))

        for fold in range(n_splits, 0, -1):
            test_end = max_date - timedelta(days=(fold - 1) * test_days)
            test_start = test_end - timedelta(days=test_days)
            # Purga de 15 días entre train y test para evitar solapamiento de targets futuros
            train_cutoff = test_start - timedelta(days=15)

            df_tr = train_df.filter(pl.col("event_date") <= train_cutoff)
            df_va = train_df.filter((pl.col("event_date") > test_start) & (pl.col("event_date") <= test_end))

            if df_tr.is_empty() or df_va.is_empty():
                continue

            X_tr, y_tr = prepare_pandas_xy(df_tr)
            X_va, y_va = prepare_pandas_xy(df_va)

            models = train_quantile_ensemble(X_tr, y_tr)

            p10 = models[0.10].predict(X_va)
            p50 = models[0.50].predict(X_va)
            p90 = models[0.90].predict(X_va)

            # Garantizar monotonicidad de cuantiles (P10 <= P50 <= P90)
            p10_clean = np.minimum(p10, p50)
            p90_clean = np.maximum(p90, p50)

            thresholds = df_va["alert_threshold_m"].to_numpy()
            flood_prob = compute_flood_probability(p10_clean, p50, p90_clean, thresholds)

            fold_res = pl.DataFrame(
                {
                    "fold": n_splits - fold + 1,
                    "horizon_day": df_va["horizon_day"],
                    "y_true": y_va.to_numpy(),
                    "p10": p10_clean,
                    "p50": p50,
                    "p90": p90_clean,
                    "flood_prob": flood_prob,
                    "is_flood_true": (y_va.to_numpy() >= thresholds).astype(int),
                }
            )
            all_preds.append(fold_res)

        eval_df = pl.concat(all_preds)

        # Métricas globales y segmentadas por horizonte
        segments = {
            "global": eval_df,
            "short_1_3d": eval_df.filter(pl.col("horizon_day").is_between(1, 3)),
            "medium_4_7d": eval_df.filter(pl.col("horizon_day").is_between(4, 7)),
            "long_8_15d": eval_df.filter(pl.col("horizon_day").is_between(8, 15)),
        }

        summary_metrics: dict[str, float] = {}
        for seg_name, seg_df in segments.items():
            yt = seg_df["y_true"].to_numpy()
            yp10 = seg_df["p10"].to_numpy()
            yp50 = seg_df["p50"].to_numpy()
            yp90 = seg_df["p90"].to_numpy()
            y_bin = seg_df["is_flood_true"].to_numpy()
            y_prob = seg_df["flood_prob"].to_numpy()

            mae = float(mean_absolute_error(yt, yp50))
            pb10 = float(mean_pinball_loss(yt, yp10, alpha=0.10))
            pb50 = float(mean_pinball_loss(yt, yp50, alpha=0.50))
            pb90 = float(mean_pinball_loss(yt, yp90, alpha=0.90))
            pinball_avg = (pb10 + pb50 + pb90) / 3.0

            summary_metrics[f"{seg_name}_mae"] = mae
            summary_metrics[f"{seg_name}_pinball_avg"] = pinball_avg

            mlflow.log_metric(f"{seg_name}_mae", mae)
            mlflow.log_metric(f"{seg_name}_pinball_avg", pinball_avg)

            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                pr_auc = float(average_precision_score(y_bin, y_prob))
                summary_metrics[f"{seg_name}_pr_auc"] = pr_auc
                mlflow.log_metric(f"{seg_name}_pr_auc", pr_auc)

        print("\n=== RESULTADOS BACKTESTING (VENTANA EXPANSIVA) ===")
        for k, v in summary_metrics.items():
            print(f"  {k:<25}: {v:.4f}")

        return summary_metrics


if __name__ == "__main__":
    run_expanding_backtest()