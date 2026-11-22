FEATURE_COLS = [
    "station_id",
    "horizon_day",
    "river_level_m",
    "precip_mm",
    "precip_acc_3d",
    "precip_acc_7d",
    "precip_acc_15d",
    "precip_acc_30d",
    "river_diff_24h",
    "river_diff_48h",
    "river_diff_7d",
    "river_roll_mean_7d",
    "oni_index",
    "oni_trend_90d",
    "is_el_nino",
    "future_precip_acc_h",
    "future_precip_day_h",
]

CATEGORICAL_COLS = ["station_id"]
TARGET_COL = "target_river_level_m"
QUANTILES = [0.10, 0.50, 0.90]