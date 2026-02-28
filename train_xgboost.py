import argparse
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from utils.log_schema import load_log_schema


def run_xgb(df, case_col, activity_col, timestamp_col):
    """
    Train XGBoost using only the sequence of activities per case.
    """
    sequences = []
    max_seq_len = 0

    # Convert activities to ordered lists per case
    for case_id, group in df.groupby(case_col):
        group = group.sort_values(timestamp_col)
        seq = group[activity_col].tolist()
        sequences.append({
            "case_id": case_id,
            "sequence": seq,
            "duration": (group[timestamp_col].max() - group[timestamp_col].min()).total_seconds() / 60.0
        })
        max_seq_len = max(max_seq_len, len(seq))

    # Build sequence features
    feature_rows = []
    valid_sequences = []
    for case in sequences:
        duration = case["duration"]
        if pd.isna(duration) or duration in (float('inf'), float('-inf')) or duration < 0:
            continue
        padded = case["sequence"] + [None] * (max_seq_len - len(case["sequence"]))      # If not longest case, pad remaining activities with None
        feature_rows.append(padded)
        valid_sequences.append(case)

    feature_df = pd.DataFrame(feature_rows)
    feature_df = pd.get_dummies(feature_df, dummy_na=True, drop_first=True)             # One-hot encode activities per position
    y = [c["duration"] for c in valid_sequences]                                        # Target = total duration

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.2, random_state=42)

    # XGBoost model
    xgb_model = XGBRegressor(objective='reg:squarederror', tree_method='hist', n_jobs=1,random_state=42)

    # Hyperparameter search
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=20, scoring='neg_root_mean_squared_error', cv=5, verbose=0, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Metrics
    y_pred = best_model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "num_train_cases": len(y_train),
        "num_test_cases": len(y_test),
        "best_params": search.best_params_
    }

    return best_model, metrics


def train_xgb(log_name, logs_dir = "logs", results_dir = "results"):
    """
    Train sequence-based XGBoost on raw and cleaned logs, save metrics as JSON.
    """
    # Load schema and get column names
    schema = load_log_schema(log_name)
    case_col = schema.case_id
    activity_col = schema.activity
    timestamp_col = schema.timestamp

    results = {}

    # Raw log
    raw_csv = Path(logs_dir) / log_name / f"{log_name}.csv"
    df_raw = pd.read_csv(raw_csv)
    df_raw[timestamp_col] = pd.to_datetime(df_raw[timestamp_col], utc=True)
    print(f"Processing raw log: {raw_csv}")
    _, metrics_raw = run_xgb(df_raw, case_col, activity_col, timestamp_col)
    results['raw'] = metrics_raw

    # Clean log if exists
    clean_csv = Path(logs_dir) / log_name / f"{log_name}_clean.csv"
    if clean_csv.exists():
        df_clean = pd.read_csv(clean_csv)
        df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], utc=True)
        print(f"Processing cleaned log: {clean_csv}")
        _, metrics_clean = run_xgb(df_clean, case_col, activity_col, timestamp_col)
        results['clean'] = metrics_clean
    else:
        print("No cleaned log found, training xgboost skipped")

    # Save metrics to file
    output_dir = Path(results_dir) / log_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"xgboost_{log_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train benchmark XGBoost model on log")
    parser.add_argument("log_name", type=str)
    args = parser.parse_args()
    train_xgb(args.log_name)