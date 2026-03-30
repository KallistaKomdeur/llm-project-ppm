import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

def build_row(seq, max_seq_len):
    """Build one row of (activity, time) with padding"""
    row = {}
    for pos in range(max_seq_len):
        if pos < len(seq):
            act, elapsed = seq[pos]
            row[f"pos_{pos}_act"]     = act
            row[f"pos_{pos}_elapsed"] = float(elapsed)
        else:
            row[f"pos_{pos}_act"]     = None   # becomes NaN dummy
            row[f"pos_{pos}_elapsed"] = 0.0
    return row

def build_datasets(data: list, n_sets: int):
    """ Build dataset from fixed sets"""
    train_rows, y_train = [], []
    test_rows,  y_test  = [], []

    train_seqs, test_seqs = [], []
    max_seq_len = 0

    for s in data[:n_sets]:
        for ex in s["examples"]:
            duration = ex.get("total_time")
            if not isinstance(duration, (int, float)) or duration < 0:
                continue
            seq = [(entry[0], entry[1]) for entry in ex["ActTimeSeq"]]
            train_seqs.append((seq, float(duration)))
            max_seq_len = max(max_seq_len, len(seq))

        tc = s["test_case"]
        true_time = tc.get("true_total_time")
        if isinstance(true_time, (int, float)) and true_time >= 0:
            seq = [(entry[0], entry[1]) for entry in tc["ActTimeSeq"]]
            test_seqs.append((seq, float(true_time)))

    # Build raw feature rows (before one-hot)
    all_raw_rows = []
    for seq, dur in train_seqs:
        all_raw_rows.append(build_row(seq, max_seq_len))
        y_train.append(dur)
    for seq, dur in test_seqs:
        all_raw_rows.append(build_row(seq, max_seq_len))
        y_test.append(dur)

    # One-hot encode activity columns together
    all_df   = pd.DataFrame(all_raw_rows)
    act_cols = [c for c in all_df.columns if c.endswith("_act")]
    all_df   = pd.get_dummies(all_df, columns=act_cols, dummy_na=True, drop_first=False)

    n_train  = len(train_seqs)
    X_train  = all_df.iloc[:n_train].reset_index(drop=True)
    X_test   = all_df.iloc[n_train:].reset_index(drop=True)

    return X_train, y_train, X_test, y_test

def run_xgb_fixed_sets(log_name, logs_dir="logs", results_dir="results", n_sets=100):
    sets_path = f"{logs_dir}/{log_name}/{log_name}_fixed_sets.json"
    results_path = f"{results_dir}/{log_name}/xgboost_fixed_sets_{log_name}.json"

    with open(sets_path, encoding="utf-8") as f:
        data = json.load(f)

    X_train, y_train, X_test, y_test = build_datasets(data, n_sets)

    xgb_model = XGBRegressor(objective="reg:absoluteerror", tree_method="hist")

    param_dist = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "num_train_cases": len(y_train),
        "num_test_cases": len(y_test),
        "best_params": search.best_params_,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"R2: {metrics['r2']:.4f}")
    return best_model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_name", type=str)
    args = parser.parse_args()
    run_xgb_fixed_sets(args.log_name)