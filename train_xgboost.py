import argparse
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

def build_features(sequences, max_seq_len):
    """ Build sequences of (activity, time) with padding"""
    feature_rows = []
    for seq in sequences:
        row = {}
        for pos in range(max_seq_len):
            if pos < len(seq):
                act, elapsed = seq[pos]
                row[f"pos_{pos}_act"] = act
                row[f"pos_{pos}_elapsed"] = elapsed
            else:
                row[f"pos_{pos}_act"] = None 
                row[f"pos_{pos}_elapsed"] = 0.0
        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)
    act_cols = [c for c in df.columns if c.endswith("_act")]
    df = pd.get_dummies(df, columns=act_cols, dummy_na=True, drop_first=False)
    return df

def truncate_sequence(seq, fraction):
    """Keep only the first fraction of events"""
    keep = max(1, int(len(seq) * fraction))
    return seq[:keep]

def run_xgb_full(log_name, truncation_fraction, logs_dir="logs", results_dir="results"):
    """ Train on full cases from preprocessed file, evaluate on truncated test cases"""
    jsonl_path = Path(logs_dir) / log_name / f"{log_name}_preprocessed.jsonl"
    sequences = []
    durations = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            case_data = json.loads(line)
            durations.append(case_data["total_time"])
            seq = [(event[0], event[1]) for event in case_data["ActTimeSeq"]]   # only taking activity and time, ignoring inter-case
            sequences.append(seq)

    split_idx = int(len(sequences) * 0.8)
    train_seqs = sequences[:split_idx]
    test_seqs = sequences[split_idx:]
    y_train = durations[:split_idx]
    y_test = durations[split_idx:]

    test_seqs_trunc = [truncate_sequence(s, truncation_fraction) for s in test_seqs]
    max_seq_len = max(len(s) for s in train_seqs)
    all_seqs = train_seqs + test_seqs_trunc
    all_df   = build_features(all_seqs, max_seq_len)

    X_train = all_df.iloc[:len(train_seqs)].reset_index(drop=True)
    X_test  = all_df.iloc[len(train_seqs):].reset_index(drop=True)

    # XGBoost with hyperparameter search
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

    # Evaluation
    y_pred = best_model.predict(X_test)
    metrics = {
        "truncation_fraction": truncation_fraction,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "num_train_cases": len(y_train),
        "num_test_cases": len(y_test),
        "best_params": search.best_params_,
    }

    # Saving
    out_dir = Path(results_dir) / log_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgboost_full_{log_name}.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"R2: {metrics['r2']:.4f}")
    return best_model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_name", type=str)
    args = parser.parse_args()
    run_xgb_full(args.log_name, truncation_fraction = 0.3)