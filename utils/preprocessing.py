import pandas as pd
from typing import Dict, Any
import random
import json
from pathlib import Path
from utils.split import train_test_split_by_case, truncate_test_traces

def load_or_preprocess(log_name: str, max_test_events: int = 5) -> dict:
    log_dir = Path("logs") / log_name
    raw_path = log_dir / f"{log_name}.csv"
    train_path = log_dir / f"{log_name}_train.json"
    test_path = log_dir / f"{log_name}_test.json"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Load existing JSONs
    if train_path.exists() and test_path.exists():
        print(f"Loading preprocessed train/test logs from {log_dir}")
        with open(train_path) as f:
            train_traces = json.load(f)
        with open(test_path) as f:
            test_traces = json.load(f)
        return {"train": train_traces, "test": test_traces}

    # Load raw CSV
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw log not found at {raw_path}")
    print(f"Processing raw log {raw_path}")
    df = pd.read_csv(raw_path)

    # Split train/test by case
    train_df, test_df = train_test_split_by_case(df)

    # Preprocess
    train_traces = preprocess(train_df, truncate=False)
    test_traces = preprocess(test_df, truncate=True, max_events=max_test_events)

    # Save JSONs
    with open(train_path, "w") as f:
        json.dump(train_traces, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_traces, f, indent=2)

    print(f"Saved preprocessed train/test logs to {log_dir}")
    return {"train": train_traces, "test": test_traces}

def preprocess_trace(trace: pd.DataFrame, case_col: str, activity_col: str, resource_col: str, time_col: str, truncate: bool = False, max_events: int = None) -> Dict[str, Any]:
    trace = trace.reset_index(drop=True)
    
    # Trace-level attributes: columns constant within this trace
    trace_attrs = {
        col: trace[col].iloc[0]
        for col in trace.columns
        if col not in [case_col, activity_col, resource_col, time_col]
        and trace[col].nunique() == 1
    }

    # Optionally truncate the trace for test set
    if truncate:
        if len(trace) <= 1:
            # Cannot truncate, leave as-is
            cutoff = len(trace)
        elif max_events is None or max_events >= len(trace):
            cutoff = random.randint(1, len(trace)-1)
        else:
            cutoff = min(len(trace), random.randint(1, max_events))
        trace = trace.iloc[:cutoff]

    # Compute event durations
    durations = trace[time_col].shift(-1) - trace[time_col]
    durations = durations.apply(lambda x: x.total_seconds() if pd.notnull(x) else 0.0)

    events = [
        {
            "activity": row[activity_col],
            "resource": row[resource_col],
            "duration": durations.iloc[i]
        }
        for i, row in trace.iterrows()
    ]

    # Total duration
    total_duration = "RUNNING" if truncate else (trace[time_col].iloc[-1] - trace[time_col].iloc[0]).total_seconds()

    return {
        "trace_attributes": trace_attrs,
        "events": events,
        "total_duration": total_duration
    }


def preprocess(df: pd.DataFrame, case_col: str = "case", activity_col: str = "activity", resource_col: str = "resource", time_col: str = "timestamp", truncate: bool = False, max_events: int = None) -> Dict[str, Dict[str, Any]]:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values([case_col, time_col])

    traces = {}
    for case_id, trace in df.groupby(case_col):
        traces[str(case_id)] = preprocess_trace(trace, case_col, activity_col, resource_col, time_col, truncate=truncate, max_events=max_events)

    return traces
