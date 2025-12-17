import pandas as pd
from typing import Dict, Any
import random

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
