import pandas as pd
from typing import Dict, Any

def preprocess(df: pd.DataFrame, case_col: str = "case", activity_col: str = "activity", resource_col: str = "resource", time_col: str = "timestamp") -> Dict[str, Dict[str, Any]]:
    """
    Automatically detects trace attributes (columns constant within a case) and builds
    a structured LLM input per trace.
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values([case_col, time_col])

    traces = {}

    for case_id, trace in df.groupby(case_col):
        trace = trace.reset_index(drop=True)

        # Detect trace-level attributes: columns constant within this trace
        trace_attrs = {
            col: trace[col].iloc[0]
            for col in trace.columns
            if col not in [case_col, activity_col, resource_col, time_col]
            and trace[col].nunique() == 1
        }

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

        # Total duration from first to last event
        total_duration = (trace[time_col].iloc[-1] - trace[time_col].iloc[0]).total_seconds()

        traces[str(case_id)] = {
            "trace_attributes": trace_attrs,
            "events": events,
            "total_duration": total_duration
        }

    return traces
