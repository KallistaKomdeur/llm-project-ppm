import pandas as pd
import numpy as np
from utils.log_schema import load_log_schema

def clean_log(original_csv_path: str, clean_csv_path: str, log_name: str,
              std_cutoff: float = 3.0, min_end_freq: int = 5, inactivity_days: float = 30):
    """
    Cleans a log CSV by:
    - Inferring completed cases dynamically (common last events)
    - Inferring completed cases by inactivity gaps (no events for inactivity_days)
    - Removing outliers longer than mean + std_cutoff * std
    - Normalizing timestamps to ISO8601 format
    """
    # Load schema
    schema = load_log_schema(log_name)
    case_col = schema.case_id
    activity_col = schema.activity
    timestamp_col = schema.timestamp

    if not timestamp_col:
        raise ValueError("Timestamp column must be defined in the schema to compute durations.")
    
    df = pd.read_csv(original_csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="raise")
    df[timestamp_col] = df[timestamp_col].dt.floor('s')

    # Common last events
    last_events = df.groupby(case_col).apply(lambda g: g.sort_values(timestamp_col).iloc[-1])[activity_col]
    end_event_counts = last_events.value_counts()
    inferred_end_events = end_event_counts[end_event_counts >= min_end_freq].index.tolist()
    cases_by_end_event = last_events[last_events.isin(inferred_end_events)].index
    
    # Inactivity gaps
    max_timestamp = df[timestamp_col].max()
    last_event_per_case = df.groupby(case_col)[timestamp_col].max()
    cases_by_inactivity = last_event_per_case[last_event_per_case <= (max_timestamp - pd.Timedelta(days=inactivity_days))].index
    
    # Union of the two
    completed_case_ids = pd.Index(cases_by_end_event).union(cases_by_inactivity)
    df = df[df[case_col].isin(completed_case_ids)].reset_index(drop=True)
    
    # Remove duration outliers
    durations = df.groupby(case_col)[timestamp_col].agg(['min', 'max'])
    durations['duration_days'] = (durations['max'] - durations['min']).dt.total_seconds() / (24*60*60)
    
    mean_dur = durations['duration_days'].mean()
    std_dur = durations['duration_days'].std()
    max_allowed = mean_dur + std_cutoff * std_dur
    good_cases = durations[durations['duration_days'] <= max_allowed].index
    df = df[df[case_col].isin(good_cases)].reset_index(drop=True)
    
    df[timestamp_col] = df[timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    df.to_csv(clean_csv_path, index=False)
