import pandas as pd
from utils.log_schema import load_log_schema

def clean_log(
    original_csv_path: str,
    clean_csv_path: str,
    log_name: str,
):
    """
    Cleans a log CSV by:
    - Filtering cases ending before a fixed time bound
    - Removing the top 5% longest cases
    - Normalizing timestamps
    """

    # Load schema
    schema = load_log_schema(log_name)
    case_col = schema.case_id
    timestamp_col = schema.timestamp

    if not timestamp_col:
        raise ValueError("Timestamp column must be defined in the schema.")

    # Load data
    df = pd.read_csv(original_csv_path)

    # Parse timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="raise")
    df[timestamp_col] = df[timestamp_col].dt.floor("s")

    # --- Compute case durations ---
    durations = df.groupby(case_col)[timestamp_col].agg(["min", "max"])
    durations["duration_days"] = (
        (durations["max"] - durations["min"]).dt.total_seconds() / (24 * 60 * 60)
    )

    # Remove top 5% longest cases
    cutoff = durations["duration_days"].quantile(0.95)
    good_cases = durations[durations["duration_days"] <= cutoff].index
    df = df[df[case_col].isin(good_cases)].reset_index(drop=True)

    # --- Time window filtering ---
    case_starts = df.groupby(case_col)[timestamp_col].min()
    case_ends = df.groupby(case_col)[timestamp_col].max()

    end_bound = pd.Timestamp("2012-03-01", tz="UTC")  # dataset-specific
    valid_time_cases = case_starts[(case_ends < end_bound)].index
    df = df[df[case_col].isin(valid_time_cases)].reset_index(drop=True)

    # Format timestamps and save
    df[timestamp_col] = df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(clean_csv_path, index=False)
