import pandas as pd
from typing import Dict, Any
from collections import defaultdict

def compute_inter_case_features(
    df: pd.DataFrame, 
    case_col: str = "case", 
    activity_col: str = "activity",
    resource_col: str = "resource",
    time_col: str = "timestamp"
) -> Dict[str, Any]:
    """
    Computes inter-case features from a completed event log for LLM prompts.
    Automatically strips column names to avoid KeyErrors.
    
    Returns a dictionary with summary statistics for activities, resources, and case attributes.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()  # <-- strip spaces from column names

    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in dataframe. Columns available: {df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values([case_col, time_col])

    # Activity-level stats
    activity_stats = df.groupby(activity_col).apply(
        lambda x: {
            "avg_duration": x[time_col].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().mean(),
            "median_duration": x[time_col].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().median(),
            "std_duration": x[time_col].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().std(),
            "count": len(x)
        }
    ).to_dict()

    # Resource-level stats
    resource_stats = df.groupby(resource_col).apply(
        lambda x: {
            "avg_duration": x[time_col].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().mean(),
            "median_duration": x[time_col].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().median(),
            "task_count": len(x)
        }
    ).to_dict()

    # Case attribute stats
    case_attributes = [col for col in df.columns if col not in [case_col, activity_col, resource_col, time_col]]
    case_attr_stats = {}
    for attr in case_attributes:
        counts = df.groupby(case_col)[attr].first().value_counts().to_dict()
        case_attr_stats[attr] = counts

    # Total durations per case
    total_durations = df.groupby(case_col)[time_col].agg(['min', 'max'])
    total_durations["duration_sec"] = (total_durations["max"] - total_durations["min"]).dt.total_seconds()
    duration_stats = {
        "avg_case_duration": total_durations["duration_sec"].mean(),
        "median_case_duration": total_durations["duration_sec"].median(),
        "min_case_duration": total_durations["duration_sec"].min(),
        "max_case_duration": total_durations["duration_sec"].max()
    }

    return {
        "activity_stats": activity_stats,
        "resource_stats": resource_stats,
        "case_attr_stats": case_attr_stats,
        "case_duration_stats": duration_stats
    }


def format_features_for_prompt(features: Dict[str, Any], max_items: int = 5) -> str:
    """
    Converts computed inter-case features into a short text snippet for the LLM prompt.
    Limits the number of activities/resources to max_items for readability.
    """
    lines = []

    lines.append("### Activity-level stats (avg duration in seconds)")
    for i, (act, stats) in enumerate(features["activity_stats"].items()):
        if i >= max_items:
            break
        lines.append(f"{act}: avg={stats['avg_duration']:.1f}, median={stats['median_duration']:.1f}, std={stats['std_duration']:.1f}")

    lines.append("\n### Resource-level stats (avg duration in seconds)")
    for i, (res, stats) in enumerate(features["resource_stats"].items()):
        if i >= max_items:
            break
        lines.append(f"{res}: avg={stats['avg_duration']:.1f}, median={stats['median_duration']:.1f}, task_count={stats['task_count']}")

    lines.append("\n### Case-level attribute distributions")
    for attr, counts in features["case_attr_stats"].items():
        lines.append(f"{attr}: " + ", ".join([f"{k}={v}" for k, v in list(counts.items())[:max_items]]))

    lines.append("\n### Overall case duration stats (seconds)")
    dur_stats = features["case_duration_stats"]
    lines.append(f"avg={dur_stats['avg_case_duration']:.1f}, median={dur_stats['median_case_duration']:.1f}, "
                 f"min={dur_stats['min_case_duration']:.1f}, max={dur_stats['max_case_duration']:.1f}")

    return "\n".join(lines)
