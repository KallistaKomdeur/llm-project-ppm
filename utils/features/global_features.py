import pandas as pd
from typing import Dict, Any

def compute_global_features(
    df: pd.DataFrame,
    case_col="case",
    activity_col="activity",
    resource_col="resource",
    time_col="timestamp"
) -> Dict[str, Any]:

    df = df.copy()
    df.columns = df.columns.str.strip()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([case_col, time_col])

    activity_stats = df.groupby(activity_col).apply(
        lambda x: {
            "avg": x[time_col].diff().dt.total_seconds().fillna(0).mean(),
            "median": x[time_col].diff().dt.total_seconds().fillna(0).median(),
            "std": x[time_col].diff().dt.total_seconds().fillna(0).std()
        }
    ).to_dict()

    resource_stats = df.groupby(resource_col).apply(
        lambda x: {
            "avg": x[time_col].diff().dt.total_seconds().fillna(0).mean(),
            "median": x[time_col].diff().dt.total_seconds().fillna(0).median(),
            "task_count": len(x)
        }
    ).to_dict()

    return {
        "activity_stats": activity_stats,
        "resource_stats": resource_stats
    }


def format_global_features(features: Dict[str, Any], max_items=5) -> str:
    lines = []

    lines.append("### Activity-level stats (avg duration in seconds)")
    for i, (act, s) in enumerate(features["activity_stats"].items()):
        if i >= max_items:
            break
        lines.append(
            f"{act}: avg={s['avg']:.1f}, median={s['median']:.1f}, std={s['std']:.1f}"
        )

    lines.append("\n### Resource-level stats (avg duration in seconds)")
    for i, (res, s) in enumerate(features["resource_stats"].items()):
        if i >= max_items:
            break
        lines.append(
            f"{res}: avg={s['avg']:.1f}, median={s['median']:.1f}, task_count={s['task_count']}"
        )

    return "\n".join(lines)
