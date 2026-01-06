import pandas as pd
from typing import List, Dict

def compute_running_activities(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    case_col: str = "case",
    activity_col: str = "activity",
    resource_col: str = "resource",
    time_col: str = "timestamp"
) -> List[Dict]:
    """
    Returns a list of currently running activities at current_time.
    Each item is {activity, resource, elapsed_time}.
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([case_col, time_col])

    running = []

    for case_id, trace in df.groupby(case_col):
        trace = trace.reset_index(drop=True)

        for i in range(len(trace) - 1):
            start = trace.loc[i, time_col]
            end = trace.loc[i + 1, time_col]

            if start <= current_time < end:
                running.append({
                    "activity": trace.loc[i, activity_col],
                    "resource": trace.loc[i, resource_col],
                    "elapsed_time": (current_time - start).total_seconds()
                })
                break

    return running

def format_running_activities_for_prompt(
    running: List[Dict],
    max_listed: int = 5
) -> str:
    lines = []
    total = len(running)

    lines.append(f"There are {total} other activities being performed right now.")

    for item in running[:max_listed]:
        lines.append(
            f"Resource {item['resource']} has been doing task "
            f"{item['activity']} for {int(item['elapsed_time'])} seconds."
        )

    return "\n".join(lines)
