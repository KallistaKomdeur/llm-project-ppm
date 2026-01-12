import pandas as pd
from typing import Dict

def inter_case_state_at(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    case_col="case",
    activity_col="activity",
    resource_col="resource",
    time_col="timestamp",
    exclude_case=None
) -> Dict:

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([case_col, time_col])

    running = []

    for case_id, trace in df.groupby(case_col):
        if exclude_case is not None and str(case_id) == str(exclude_case):
            continue

        trace = trace.reset_index(drop=True)

        for i in range(len(trace) - 1):
            start = trace.loc[i, time_col]
            end = trace.loc[i + 1, time_col]

            if start <= current_time < end:
                running.append({
                    "activity": trace.loc[i, activity_col],
                    "resource": trace.loc[i, resource_col],
                    "elapsed": int((current_time - start).total_seconds())
                })
                break

    return {
        "running_count": len(running),
        "running": running
    }


def format_inter_case_state(state: Dict, max_items=5) -> str:
    lines = [
        f"There are {state['running_count']} other activities being performed right now."
    ]

    for r in state["running"][:max_items]:
        lines.append(
            f"Resource {r['resource']} has been doing task "
            f"{r['activity']} for {r['elapsed']} seconds."
        )

    return "\n".join(lines)
