import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import json
from utils.log_schema import load_log_schema

WORKLOAD_WINDOWS = [60, 240, 1440]  # minutes

# ======================
# FEATURE FUNCTIONS
# ======================

def extract_timestamp_features(group, timestamp_col):
    group = group.sort_values(timestamp_col, ascending=False, kind="mergesort")

    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    group["timesincelastevent"] = tmp.fillna(pd.Timedelta(0)).dt.total_seconds() / 60

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    group["timesincecasestart"] = tmp.dt.total_seconds() / 60

    group = group.sort_values(timestamp_col, ascending=True, kind="mergesort")
    group["event_nr"] = range(1, len(group) + 1)

    return group


def get_prev_resource(group, resource_col):
    group = group.sort_values(group.columns[0])
    group["prev_resource"] = group[resource_col].shift(1).fillna("first")
    return group


def build_event_features(group, timestamp_col, activity_col):
    group = group.sort_values(timestamp_col)
    cols = set(group.columns)

    act_freq = group[activity_col].value_counts().to_dict() if activity_col in cols else {}
    handoff_freq = group["prev_resource"].value_counts().to_dict() if "prev_resource" in cols else {}

    seq = []

    for _, row in group.iterrows():
        event_features = {}

        for k in [
            "timesincemidnight","weekday","month","timesincelastevent","timesincecasestart",
            "event_nr","prev_resource","ent_act","ent_case","ent_handoff","busyness",
            "open_cases","res_work_items","res_cases","res_unique_tasks",
            "res_unique_handoffs","res_ratio_workitems_global",
            "res_ratio_workitems_resource","res_ratio_task_specific",
            "res_ratio_handoff_specific","res_work_items_per_min"
        ]:
            if k in cols:
                event_features[k] = row[k]

        if act_freq:
            event_features["act_freq"] = act_freq
        if handoff_freq:
            event_features["handoff_freq"] = handoff_freq

        seq.append([
            row[activity_col],
            row.get("timesincecasestart", 0),
            event_features
        ])

    return seq


def safe_convert(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_workload_windows(df, windows, case_id_col, timestamp_col):
    starts = df.groupby(case_id_col)[timestamp_col].min()
    ends = df.groupby(case_id_col)[timestamp_col].max()

    def workload_at(t, w):
        start_cutoff = t - pd.Timedelta(minutes=w)
        return ((starts <= t) & (ends > start_cutoff)).sum()

    return pd.DataFrame({
        f"open_cases_{w}min": df[timestamp_col].apply(lambda x: workload_at(x, w))
        for w in windows
    })


# ======================
# MAIN FUNCTION
# ======================

def preprocess_log(log_name: str):

    schema = load_log_schema(log_name)
    case_id_col = schema.case_id
    activity_col = schema.activity
    resource_col = schema.resource
    timestamp_col = schema.timestamp

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name
    input_file = log_dir / f"{log_name}.csv"
    output_file = log_dir / f"{log_name}_preprocessed.jsonl"

    data = pd.read_csv(input_file, encoding="latin-1")

    # ======================
    # Timestamp features
    # ======================
    if timestamp_col in data.columns:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)

        data["timesincemidnight"] = (
            data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
        )
        data["weekday"] = data[timestamp_col].dt.weekday
        data["month"] = data[timestamp_col].dt.month

        data = pd.concat(
            [extract_timestamp_features(g, timestamp_col)
             for _, g in data.groupby(case_id_col)],
            ignore_index=True
        )

    # ======================
    # Previous resource
    # ======================
    if resource_col in data.columns:
        data = pd.concat(
            [get_prev_resource(g, resource_col)
             for _, g in data.groupby(case_id_col)],
            ignore_index=True
        )

    # ======================
    # FAST RESOURCE EXPERIENCE (O(N))
    # ======================
    if (
        resource_col in data.columns
        and activity_col in data.columns
        and timestamp_col in data.columns
    ):

        data = data.sort_values(timestamp_col).reset_index(drop=True)

        int_cols = [
            "res_work_items","res_cases","res_unique_tasks",
            "res_unique_handoffs"
        ]

        float_cols = [
            "res_ratio_workitems_global","res_ratio_workitems_resource",
            "res_ratio_task_specific","res_ratio_handoff_specific",
            "res_work_items_per_min","ent_act","ent_case",
            "ent_handoff","busyness"
        ]

        for col in int_cols:
            data[col] = 0
        for col in float_cols:
            data[col] = 0.0

        global_cases_seen = set()

        for res, g in data.groupby(resource_col, sort=False):

            tasks = defaultdict(int)
            cases = set()
            handoffs = defaultdict(int)
            handoff_set = set()
            first_ts = None
            total = 0

            for idx in g.index:

                row = data.loc[idx]

                cid = row[case_id_col]
                act = row[activity_col]
                ho = row["prev_resource"]
                ts = row[timestamp_col]

                total += 1
                global_cases_seen.add(cid)

                cases.add(cid)
                tasks[act] += 1
                handoffs[ho] += 1
                handoff_set.add(ho)

                if first_ts is None:
                    first_ts = ts

                duration = (ts - first_ts).total_seconds() / 60

                # entropy from counts
                def entropy_from_counts(counter):
                    n = sum(counter.values())
                    if n <= 1:
                        return 0.0
                    probs = np.array(list(counter.values())) / n
                    return float(-(probs * np.log(probs)).sum())

                data.at[idx, "res_work_items"] = total
                data.at[idx, "res_cases"] = len(cases)
                data.at[idx, "res_unique_tasks"] = len(tasks)
                data.at[idx, "res_unique_handoffs"] = len(handoff_set)

                data.at[idx, "res_ratio_workitems_global"] = total / len(global_cases_seen)
                data.at[idx, "res_ratio_workitems_resource"] = total / len(cases)
                data.at[idx, "res_ratio_task_specific"] = tasks[act] / total
                data.at[idx, "res_ratio_handoff_specific"] = handoffs[ho] / total
                data.at[idx, "res_work_items_per_min"] = (
                    total / duration if duration > 0 else 0.0
                )

                data.at[idx, "ent_act"] = entropy_from_counts(tasks)
                data.at[idx, "ent_case"] = entropy_from_counts({c: 1 for c in cases})
                data.at[idx, "ent_handoff"] = entropy_from_counts(handoffs)

                days = (ts - first_ts).days
                data.at[idx, "busyness"] = total / days if days > 0 else 0.0

    # ======================
    # Workload
    # ======================
    if timestamp_col in data.columns:
        case_windows = data.groupby(case_id_col)[timestamp_col].agg(
            start="min", end="max"
        )

        data["open_cases"] = data[timestamp_col].apply(
            lambda t: ((case_windows["start"] <= t) &
                       (case_windows["end"] > t)).sum()
        )

        data = pd.concat(
            [data, compute_workload_windows(
                data, WORKLOAD_WINDOWS, case_id_col, timestamp_col)],
            axis=1
        )

    # ======================
    # Case attributes
    # ======================
    case_attr_cols = [
        c for c in data.columns
        if c not in [case_id_col, activity_col,
                     resource_col, timestamp_col]
        and data.groupby(case_id_col)[c].nunique().max() == 1
    ]

    # ======================
    # Build JSON
    # ======================
    output = []

    for cid, group in data.groupby(case_id_col):

        total_time = (
            (group[timestamp_col].max()
             - group[timestamp_col].min()).total_seconds() / 60
            if timestamp_col in group.columns else 0
        )

        case_attrs = {c: group[c].iloc[0] for c in case_attr_cols}

        output.append({
            case_id_col: cid,
            **case_attrs,
            "ActTimeSeq": build_event_features(
                group, timestamp_col, activity_col
            ) if activity_col in group.columns else [],
            "total_time": total_time
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for case in output:
            f.write(json.dumps(case, default=safe_convert) + "\n")

    return output_file
