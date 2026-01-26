import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from pathlib import Path
import json
from utils.log_schema import load_log_schema

# ======================
# CONSTANTS
# ======================

WORKLOAD_WINDOWS = [60, 240, 1440]  # minutes

# ======================
# FEATURE FUNCTIONS
# ======================
def extract_timestamp_features(group, timestamp_col):
    """
    Gets timestamp features:
    - Time since the previous event
    - Time since the start of the case
    - The how manieth event in the case this is
    """
    group = group.sort_values(timestamp_col, ascending=False, kind="mergesort")

    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    group["timesincelastevent"] = tmp.fillna(pd.Timedelta(0)).dt.total_seconds() / 60

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    group["timesincecasestart"] = tmp.dt.total_seconds() / 60

    group = group.sort_values(timestamp_col, ascending=True, kind="mergesort")
    group["event_nr"] = range(1, len(group) + 1)

    return group

def ent(data, col):
    """
    Returns the entropy of a particular attribute
    """
    p = data[col].value_counts(normalize=True)
    return stats.entropy(p)

def get_prev_resource(group, resource_col):
    """
    Get the previous resource that was working on this case.
    """
    group["prev_resource"] = group[resource_col].shift(1).fillna("first")
    return group

def extract_resource_experience(group, case_id_col, activity_col, timestamp_col):
    """
    Extracts the resource experience, which is defined as:
    - Total number of tasks (= activity instances) the resource has worked on
    - Total number of cases the resource has worked on
    - Total number of activities the resource has worked on
    - Total number of handoffs the resource has participated in
    - The entropy of the activities, cases, and handoffs
    - The busyness of the resource
    """
    group = group.reset_index(drop=True)

    for i in range(len(group)):
        hist = group.iloc[: i + 1]

        group.loc[i, "ent_act"] = ent(hist, activity_col)
        group.loc[i, "ent_case"] = ent(hist, case_id_col)
        group.loc[i, "ent_handoff"] = ent(hist, "prev_resource")

        days = (hist[timestamp_col].max() - hist[timestamp_col].min()).days
        group.loc[i, "busyness"] = len(hist) / days if days > 0 else 0

    return group

def compute_workload_windows(df, windows, case_id_col, timestamp_col):
    starts = df.groupby(case_id_col)[timestamp_col].min().values.astype(np.int64)
    ends = df.groupby(case_id_col)[timestamp_col].max().values.astype(np.int64)

    def workload_at(t, w):
        t = np.int64(t)
        start_cutoff = t - np.int64(w * 60 * 1_000_000_000)
        return (starts <= t).sum() - (ends <= start_cutoff).sum()

    res = {}
    for w in windows:
        res[f"open_cases_{w}min"] = df[timestamp_col].apply(lambda x: workload_at(x.value, w))

    return pd.DataFrame(res)


def build_event_features(group, timestamp_col, activity_col):
    """
    Summarizes all previous event features
    """
    group = group.sort_values(timestamp_col)
    cols = set(group.columns)

    act_freq = (group[activity_col].value_counts().to_dict() if activity_col in cols else {})
    handoff_freq = (group["prev_resource"].value_counts().to_dict() if "prev_resource" in cols else {})

    seq = []
    for _, row in group.iterrows():
        event_features = {}

        def add(k):
            if k in cols:
                event_features[k] = row[k]

        add("timesincemidnight")
        add("weekday")
        add("month")
        add("timesincelastevent")
        add("timesincecasestart")
        add("event_nr")
        add("prev_resource")
        add("ent_act")
        add("ent_case")
        add("ent_handoff")
        add("busyness")
        add("open_cases")
        add("res_work_items")
        add("res_cases")
        add("res_unique_tasks")
        add("res_unique_handoffs")
        add("res_ratio_workitems_global")
        add("res_ratio_workitems_resource")
        add("res_ratio_task_specific")
        add("res_ratio_handoff_specific")
        add("res_work_items_per_min")

        if act_freq:
            event_features["act_freq"] = act_freq
        if handoff_freq:
            event_features["handoff_freq"] = handoff_freq

        seq.append([row[activity_col], row["timesincecasestart"] if "timesincecasestart" in cols else 0, event_features])

    return seq

def safe_convert(obj):
    """
    Transforms event features to a format that can safely be converted into a json file
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ======================
# MAIN FUNCTION
# ======================
def preprocess_log(log_name: str):
    """
    Processes a log and writes log_name_preprocessed.jsonl
    to root/logs/<log_name>/
    """
    schema = load_log_schema(log_name)
    case_id_col = schema.case_id
    activity_col = schema.activity
    resource_col = schema.resource
    timestamp_col = schema.timestamp

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name
    input_file = log_dir / f"{log_name}.csv"
    output_file = log_dir / f"{log_name}_preprocessed.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(f"Log not found: {input_file}")

    data = pd.read_csv(input_file, encoding="latin-1")

    available_cols = set(data.columns)
    has_ts = timestamp_col in available_cols
    has_res = resource_col in available_cols
    has_act = activity_col in available_cols

    # Timestamp features
    if has_ts:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)

        data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
        data["weekday"] = data[timestamp_col].dt.weekday
        data["month"] = data[timestamp_col].dt.month

        data = data.groupby(case_id_col).apply(lambda g: extract_timestamp_features(g, timestamp_col)).reset_index(drop=True)

    # Previous resource
    if has_res:    
        data = data.groupby(case_id_col).apply(lambda g: get_prev_resource(g, resource_col))

    # Resource experience
    if has_res and has_act and has_ts:
        data = data.sort_values(timestamp_col)
        data = data.groupby(resource_col).apply(lambda g: extract_resource_experience(g, case_id_col, activity_col, timestamp_col)).reset_index(drop=True)

    # Open cases
    if has_ts:
        case_windows = data.groupby(case_id_col)[timestamp_col].agg(start="min", end="max")
        data["open_cases"] = data[timestamp_col].apply(lambda t: ((case_windows["start"] <= t) & (case_windows["end"] > t)).sum())

        # Workload windows
        data = data.join(compute_workload_windows(data, WORKLOAD_WINDOWS, case_id_col, timestamp_col))

    # Resource-level stats
    if has_res and has_act and has_ts:
        resource_stats = defaultdict(lambda: {
            "work_items": 0,
            "cases": set(),
            "tasks": defaultdict(int),
            "handoffs": defaultdict(int),
            "handoff_set": set(),
            "first_ts": None
        })

        global_cases_seen = set()
        data = data.sort_values(timestamp_col).reset_index(drop=True)

        for i, row in data.iterrows():
            res = row[resource_col]
            cid = row[case_id_col]
            act = row[activity_col]
            ho = row["prev_resource"]
            ts = row[timestamp_col]

            global_cases_seen.add(cid)
            st = resource_stats[res]

            st["work_items"] += 1
            st["cases"].add(cid)
            st["tasks"][act] += 1
            st["handoffs"][ho] += 1
            st["handoff_set"].add(ho)
            st["first_ts"] = st["first_ts"] or ts

            n = st["work_items"]
            duration = (ts - st["first_ts"]).total_seconds() / 60

            data.loc[i, "res_work_items"] = n
            data.loc[i, "res_cases"] = len(st["cases"])
            data.loc[i, "res_unique_tasks"] = len(st["tasks"])
            data.loc[i, "res_unique_handoffs"] = len(st["handoff_set"])
            data.loc[i, "res_ratio_workitems_global"] = n / len(global_cases_seen)
            data.loc[i, "res_ratio_workitems_resource"] = n / len(st["cases"])
            data.loc[i, "res_ratio_task_specific"] = st["tasks"][act] / n
            data.loc[i, "res_ratio_handoff_specific"] = st["handoffs"][ho] / n
            data.loc[i, "res_work_items_per_min"] = n / duration if duration > 0 else 0

    # Detect case attributes
    case_attr_cols = [
        c for c in data.columns
        if c not in [case_id_col, activity_col, resource_col, timestamp_col]
        and data.groupby(case_id_col)[c].nunique().max() == 1
    ]

    # Build output
    output = []
    for cid, group in data.groupby(case_id_col):
        total_time = ((group[timestamp_col].max() - group[timestamp_col].min()).total_seconds() / 60 if has_ts else 0)
        case_attrs = {c: group[c].iloc[0] for c in case_attr_cols}

        output.append({
            case_id_col: cid,
            **case_attrs,
            "ActTimeSeq": (build_event_features(group, timestamp_col, activity_col) if has_act else []),
            "total_time": total_time
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for case in output:
            f.write(json.dumps(case, default=safe_convert) + "\n")

    return output_file
