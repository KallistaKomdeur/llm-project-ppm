import json
from pathlib import Path
import random
from typing import Tuple, List, Dict

# ======================
# HELPER FUNCTIONS
# ======================
def load_cases(jsonl_path: Path) -> List[Dict]: 
    """ Loads the desired cases """ 
    cases = [] 
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f: 
            cases.append(json.loads(line)) 
    return cases

def completion_time(case: Dict) -> float:
    """
    Returns the completion time of a case (timestamp of last event)
    """
    try:
        return float(case["ActTimeSeq"][-1][1])
    except (KeyError, IndexError, TypeError):
        raise ValueError("Invalid ActTimeSeq structure in case")

# ======================
# MAIN FUNCTION
# ======================
def temporal_train_test_split(cases: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits cases temporally assuming cases are already sorted by completion time (end_ts).
    
    Train: first train_ratio of cases (earliest completions).
    Test: cases from remaining cases that have at least one event before t_split 
          (end_ts of last train case) and at least one event after t_split.
    
    Requires: cases have 'end_ts', 'start_ts', and 'ActTimeSeq' with format [activity, time_since_start, features].
    """
    if not cases:
        raise ValueError("Empty case list")

    # Assume cases are sorted by completion time (end_ts in ascending order)
    n_train = max(1, int(len(cases) * train_ratio))
    train_cases = cases[:n_train]
    
    # t_split is the end time of the last training case
    if "end_ts" not in train_cases[-1] or train_cases[-1]["end_ts"] is None:
        raise ValueError("Training cases must have 'end_ts' (epoch seconds)")
    t_split = train_cases[-1]["end_ts"]

    # Select test cases: any case that spans t_split (has events before and after)
    test_cases_trunc = []
    for c in cases[n_train:]:
        seq = c.get("ActTimeSeq", [])
        if not seq:
            continue

        # Convert event times (minutes since case start) to absolute epoch seconds
        if "start_ts" not in c or c["start_ts"] is None:
            raise ValueError(f"Case {c.get('CaseId')} missing 'start_ts'")
        times = [c["start_ts"] + float(e[1]) * 60.0 for e in seq]

        has_before = any(t <= t_split for t in times)
        has_after = any(t > t_split for t in times)

        if has_before and has_after:
            # Keep only events up to t_split
            truncated_seq = [e for e, abs_t in zip(seq, times) if abs_t <= t_split]
            truncated_case = dict(c)
            truncated_case["ActTimeSeq"] = truncated_seq
            truncated_case["true_total_time"] = c.get("total_time")
            truncated_case["total_time"] = "RUNNING"
            test_cases_trunc.append(truncated_case)

    return train_cases, test_cases_trunc