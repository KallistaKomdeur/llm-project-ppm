import json
from pathlib import Path
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
    Computes t_split such that train_ratio of cases are completed by t_split
    """
    if not cases:
        raise ValueError("Empty case list")

    # Compute completion times
    for case in cases:
        case["_completion_time"] = completion_time(case)

    # Sort cases by completion time
    sorted_cases = sorted(cases, key=lambda x: x["_completion_time"])

    # Compute t_split (80% of cases completed)
    n_train = max(1, int(len(sorted_cases) * train_ratio))  # at least 1
    t_split = sorted_cases[n_train - 1]["_completion_time"]

    # Training = cases completed by t_split
    train_cases = [c for c in sorted_cases if c["_completion_time"] <= t_split]

    # Test = remaining cases
    test_cases_full = [c for c in sorted_cases if c["_completion_time"] > t_split]

    # Truncate test cases
    test_cases_trunc = []
    for c in test_cases_full:
        truncated_seq = [e for e in c["ActTimeSeq"] if e[1] <= t_split]
        if truncated_seq:  # skip empty truncated cases
            truncated_case = dict(c)
            truncated_case["ActTimeSeq"] = truncated_seq
            truncated_case["total_time"] = "RUNNING"
            test_cases_trunc.append(truncated_case)

    # Clean up helper field
    for c in cases:
        del c["_completion_time"]

    return train_cases, test_cases_trunc
