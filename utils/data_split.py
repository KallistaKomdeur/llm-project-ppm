import json
from pathlib import Path
from typing import Tuple, List, Dict

# ======================
# HELPER FUNCTIONS
# ======================
def load_cases(jsonl_path: Path) -> List[Dict]:
    """
    Loads the desired cases
    """
    cases = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases

# ======================
# MAIN FUNCTION
# ======================
def first_event_completion_time(case: Dict) -> float:
    """
    Returns the completion time of the first event of a case (basis for temporal ordering)
    This is ActTimeSeq[0][1].
    """
    try:
        return float(case["ActTimeSeq"][0][1])
    except (KeyError, IndexError, TypeError):
        raise ValueError("Invalid ActTimeSeq structure in case")

def temporal_train_test_split(cases: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits cases temporally based on first event completion time, with 80% train 20% test. 
    """
    sorted_cases = sorted(cases, key=first_event_completion_time)
    split_idx = int(len(sorted_cases) * train_ratio)

    train_cases = sorted_cases[:split_idx]
    test_cases = sorted_cases[split_idx:]

    if not train_cases or not test_cases:
        raise ValueError("Temporal split failed: empty train or test set")

    return train_cases, test_cases
