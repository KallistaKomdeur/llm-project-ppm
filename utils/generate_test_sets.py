import json
import random
from pathlib import Path
from utils.general_utils import load_cases
from utils.data_split import temporal_train_test_split_random_truncation

def generate_fixed_sets(log_name, n_sets=100, examples_count=10, clean_first=False, seed=42):
    """
    Pre-generates and saves n_sets fixed (examples, test_case). All modes will load these for identical experiments.
    """
    random.seed(seed)

    root = Path(__file__).resolve().parents[1]
    if clean_first:
        preprocessed_path = root / "logs" / log_name / f"{log_name}_clean_preprocessed.jsonl"
    else:
        preprocessed_path = root / "logs" / log_name / f"{log_name}_preprocessed.jsonl"

    cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split_random_truncation(cases)

    # Sample n_sets unique test cases
    sampled_test_cases = random.sample(test_cases, n_sets)

    sets = []
    for test_case in sampled_test_cases:
        example_cases = random.sample(train_cases, examples_count)
        sets.append({"examples": example_cases, "test_case": test_case})

    out_path = root / "logs" / log_name / f"{log_name}_fixed_sets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sets, f, ensure_ascii=False, indent=2)

    return out_path