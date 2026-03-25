import json
import random
from pathlib import Path
from utils.general_utils import load_cases
from utils.data_split import temporal_train_test_split
from utils.load_config import load_config

def generate_fixed_sets(log_name, n_sets, examples_count, n_prefixes, clean_first, seed):
    """
    Pre-generates and saves n_sets fixed (examples, test_case). All modes will load these for identical experiments.
    """
    random.seed(seed)
    config = load_config()
    truncate_train = config.get("truncate_training_examples", False)

    root = Path(__file__).resolve().parents[1]
    if clean_first:
        preprocessed_path = root / "logs" / log_name / f"{log_name}_clean_preprocessed.jsonl"
    else:
        preprocessed_path = root / "logs" / log_name / f"{log_name}_preprocessed.jsonl"

    cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split(cases)  # both untruncated now
    
    # Sample n_sets unique test cases
    n_sets = min(len(test_cases), n_sets)   # don't sample more than available
    sampled_test_cases = random.sample(test_cases, n_sets)

    sets = []

    for test_case in sampled_test_cases:
        seq = test_case.get("ActTimeSeq", [])
        case_len = len(seq)

        possible_lengths = list(range(2, case_len))
        if not possible_lengths:
            continue

        num_to_sample = min(len(possible_lengths), n_prefixes)
        prefix_lengths = sorted(random.sample(possible_lengths, num_to_sample))
        base_examples = random.sample(train_cases, examples_count)

        for prefix_len in prefix_lengths:
            truncated_test = dict(test_case)
            truncated_test["ActTimeSeq"] = seq[:prefix_len]
            truncated_test["true_total_time"] = test_case.get("total_time")
            truncated_test["total_time"] = "RUNNING"
            truncated_test["true_total_length"] = case_len

            final_examples = []
            for example in base_examples:
                if truncate_train:
                    truncated_example = dict(example)
                    truncated_example["ActTimeSeq"] = example.get("ActTimeSeq", [])[:prefix_len]
                    final_examples.append(truncated_example)
                else:
                    final_examples.append(example)
            
            sets.append({
                "examples": final_examples,
                "test_case": truncated_test,
                "prefix_length": prefix_len,
                "total_case_length": case_len,
                "mode_truncated_training": truncate_train})
    
    output_path = root / "logs" / log_name / f"{log_name}_fixed_sets.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sets, f, ensure_ascii=False, indent=2)
    
    return output_path