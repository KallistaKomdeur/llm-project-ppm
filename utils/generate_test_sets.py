import json
import random
from pathlib import Path
from utils.general_utils import load_cases
from utils.data_split import temporal_train_test_split
from utils.load_config import load_config
from utils.similar_prefix_sets import generate_similar_prefix_sets, generate_similar_prefix_temporal_sets

def generate_fixed_sets_random(n_sets, examples_count, seed, train_cases, test_cases, truncate_train):
    """ Randomly enerates n_sets fixed (examples, test_case) """
    random.seed(seed)

    n_sets = min(len(test_cases), n_sets) 
    sampled_test_cases = random.sample(test_cases, n_sets)

    sets = []

    for test_case in sampled_test_cases:
        seq = test_case.get("ActTimeSeq", [])
        case_len = len(seq)

        possible_lengths = list(range(2, case_len))
        if not possible_lengths:
            continue

        prefix_lengths = random.sample(possible_lengths)
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
                "mode_truncated_training": truncate_train,
            })

    return sets


def generate_fixed_sets(log_name, n_sets, examples_count, clean_first, seed):
    """
    - random: random sampling of test cases and training examples.
    - similar_prefix: training examples have the closest prefix to the test case prefix, by control-flow
      (normalized Damerau-Levenshtein) distance alone.
    - similar_prefix_temporal: training examples are, per control-flow variant, the case whose prefix cycle
      time is closest to the test case's, scored by 0.25 * normalized control-flow distance
      + 0.75 * normalized prefix-cycle-time distance.
    """
    config = load_config()
    truncate_train = config.get("truncate_training_examples", False)
    selection_mode = config.get("selection_mode", "random")

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name

    if clean_first:
        preprocessed_path = log_dir / f"{log_name}_clean_preprocessed.jsonl"
    else:
        preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    all_cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split(all_cases)

    timing_summary = None

    if selection_mode == "similar_prefix":
        print(f"Generating fixed sets (similar_prefix)")
        sets, timing_summary = generate_similar_prefix_sets(train_cases=train_cases, test_cases=test_cases, n_sets=n_sets, examples_count=examples_count, truncate_train=truncate_train, seed=seed)
    elif selection_mode == "similar_prefix_temporal":
        print(f"Generating fixed sets (similar_prefix_temporal)")
        sets, timing_summary = generate_similar_prefix_temporal_sets(train_cases=train_cases, test_cases=test_cases, n_sets=n_sets, examples_count=examples_count, truncate_train=truncate_train, seed=seed)
    else:
        if selection_mode != "random":
            print(f"Generating fixed sets, unknown selection_mode, falling back to random")
        print(f"Generating fixed sets (random)")
        sets = generate_fixed_sets_random(n_sets=n_sets, examples_count=examples_count, seed=seed, train_cases=train_cases, test_cases=test_cases, truncate_train=truncate_train)

    output_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sets, f, ensure_ascii=False, indent=2)

    if timing_summary is not None:
        timing_path = log_dir / f"{log_name}_{selection_mode}_timings.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing_summary, f, ensure_ascii=False, indent=2)
        print(f"Saved selection timings to {timing_path}")

    return output_path