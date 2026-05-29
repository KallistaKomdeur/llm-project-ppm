import json
import random
from pathlib import Path
from utils.general_utils import load_cases
from utils.data_split import temporal_train_test_split
from utils.load_config import load_config
from utils.similar_prefix_sets import generate_similar_prefix_sets

def generate_fixed_sets_random(n_sets, examples_count, n_prefixes, seed, train_cases, test_cases, truncate_train):
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
                "mode_truncated_training": truncate_train,
            })

    return sets


def generate_fixed_sets(log_name, n_sets, examples_count, n_prefixes, clean_first, seed):
    """
    - random: random sampling of test cases and training examples.
    - representative: training examples scored against all train_cases
    - similar_prefix: training examples have the closest prefix to the test case prefix
    """
    config = load_config()
    truncate_train = config.get("truncate_training_examples", False)
    # "selection_mode" with values "random" or "representative" or "similar_prefix".
    raw_mode = config.get("selection_mode", config.get("representative_selection", "random"))
    if raw_mode is True:
        selection_mode = "representative"
    elif raw_mode is False:
        selection_mode = "random"
    else:
        selection_mode = str(raw_mode)

    n_candidates = config.get("n_candidates", 200)

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name

    if clean_first:
        preprocessed_path = log_dir / f"{log_name}_clean_preprocessed.jsonl"
    else:
        preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    all_cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split(all_cases)

    if selection_mode == "similar_prefix":
        random_test_selection = config.get("random_test_selection", True)
        print(f"Generating fixed sets, mode=similar_prefix, n_candidates={n_candidates}, random_test_selection={random_test_selection}")
        sets = generate_similar_prefix_sets(log_name=log_name, all_cases=all_cases, train_cases=train_cases, test_cases=test_cases, n_sets=n_sets, examples_count=examples_count, n_candidates=n_candidates, truncate_train=truncate_train, seed=seed, random_test_selection=random_test_selection)
    else:
        if selection_mode != "random":
            print(f"Generating fixed sets, unknown selection_mode={selection_mode!r}, falling back to random")
        print(f"Generating fixed sets, mode=random")
        sets = generate_fixed_sets_random(n_sets=n_sets, examples_count=examples_count, n_prefixes=n_prefixes, seed=seed, train_cases=train_cases, test_cases=test_cases, truncate_train=truncate_train)

    output_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sets, f, ensure_ascii=False, indent=2)

    return output_path