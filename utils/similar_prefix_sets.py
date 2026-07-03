import random
import time
from collections import defaultdict
from copy import deepcopy

def _edit_distance(seq_a, seq_b):
    """Levenshtein distance between two activity sequences (control-flow distance)."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _extract_prefix_activities(case, prefix_len):
    seq = case.get("ActTimeSeq", [])
    return tuple(e[0] if isinstance(e, (list, tuple)) else str(e) for e in seq[:prefix_len])

def _select_test_cases_randomly(test_cases, n_sets, seed):
    """Sample n_sets test cases uniformly at random with a random prefix length in [2, case_len)."""
    random.seed(seed)

    eligible = [c for c in test_cases if len(c.get("ActTimeSeq", [])) >= 3]
    n_sets = min(n_sets, len(eligible))
    sampled = random.sample(eligible, n_sets)

    result = []
    for test_case_full in sampled:
        seq = test_case_full.get("ActTimeSeq", [])
        case_len = len(seq)
        prefix_len = random.randint(2, case_len - 1)

        truncated_test = deepcopy(test_case_full)
        truncated_test["ActTimeSeq"] = seq[:prefix_len]
        truncated_test["true_total_time"] = test_case_full.get("total_time")
        truncated_test["total_time"] = "RUNNING"
        truncated_test["true_total_length"] = case_len

        result.append({"test_case": truncated_test, "prefix_len": prefix_len, "case_len": case_len})

    return result

def _build_train_variants(train_cases, prefix_len):
    """Group training cases by their prefix control-flow variant, so the edit distance to the test prefix is computed once per variant."""
    variants = defaultdict(list)
    for case in train_cases:
        if len(case.get("ActTimeSeq", [])) < prefix_len:
            continue
        variants[_extract_prefix_activities(case, prefix_len)].append(case)
    return variants

def _retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count):
    """Pick the training cases whose prefix variant has the smallest Levenshtein
    distance to the test prefix, searching the full training set. Ties (either within
    a variant or across equidistant variants) are broken randomly."""
    
    test_activities = _extract_prefix_activities(truncated_test, prefix_len)
    variants = _build_train_variants(train_cases, prefix_len)
    scored_variants = sorted(((_edit_distance(test_activities, variant), variant) for variant in variants), key=lambda x: x[0])

    selected = []
    i = 0

    while i < len(scored_variants) and len(selected) < examples_count:
        dist = scored_variants[i][0]

        tied_variants = []
        while i < len(scored_variants) and scored_variants[i][0] == dist:
            tied_variants.append(scored_variants[i][1])
            i += 1

        tied_cases = [case for variant in tied_variants for case in variants[variant]]
        random.shuffle(tied_cases)

        remaining = examples_count - len(selected)
        selected.extend(tied_cases[:remaining])

    return [deepcopy(c) for c in selected]

def generate_similar_prefix_sets(train_cases, test_cases, n_sets, examples_count, truncate_train, seed):
    """Select (examples, test_case, prefix_length) triples where the training examples
    are the training cases whose control-flow prefix (Levenshtein distance) is most
    similar to the test prefix."""
    
    print(f"Sampling {n_sets} test cases uniformly")
    best_tests = _select_test_cases_randomly(test_cases, n_sets, seed)
    random.seed(seed + 1)

    result_sets = []
    total_start = time.perf_counter()

    for test_record in best_tests:
        prefix_len = test_record["prefix_len"]
        case_len = test_record["case_len"]
        truncated_test = test_record["test_case"]

        start = time.perf_counter()
        similar_examples = _retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count)
        elapsed = time.perf_counter() - start
        print(f"Selected {len(similar_examples)} examples for prefix_len={prefix_len} in {elapsed:.4f}s")

        if truncate_train:
            for ex in similar_examples:
                ex["ActTimeSeq"] = ex.get("ActTimeSeq", [])[:prefix_len]

        result_sets.append({
            "examples": similar_examples,
            "test_case": truncated_test,
            "prefix_length": prefix_len,
            "total_case_length": case_len,
            "mode_truncated_training": truncate_train,
            "selection_time_seconds": elapsed,
        })

    print(f"Total selection time: {time.perf_counter() - total_start:.4f}s")
    return result_sets