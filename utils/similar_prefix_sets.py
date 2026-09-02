import random
import time
from collections import defaultdict
from copy import deepcopy

import pandas as pd

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _edit_distance(seq_a, seq_b):
    """Damerau-Levenshtein distance (OSA) between two activity sequences. 
    Insertion, deletion, substitution, and transposition each cost 1, see paper."""
    m, n = len(seq_a), len(seq_b)
    two_back = list(range(n + 1))   # row i-2
    one_back = list(range(n + 1))   # row i-1
    current = [0] * (n + 1)         # row i 

    for i in range(1, m + 1):
        current[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            current[j] = min(
                one_back[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                one_back[j - 1] + cost, # substitution
            )
            if (i > 1 and j > 1 and seq_a[i - 1] == seq_b[j - 2] and seq_a[i - 2] == seq_b[j - 1]):
                current[j] = min(current[j], two_back[j - 2] + 1)  # transposition
        two_back, one_back, current = one_back, current, two_back

    return one_back[n]

def _normalized_edit_distance(seq_a, seq_b):
    """Raw DL distance divided by the length of the longer of the two sequences."""
    dist = _edit_distance(seq_a, seq_b)
    denom = max(len(seq_a), len(seq_b), 1)
    return dist / denom

def _extract_prefix_activities(case, prefix_len):
    "Gets the variant of the prefix."
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

# ---------------------------------------------------------------------------
# similar_prefix (control-flow only)
# ---------------------------------------------------------------------------

def _build_train_variants(train_cases, prefix_len):
    """Group training cases by their prefix control-flow variant, so distance to the test prefix is computed once per variant."""
    variants = defaultdict(list)
    for case in train_cases:
        if len(case.get("ActTimeSeq", [])) < prefix_len:
            continue
        variants[_extract_prefix_activities(case, prefix_len)].append(case)
    return variants

def _retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count):
    """Pick the training cases with the smallest normalized DL distance to the test prefix, ties are broken randomly."""

    test_activities = _extract_prefix_activities(truncated_test, prefix_len)
    variants = _build_train_variants(train_cases, prefix_len)
    scored_variants = sorted(((_normalized_edit_distance(test_activities, variant), variant) for variant in variants), key=lambda x: x[0])

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

# ---------------------------------------------------------------------------
# similar_prefix_temporal
# ---------------------------------------------------------------------------
def _prefix_cycle_time_seconds(case, prefix_len):
    """Gets timestamp of the final event in the prefix"""
    seq = case.get("ActTimeSeq", [])[:prefix_len]

    if len(seq) < 1:
        return 0.0

    last_entry = seq[-1]

    return pd.to_numeric(last_entry[1])

def _normalized_cycle_time_distance(cycle_time_a, cycle_time_b):
    """Absolute difference between two prefix cycle times, normalized by the larger of the two so the
    result is on the same [0, 1]-ish scale as the normalized DL distance."""
    denom = max(abs(cycle_time_a), abs(cycle_time_b), 1.0)
    return abs(cycle_time_a - cycle_time_b) / denom

def _select_variant_representative(cases, prefix_len, test_cycle_time):
    """Among training cases sharing a control-flow variant, pick the one whose prefix cycle time is
    closest to the test sample"""
    return min(cases, key=lambda c: abs(_prefix_cycle_time_seconds(c, prefix_len) - test_cycle_time))

def _retrieve_similar_train_cases_temporal(truncated_test, prefix_len, train_cases, examples_count):
    """ Pick training cases using x * normalized control-flow distance + y * normalized prefix-cycle-time distance"""
    test_activities = _extract_prefix_activities(truncated_test, prefix_len)
    test_cycle_time = _prefix_cycle_time_seconds(truncated_test, prefix_len)
    variants = _build_train_variants(train_cases, prefix_len)
    scored = []

    for variant, cases in variants.items():
        # Control-flow distance is the same for every case in this variant
        cf_dist = _normalized_edit_distance(test_activities, variant)

        # Score every case individually
        for case in cases:
            case_cycle_time = _prefix_cycle_time_seconds(case, prefix_len)

            ct_dist = _normalized_cycle_time_distance(test_cycle_time,case_cycle_time)
            score = (0.5 * cf_dist + 0.5 * ct_dist)
            scored.append((score, case))

    # Lowest combined distance = most similar.
    scored.sort(key=lambda x: x[0])

    # Randomly break ties
    selected = []
    i = 0
    
    while i < len(scored) and len(selected) < examples_count:
        dist = scored[i][0]

        tied_cases = []

        while i < len(scored) and scored[i][0] == dist:
            tied_cases.append(scored[i][1])
            i += 1

        random.shuffle(tied_cases)
        remaining = examples_count - len(selected)
        selected.extend(tied_cases[:remaining])

    return [deepcopy(c) for c in selected]

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def _generate_sets(train_cases, test_cases, n_sets, examples_count, truncate_train, seed, retrieve_fn, mode_label):
    best_tests = _select_test_cases_randomly(test_cases, n_sets, seed)
    random.seed(seed + 1)

    result_sets = []
    per_test_timings = []
    total_start = time.perf_counter()

    for test_record in best_tests:
        prefix_len = test_record["prefix_len"]
        case_len = test_record["case_len"]
        truncated_test = test_record["test_case"]

        start = time.perf_counter()
        similar_examples = retrieve_fn(truncated_test, prefix_len, train_cases, examples_count)
        elapsed = time.perf_counter() - start
        per_test_timings.append({"prefix_length": prefix_len, "total_case_length": case_len, "selection_time_seconds": elapsed})

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

    total_elapsed = time.perf_counter() - total_start
    timing_summary = {
        "mode": mode_label,
        "n_sets": len(result_sets),
        "total_selection_time_seconds": total_elapsed,
        "average_selection_time_seconds": (total_elapsed / len(result_sets)) if result_sets else 0.0,
        "per_test_case": per_test_timings,
    }

    return result_sets, timing_summary

def generate_similar_prefix_sets(train_cases, test_cases, n_sets, examples_count, truncate_train, seed):
    """Select (examples, test_case, prefix_length) triples where the training examples are the training
    cases whose control-flow prefix is most similar to the test prefix, using normalized DL distance. """
    return _generate_sets(
        train_cases, test_cases, n_sets, examples_count, truncate_train, seed,
        retrieve_fn=_retrieve_similar_train_cases,
        mode_label="similar_prefix"
    )

def generate_similar_prefix_temporal_sets(train_cases, test_cases, n_sets, examples_count, truncate_train, seed):
    """Select (examples, test_case, prefix_length) triples where the training examples are the training
    cases whose control-flow variant representative is most similar to the test prefix under
    x * normalized control-flow distance + y * normalized prefix-cycle-time distance."""
    return _generate_sets(
        train_cases, test_cases, n_sets, examples_count, truncate_train, seed,
        retrieve_fn=_retrieve_similar_train_cases_temporal,
        mode_label="similar_prefix_temporal",
    )