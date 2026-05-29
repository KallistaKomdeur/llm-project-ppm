import random
from copy import deepcopy
import pandas as pd

from utils.log_schema import load_log_schema

from log_distance_measures.config import EventLogIDs
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.control_flow_log_distance import control_flow_log_distance

CTD_BIN_SIZE = pd.Timedelta(hours=1)

def _make_ids():
    return EventLogIDs(case="case_id", activity="activity", start_time="start_time", end_time="end_time", resource="resource")

def _cases_to_dataframe(log_name, cases, ids):
    from datetime import datetime, timezone, timedelta

    schema = load_log_schema(log_name)
    EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
    events = []

    for case in cases:
        case_id = case.get(schema.case_id)
        seq = case.get("ActTimeSeq", [])

        raw_start = case.get("start_ts")
        raw_end = case.get("end_ts")
        total = case.get("total_time", 0) or 0

        if raw_start is not None and raw_end is not None:
            start_ts = datetime.fromtimestamp(float(raw_start), tz=timezone.utc)
            end_ts = datetime.fromtimestamp(float(raw_end), tz=timezone.utc)
        else:
            start_ts = EPOCH
            end_ts = EPOCH + timedelta(seconds=float(total))

        if seq:
            for event in seq:
                activity_name = event[0] if isinstance(event, (list, tuple)) else str(event)
                events.append({ids.case: str(case_id), ids.activity: activity_name, ids.start_time: start_ts, ids.end_time: end_ts})
        else:
            events.append({ids.case: str(case_id), ids.activity: "__unknown__", ids.start_time: start_ts, ids.end_time: end_ts})

    return pd.DataFrame(events)

def _set_distance(log_name, candidate_cases, reference_df, ids):
    candidate_df = _cases_to_dataframe(log_name, candidate_cases, ids)

    ctd = cycle_time_distribution_distance(reference_df, ids, candidate_df, ids, bin_size=CTD_BIN_SIZE)
    cfld = control_flow_log_distance(reference_df, ids, candidate_df, ids)
    return float(ctd) + float(cfld)

# Fast distance for trace comparison
def _edit_distance(seq_a, seq_b):
    """Levenshtein distance on two activity sequences"""
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

def _prefix_similarity_distance(prefix_a, total_time_a, prefix_b, total_time_b):
    max_len = max(len(prefix_a), len(prefix_b), 1)
    edit = _edit_distance(prefix_a, prefix_b) / max_len

    max_time = max(total_time_a, total_time_b, 1)
    time_diff = abs(total_time_a - total_time_b) / max_time

    return edit + time_diff

def _extract_prefix_activities(case, prefix_len):
    seq = case.get("ActTimeSeq", [])
    return [e[0] if isinstance(e, (list, tuple)) else str(e) for e in seq[:prefix_len]]


def _select_test_cases_randomly(test_cases, n_sets, seed):
    """Sample n_sets test cases uniformly at random with a random prefix length in [2, case_len). """
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

# NOT USED ANYMORE
def _select_test_cases_by_distance(log_name, test_cases, n_sets, n_candidates, seed):
    """ Select the test cases most representative of the log"""
    random.seed(seed)

    n_sets = min(n_sets, len(test_cases))
    n_candidates = max(n_candidates, n_sets)

    ids = _make_ids()
    reference_df = _cases_to_dataframe(log_name, test_cases, ids)

    scored_tests: list[tuple[float, dict]] = []

    for i in range(n_candidates):
        test_case_full = random.choice(test_cases)
        seq = test_case_full.get("ActTimeSeq", [])
        case_len = len(seq)

        possible_lengths = list(range(2, case_len))
        if not possible_lengths:
            continue

        prefix_len = random.choice(possible_lengths)

        truncated_test = deepcopy(test_case_full)
        truncated_test["ActTimeSeq"] = seq[:prefix_len]
        truncated_test["true_total_time"] = test_case_full.get("total_time")
        truncated_test["total_time"] = "RUNNING"
        truncated_test["true_total_length"] = case_len

        try:
            dist = _set_distance(log_name, [test_case_full], reference_df, ids)
        except Exception as exc:
            dist = float("inf")

        scored_tests.append((dist, {"test_case": truncated_test, "prefix_len": prefix_len, "case_len": case_len}))

        if (i + 1) % 10 == 0 or (i + 1) == n_candidates:
            best = min(s[0] for s in scored_tests)
            print(f"  [test selection {i+1}/{n_candidates}] best distance: {best:.6f}")

    scored_tests.sort(key=lambda x: x[0])
    return [record for _, record in scored_tests[:n_sets]]

def _retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count, pool_size):
    """ Sample pool_size training cases at random, rank them by prefix similarity to the test prefix, and return the
    examples_count closest untruncated training traces."""
    eligible = [c for c in train_cases if len(c.get("ActTimeSeq", [])) >= 2]
    pool = random.sample(eligible, min(pool_size, len(eligible)))

    # Extract the test prefix once outside the loop
    test_activities = _extract_prefix_activities(truncated_test, prefix_len)
    # Use true_total_time because total_time is overwritten to "RUNNING"
    test_time = float(truncated_test.get("true_total_time") or 0)

    scored: list[tuple[float, dict]] = []
    for train_case in pool:
        train_activities = _extract_prefix_activities(train_case, prefix_len)
        train_time = float(train_case.get("total_time") or 0)

        dist = _prefix_similarity_distance(test_activities, test_time, train_activities, train_time)
        scored.append((dist, train_case))

    scored.sort(key=lambda x: x[0])
    return [deepcopy(c) for _, c in scored[:examples_count]]

def generate_similar_prefix_sets(log_name, all_cases, train_cases, test_cases, n_sets, examples_count, n_candidates, truncate_train, seed, random_test_selection: bool = True):
    """ Select (examples, test_case, prefix_length) triples where the training examples are the training cases whose 
    prefix is most similar to the test prefix """
    pool_size = max(10 * examples_count, 100)

    # Select test cases
    if random_test_selection:
        print(f"Sampling {n_sets} test cases uniformly")
        best_tests = _select_test_cases_randomly(test_cases, n_sets, seed)
        random.seed(seed + 1)
    else:
        print(f"Selecting {n_sets} test cases most representative of log")
        best_tests = _select_test_cases_by_distance(log_name, test_cases, n_sets, n_candidates, seed)
        random.seed(seed + 1)

    # Get similar training examples
    result_sets = []

    for rank, test_record in enumerate(best_tests):
        prefix_len = test_record["prefix_len"]
        case_len = test_record["case_len"]
        truncated_test = test_record["test_case"]

        actual_pool = min(pool_size, len(train_cases))
        similar_examples = _retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count, pool_size)

        if truncate_train:
            for ex in similar_examples:
                ex["ActTimeSeq"] = ex.get("ActTimeSeq", [])[:prefix_len]

        result_sets.append({"examples": similar_examples, "test_case": truncated_test, "prefix_length": prefix_len, "total_case_length": case_len, "mode_truncated_training": truncate_train})

    return result_sets