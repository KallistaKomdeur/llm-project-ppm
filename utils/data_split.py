def temporal_train_test_split(cases, train_ratio = 0.8):
    """
    Splits cases temporally assuming cases are already sorted by completion time (see paper)
    Train: first train_ratio of cases (= earliest completions).
    Test: cases from remaining cases that have at least one event before t_split and at least one event after t_split.
    """
    if not cases:
        raise ValueError("Empty case list")

    # Assume cases are sorted by completion time
    n_train = max(1, int(len(cases) * train_ratio))
    train_cases = cases[:n_train]
    
    # t_split is the end time of the last training case
    if "end_ts" not in train_cases[-1] or train_cases[-1]["end_ts"] is None:
        raise ValueError("Training cases must have end timestamp")
    t_split = train_cases[-1]["end_ts"]

    # Select test cases
    test_cases_trunc = []
    for c in cases[n_train:]:
        seq = c.get("ActTimeSeq", [])
        times = [c["start_ts"] + float(e[1]) * 60.0 for e in seq]

        has_before = any(t <= t_split for t in times)
        has_after = any(t > t_split for t in times)

        # Test cases must have events before and after t-split
        if has_before and has_after:
            # Keep only events up to t_split
            truncated_seq = [e for e, abs_t in zip(seq, times) if abs_t <= t_split]
            truncated_case = dict(c)
            truncated_case["ActTimeSeq"] = truncated_seq
            truncated_case["true_total_time"] = c.get("total_time")
            truncated_case["total_time"] = "RUNNING"
            truncated_case["true_total_length"] = len(c.get("ActTimeSeq", []))
            test_cases_trunc.append(truncated_case)

    return train_cases, test_cases_trunc

def temporal_train_test_split(cases, train_ratio = 0.8, MIN_PREFIX_LENGTH = 1):
    """
    Splits cases temporally assuming cases are already sorted by completion time (see paper)
    Train: first train_ratio of cases (= earliest completions).
    Test: remaining 20% of cases, untruncated.
    """
    if not cases:
        raise ValueError("Empty case list")
    
    # Assume cases are sorted by completion time
    n_train = max(1, int(len(cases) * train_ratio))
    
    return cases[:n_train], cases[n_train:]

