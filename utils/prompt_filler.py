import json
import random
from pathlib import Path
import pandas as pd

from utils.local_inter_case_features import (
    compute_running_activities,
    format_running_activities_for_prompt
)


def fill_prompt(log_name: str, features_text: str, examples_count: int = 5) -> str:
    """
    Fills a prompt template for a log with completed train examples
    and one running test case, including local inter-case context
    for BOTH examples and the test case.
    """

    prompts_dir = Path("prompts") / log_name
    template_path = prompts_dir / f"{log_name}_template_paper.txt"

    logs_dir = Path("logs") / log_name
    train_path = logs_dir / f"{log_name}_train.json"
    test_path = logs_dir / f"{log_name}_test.json"
    raw_csv_path = logs_dir / f"{log_name}.csv"

    # Load JSONs
    with open(train_path) as f:
        train_traces = json.load(f)
    with open(test_path) as f:
        test_traces = json.load(f)

    # Load raw CSV ONCE
    df_raw = pd.read_csv(raw_csv_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")

    def compute_local_context(case_id, max_listed=5):
        """
        Compute local inter-case context at the cutoff time
        of the given case.
        """
        case_events = df_raw[df_raw["case"] == int(case_id)]
        cutoff_time = case_events["timestamp"].max()

        running = compute_running_activities(
            df_raw,
            current_time=cutoff_time
        )

        return format_running_activities_for_prompt(
            running,
            max_listed=max_listed
        )

    def format_trace(trace, total_time_override=None):
        """
        Formats a trace into the JSON-like structure used in the prompt.
        """
        trace_dict = {}
        trace_dict.update(trace["trace_attributes"])

        cumulative = 0
        act_seq = []
        for e in trace["events"]:
            duration = e["duration"]
            if isinstance(duration, (int, float)):
                cumulative += duration
                act_seq.append([e["activity"], int(cumulative)])
            else:
                act_seq.append(["RUNNING"])

        trace_dict["ActTimeSeq"] = act_seq

        if total_time_override is not None:
            trace_dict["total_time"] = total_time_override
        else:
            trace_dict["total_time"] = (
                str(int(trace["total_duration"]))
                if isinstance(trace["total_duration"], (int, float))
                else trace["total_duration"]
            )

        return json.dumps(trace_dict)

    train_keys = list(train_traces.keys())
    train_sample_keys = random.sample(
        train_keys,
        min(examples_count, len(train_keys))
    )

    example_blocks = []
    for case_id in train_sample_keys:
        local_context = compute_local_context(case_id)
        trace_str = format_trace(train_traces[case_id])

        example_blocks.append(
            local_context + "\n" + trace_str
        )

    examples_str = "\n\n".join(example_blocks)

    test_keys = list(test_traces.keys())
    test_case_id = random.choice(test_keys)
    test_trace = test_traces[test_case_id]

    test_local_context = compute_local_context(test_case_id)

    test_case_str = json.dumps({
        f"Case_{test_case_id}": json.loads(
            format_trace(test_trace, total_time_override="RUNNING")
        )
    })

    with open(template_path) as f:
        template = f.read()

    prompt_filled = (
        template
        .replace("{INTER_CASE_FEATURES}", features_text)
        .replace("{EXAMPLES}", examples_str)
        .replace("{LOCAL_INTER_CASE_CONTEXT}", test_local_context)
        .replace("{NEW_CASE}", test_case_str)
    )

    return prompt_filled, test_case_id
