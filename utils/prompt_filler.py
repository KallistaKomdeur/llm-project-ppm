import json
import random
from pathlib import Path
import pandas as pd

from utils.features.single_case import format_single_case
from utils.features.inter_case_state import (
    inter_case_state_at,
    format_inter_case_state
)

def fill_prompt(
    log_name: str,
    prompt_file: str,
    configuration: str,
    global_features_text: str | None,
    examples_count: int = 5
):

    logs_dir = Path("logs") / log_name
    prompts_dir = Path("prompts") / log_name

    with open(logs_dir / f"{log_name}_train.json") as f:
        train_traces = json.load(f)

    with open(logs_dir / f"{log_name}_test.json") as f:
        test_traces = json.load(f)

    df_raw = pd.read_csv(logs_dir / f"{log_name}.csv")
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")

    with open(prompts_dir / prompt_file) as f:
        template = f.read()

    def inter_case_states_for_trace(trace, case_id):
        """
        Computes inter-case state at each event timestamp.
        """
        states = []
        timestamps = (
            df_raw[df_raw["case"] == int(case_id)]
            .sort_values("timestamp")["timestamp"]
            .tolist()
        )

        for ts in timestamps:
            state = inter_case_state_at(
                df_raw,
                ts,
                exclude_case=case_id
            )
            states.append(format_inter_case_state(state))

        return states

    # ---- examples ----
    example_blocks = []

    for case_id in random.sample(
        list(train_traces.keys()),
        min(examples_count, len(train_traces))
    ):
        trace = train_traces[case_id]

        inter_case_states = None
        if configuration == "inter-case_only":
            inter_case_states = inter_case_states_for_trace(trace, case_id)

        single = format_single_case(trace, inter_case_states)
        example_blocks.append(single)

    examples_str = "\n\n".join(example_blocks)

    # ---- test case ----
    test_case_id = random.choice(list(test_traces.keys()))
    test_trace = test_traces[test_case_id]

    test_inter_case_states = None
    if configuration == "inter-case_only":
        test_inter_case_states = inter_case_states_for_trace(
            test_trace,
            test_case_id
        )

    test_block = json.dumps(
        {
            f"Case_{test_case_id}": json.loads(
                format_single_case(test_trace, test_inter_case_states)
            )
        },
        indent=2
    )

    filled = (
        template
        .replace("{GLOBAL_FEATURES}", global_features_text or "")
        .replace("{EXAMPLES}", examples_str)
        .replace("{NEW_CASE}", test_block)
    )

    return filled, test_case_id
