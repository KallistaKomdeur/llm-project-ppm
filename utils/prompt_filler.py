import json
import random
from pathlib import Path
import pandas as pd

from utils.features.single_case import format_single_case
from utils.features.inter_case_state import (
    inter_case_state_at,
    format_inter_case_state
)
from utils.features.registry import build_feature_block

def fill_prompt(
    log_name: str,
    prompt_file: str,
    configuration: str,              # "single" | "global" | "inter-case_only"
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

    # inter-case context helper
    def inter_case_context(case_id: str):
        cutoff = df_raw[df_raw["case"] == int(case_id)]["timestamp"].max()
        state = inter_case_state_at(df_raw, cutoff, exclude_case=case_id)
        return format_inter_case_state(state)

    # examples
    example_blocks = []

    sampled_cases = random.sample(
        list(train_traces.keys()),
        min(examples_count, len(train_traces))
    )

    for case_id in sampled_cases:
        single = format_single_case(train_traces[case_id])
        inter_case = (
            inter_case_context(case_id)
            if configuration == "inter-case_only"
            else None
        )

        example_blocks.append(
            build_feature_block(
                configuration,
                single_case=single,
                global_features=None,   
                inter_case=inter_case
            )
        )

    examples_str = "\n\n".join(example_blocks)

    # test case 
    test_case_id = random.choice(list(test_traces.keys()))
    test_trace = test_traces[test_case_id]

    single_test = json.dumps({
        f"Case_{test_case_id}": json.loads(format_single_case(test_trace))
    })

    test_inter_case = (
        inter_case_context(test_case_id)
        if configuration == "inter-case_only"
        else None
    )

    test_block = build_feature_block(
        configuration,
        single_case=single_test,
        global_features=None,          
        inter_case=test_inter_case
    )

    # final assembly
    filled = (
        template
        .replace("{GLOBAL_FEATURES}", global_features_text or "")
        .replace("{EXAMPLES}", examples_str)
        .replace("{NEW_CASE}", test_block)
    )

    return filled, test_case_id
