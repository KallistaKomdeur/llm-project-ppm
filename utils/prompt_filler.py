import json
import random
from pathlib import Path
from typing import Dict, List

from utils.data_split import load_cases, temporal_train_test_split

# ======================
# HELPER FUNCTIONS
# ======================
def format_case(case: Dict, configuration: str, include_case_attr: bool, included_inter_case: List[str]) -> Dict:
    """
    Formats a full case (examples always complete).
    """
    # Potentially add case attributes
    result = {}
    result.update(select_case_attributes(case, include_case_attr))
    seq = []

    # If single variant, only include the activity and time since case start
    if configuration.startswith("single"):
        for event in case["ActTimeSeq"]:
            act, t = event[:2]
            seq.append([act, t])

    # Determines which inter-case features are included in the prompt
    elif configuration.startswith("inter-case"):
        allowed = set(included_inter_case)

        for event in case["ActTimeSeq"]:
            if len(event) == 3:
                act, t, ef = event
            else:
                act, t = event
                ef = {}

            filtered_ef = {k: v for k, v in ef.items() if k in allowed}
            seq.append([act, t, filtered_ef])

    result["ActTimeSeq"] = seq
    result["total_time"] = case["total_time"]
    return result

def truncate_case(case: Dict) -> Dict:
    """
    Randomly truncates a case and marks total_time as RUNNING.
    """

    seq = case["ActTimeSeq"]
    if len(seq) < 2:
        raise ValueError("Cannot truncate case with <2 events")

    cut_idx = random.randint(1, len(seq) - 1)

    truncated_case = {
        **case,
        "ActTimeSeq": seq[:cut_idx],
        "total_time": "RUNNING"
    }

    return truncated_case, cut_idx

def select_case_attributes(case:Dict, include_case_attr:bool) -> Dict:
    if include_case_attr:
        return {
            k: v for k, v in case.items()
            if k not in {"ActTimeSeq", "total_time"}
        }
    else:
        return {}

# ======================
# MAIN FUNCTION
# ======================
def fill_prompt(
    log_name: str,
    configuration: str,
    examples_count: int
):
    """
    Fills the prompt template with values. 
    - Examples come only from temporal training set
    - Prediction case comes only from temporal test set
    """

    root = Path(__file__).resolve().parents[1]

    log_dir = root / "logs" / log_name
    prompt_path = root / "prompts" / f"{configuration}.txt"
    preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    if not preprocessed_path.exists():
        raise FileNotFoundError(preprocessed_path)

    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    # Change this to select which features to include
    include_case_attr = True
    
    included_inter_case = [
        "timesincemidnight",
        "weekday",
        "month",
        "timesincelastevent",
        "timesincecasestart",
        "event_nr",
        "prev_resource",
        "ent_act",
        "ent_case",
        "ent_handoff",
        "busyness",
        "open_cases",
        "res_work_items",
        "res_cases",
        "res_unique_tasks",
        "res_unique_handoffs",
        "res_ratio_workitems_global",
        "res_ratio_workitems_resource",
        "res_ratio_task_specific",
        "res_ratio_handoff_specific",
        "res_work_items_per_min",
        #"act_freq",
        #"handoff_freq",
        "total_time"
    ]

    # Load & split
    cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split(cases)

    if len(train_cases) < examples_count:
        raise ValueError("Not enough training cases for examples")

    # Sample examples from TRAIN
    example_cases = random.sample(train_cases, examples_count)

    example_blocks = []
    for i, case in enumerate(example_cases, start=1):
        formatted = format_case(case, configuration, include_case_attr, included_inter_case)
        example_blocks.append(
            json.dumps({f"Example_{i}": formatted}, indent=2)
        )

    examples_str = "\n\n".join(example_blocks)

    # Sample prediction case from TEST
    test_case = random.choice(test_cases)
    true_total_time = test_case["total_time"]

    truncated_case, prefix_length = truncate_case(test_case)
    formatted_truncated = format_case(truncated_case, configuration, include_case_attr, included_inter_case)

    test_block = json.dumps(
        {"NEW_CASE": formatted_truncated},
        indent=2
    )

    # Fill template
    template = prompt_path.read_text(encoding="utf-8")

    filled_prompt = (
        template
        .replace("{EXAMPLES}", examples_str)
        .replace("{NEW_CASE}", test_block)
    )

    return filled_prompt, true_total_time, prefix_length, include_case_attr, included_inter_case
