import json
import random
from pathlib import Path
from typing import Dict, List

from utils.data_split import load_cases, temporal_train_test_split
from utils.load_config import load_config
from utils.log_schema import load_log_schema

# ======================
# HELPER FUNCTIONS
# ======================
def format_case(case: Dict, configuration: str, include_case_attr: bool, included_inter_case: List[str], case_attr_keys: List[str]) -> Dict:
    """
    Formats a full case.
    """
    # Potentially add case attributes
    result = get_case_attributes(case, include_case_attr, case_attr_keys)
    seq = []

    # If single variant, only include the activity and time since case start
    if configuration.startswith("single"):
        for act, t, *_ in case["ActTimeSeq"]:
            seq.append([act, t])

    # Determines which inter-case features are included in the prompt
    elif configuration.startswith("inter-case"):
        allowed = set(included_inter_case)

        for event in case["ActTimeSeq"]:
            act, t = event[:2]
            ef = event[2] if len(event) == 3 else {}
            seq.append([act, t, {k: v for k, v in ef.items() if k in allowed}])

    result["ActTimeSeq"] = seq
    result["total_time"] = case["total_time"]
    return result

def truncate_case(case: Dict) -> Dict:
    """
    Randomly truncates a case and marks total_time as Running.
    """

    seq = case["ActTimeSeq"]
    if len(seq) < 2:
        raise ValueError("Cannot truncate case with <2 events")

    cut_idx = random.randint(1, len(seq) - 1)

    truncated_case = {
        **case,
        "ActTimeSeq": seq[:cut_idx],
        "total_time": "Running"
    }

    return truncated_case, cut_idx

def get_case_attributes(case: Dict, include_case_attr: bool, case_attribute_keys: List[str]) -> Dict:
    """
    Adds schema-defined case attributes depending on boolean.
    """
    if not include_case_attr:
        return {}

    return {
        k: case[k]
        for k in case_attribute_keys
        if k in case
    }

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
    """

    root = Path(__file__).resolve().parents[1]

    log_dir = root / "logs" / log_name
    prompt_path = root / "prompts" / f"{configuration}.txt"
    preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    if not preprocessed_path.exists():
        raise FileNotFoundError(preprocessed_path)

    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    config = load_config()
    include_case_attr = config.get("include_case_attributes", False)
    included_inter_case = config.get("included_inter_case", [])
    include_log_info = config.get("include_log_info", False)

    # Optionally collect log information
    case_attr_expl = ""
    process_context = ""

    schema = load_log_schema(log_name) 
    case_attrs = schema.case_attributes

    # Case attribute explanations
    if include_log_info:
        
        case_attr_expl = "\n".join(f"- the key \"{k}\", which value is {v}" for k, v in case_attrs.items())

        # Process context
        process_context = schema.log_description or ""

    # Load & split
    cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split(cases)

    if len(train_cases) < examples_count:
        raise ValueError("Not enough training cases for examples")

    # Sample examples from TRAIN
    example_cases = random.sample(train_cases, examples_count)

    example_blocks = []
    case_attr_keys = list(case_attrs)

    for i, case in enumerate(example_cases, start=1):
        formatted = format_case(case, configuration, include_case_attr, included_inter_case, case_attr_keys)
        example_blocks.append(
            json.dumps(formatted, separators=(", ", ": "))
        )

    examples_str = "\n\n".join(example_blocks)

    # Sample prediction case from TEST
    test_case = random.choice(test_cases)
    prefix_length = len(test_case["ActTimeSeq"])
    true_total_time = test_case["true_total_time"]

    formatted_truncated = format_case(test_case, configuration, include_case_attr, included_inter_case, case_attr_keys)

    test_block = json.dumps(formatted_truncated, separators=(", ", ": "))

    # Fill template
    template = prompt_path.read_text(encoding="utf-8")

    filled_prompt = (
        template
        .replace("{EXAMPLES}", examples_str)
        .replace("{NEW_CASE}", test_block)
        .replace("{CASE_ATTRIBUTE_EXPLANATIONS}", case_attr_expl)
        .replace("{PROCESS_CONTEXT}", process_context)
    )

    return filled_prompt, true_total_time, prefix_length, include_case_attr, include_log_info, included_inter_case
