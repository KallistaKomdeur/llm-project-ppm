import json
import random
from pathlib import Path
from typing import Dict

from utils.data_split import load_cases, temporal_train_test_split

# ======================
# HELPER FUNCTIONS
# ======================
def format_case(case: Dict, configuration: str) -> Dict:
    """
    Formats a full case (examples always complete).
    """

    # If single variant, only include the activity and time since case start
    if configuration.startswith("single"):
        seq = []
        for act, t, _ in case["ActTimeSeq"]:
            seq.append([act, t])

        return {
            "ActTimeSeq": seq,
            "total_time": case["total_time"]
        }

    # Determines whether inter-case features are included in the prompt
    elif configuration.startswith("inter-case"):
        return case

    else:
        raise ValueError(f"Unknown configuration: {configuration}")

def truncate_case(case: Dict, configuration: str) -> Dict:
    """
    Randomly truncates a case and marks total_time as RUNNING.
    """

    seq = case["ActTimeSeq"]
    if len(seq) < 2:
        raise ValueError("Cannot truncate case with <2 events")

    cut_idx = random.randint(1, len(seq) - 1)
    truncated_seq = seq[:cut_idx]

    if configuration.startswith("single"):
        truncated_seq = [[act, t] for act, t, _ in truncated_seq]

    truncated_case = {
        **case,
        "ActTimeSeq": truncated_seq,
        "total_time": "RUNNING"
    }

    return truncated_case, cut_idx

# ======================
# MAIN FUNCTION
# ======================
def fill_prompt(
    log_name: str,
    configuration: str,
    examples_count: int = 5
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

    # Load & split
    cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split(cases)

    if len(train_cases) < examples_count:
        raise ValueError("Not enough training cases for examples")

    # Sample examples from TRAIN
    example_cases = random.sample(train_cases, examples_count)

    example_blocks = []
    for i, case in enumerate(example_cases, start=1):
        formatted = format_case(case, configuration)
        example_blocks.append(
            json.dumps({f"Example_{i}": formatted}, indent=2)
        )

    examples_str = "\n\n".join(example_blocks)

    # Sample prediction case from TEST
    test_case = random.choice(test_cases)
    true_total_time = test_case["total_time"]

    truncated_case, prefix_length = truncate_case(test_case, configuration)

    test_block = json.dumps(
        {"NEW_CASE": truncated_case},
        indent=2
    )

    # Fill template
    template = prompt_path.read_text(encoding="utf-8")

    filled_prompt = (
        template
        .replace("{EXAMPLES}", examples_str)
        .replace("{NEW_CASE}", test_block)
    )

    return filled_prompt, true_total_time, prefix_length
