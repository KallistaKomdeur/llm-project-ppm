import json
import random
from pathlib import Path

def fill_prompt(log_name: str, examples_count: int = 5) -> str:
    """
    Fills a prompt template for a log with completed train examples and one running test case.

    Args:
        log_name: Name of the log folder (and CSV/JSON files).
        examples_count: Number of completed train traces to include as examples.

    Returns:
        Filled prompt string ready to send to an AI.
    """

    prompts_dir = Path("prompts") / log_name
    template_path = prompts_dir / f"{log_name}_template_paper.txt"

    train_path = Path("logs") / log_name / f"{log_name}_train.json"
    test_path = Path("logs") / log_name / f"{log_name}_test.json"

    # Load JSONs
    with open(train_path) as f:
        train_traces = json.load(f)
    with open(test_path) as f:
        test_traces = json.load(f)

    # Pick random examples
    train_keys = list(train_traces.keys())
    train_sample_keys = random.sample(train_keys, min(examples_count, len(train_keys)))
    example_traces = [train_traces[k] for k in train_sample_keys]

    test_keys = list(test_traces.keys())
    test_sample_key = random.choice(test_keys)
    new_case_trace = test_traces[test_sample_key]

    # Helper to format trace for prompt
    def format_trace(trace):
        trace_dict = {}
        trace_dict.update(trace["trace_attributes"])
        cumulative = 0
        act_seq = []
        for e in trace["events"]:
            duration = e["duration"]
            cumulative += duration if isinstance(duration, (int, float)) else 0
            act_seq.append([e["activity"], int(cumulative)])
        trace_dict["ActTimeSeq"] = act_seq
        trace_dict["total_time"] = str(int(trace["total_duration"])) if isinstance(trace["total_duration"], (int, float)) else trace["total_duration"]
        return json.dumps(trace_dict)

    examples_str = "\n".join([format_trace(t) for t in example_traces])
    new_case_str = json.dumps({
        f"Case_{test_sample_key}": {
            **new_case_trace["trace_attributes"],
            "ActTimeSeq": [[e["activity"], int(e["duration"])] if isinstance(e["duration"], (int, float)) else ["Running"] for e in new_case_trace["events"]],
            "total_time": "RUNNING"
        }
    })

    # Load template and fill in placeholders
    with open(template_path) as f:
        template = f.read()

    prompt_filled = template.replace("{EXAMPLES}", examples_str).replace("{NEW_CASE}", new_case_str)
    return prompt_filled
