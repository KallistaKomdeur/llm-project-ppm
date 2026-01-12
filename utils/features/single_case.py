import json
from typing import Dict, Any, List, Optional

def single_case_features(
    trace: Dict[str, Any],
    inter_case_states: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Builds the single-case feature block used in prompts.
    Optionally embeds inter-case state per event (only for inter-case_only).
    """
    cumulative = 0
    act_seq = []

    events = trace["events"]

    for idx, e in enumerate(events):
        duration = e["duration"]

        if isinstance(duration, (int, float)):
            cumulative += duration
            entry = {
                "activity": e["activity"],
                "time": int(cumulative)
            }
        else:
            entry = {
                "activity": "RUNNING",
                "time": "RUNNING"
            }

        if inter_case_states is not None:
            entry["inter_case"] = inter_case_states[idx]

        act_seq.append(entry)

    return {
        "ActTimeSeq": act_seq,
        "total_time": (
            int(trace["total_duration"])
            if isinstance(trace["total_duration"], (int, float))
            else trace["total_duration"]
        )
    }


def format_single_case(
    trace: Dict[str, Any],
    inter_case_states: Optional[List[str]] = None
) -> str:
    """
    Formats single-case features as JSON for prompt inclusion.
    """
    out = {}
    out.update(trace["trace_attributes"])
    out.update(single_case_features(trace, inter_case_states))
    return json.dumps(out, indent=2)
