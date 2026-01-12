import json
import pandas as pd
from typing import Dict, Any

def single_case_features(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the single-case feature block used in prompts.
    """
    cumulative = 0
    act_seq = []

    for e in trace["events"]:
        duration = e["duration"]
        if isinstance(duration, (int, float)):
            cumulative += duration
            act_seq.append([e["activity"], int(cumulative)])
        else:
            act_seq.append(["RUNNING"])

    return {
        "ActTimeSeq": act_seq,
        "total_time": (
            str(int(trace["total_duration"]))
            if isinstance(trace["total_duration"], (int, float))
            else trace["total_duration"]
        )
    }


def format_single_case(trace: Dict[str, Any]) -> str:
    """
    Formats single-case features as JSON for prompt inclusion.
    """
    out = {}
    out.update(trace["trace_attributes"])
    out.update(single_case_features(trace))
    return json.dumps(out)
