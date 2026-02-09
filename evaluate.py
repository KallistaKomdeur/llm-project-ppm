import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from utils.io_utils import get_input

# ======================
# HELPER FUNCTIONS
# ======================
import re
from typing import Optional

def extract_prediction_from_raw(llm_raw_output) -> Optional[float]:
    """
    Extract predicted lead time from raw LLM output.
    Supports:
    - [[ 12345 ]]
    - [[ ## answer ## ]]\\n12345
    """

    if not llm_raw_output:
        return None

    # llm_raw_output is a LIST of strings → join safely
    if isinstance(llm_raw_output, list):
        text = "\n".join(llm_raw_output)
    else:
        text = str(llm_raw_output)

    # 1) Prefer numbers inside [[ ... ]]
    bracket_matches = re.findall(r"\[\[\s*([0-9]+)\s*\]\]", text)
    if bracket_matches:
        return float(bracket_matches[0])

    # 2) Fallback: first standalone integer
    number_matches = re.findall(r"\b([0-9]+)\b", text)
    if number_matches:
        return float(number_matches[0])

    return None

def load_all_runs(results_dir: Path) -> List[Dict]:
    """
    Get all runs in this folder
    """
    runs = sorted(results_dir.glob("run_*.jsonl"))
    records = []
    for file in runs:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    return records

def compute_mae(y_true: List[float], y_pred: List[float]) -> float:
    """
    Computes and returns mean absolute erorr (MAE) in minutes
    """
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def compute_mape(y_true: List[float], y_pred: List[float]) -> float:
    """
    Computes Mean Absolute Percentage Error (MAPE)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # avoid division by zero
    non_zero = y_true != 0
    if not np.any(non_zero):
        raise ValueError("All true values are zero; MAPE undefined.")

    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])))

# ======================
# MAIN FUNCTION
# ======================
def evaluate(configuration: str, log_name: str, provider: str):
    BASE_DIR = Path(__file__).resolve().parent
    results_dir = BASE_DIR / "results" / configuration / log_name / provider
    records = load_all_runs(results_dir)

    y_true = []
    y_pred = []

    for r in records:
        actual = r.get("actual_case_duration")
        pred = extract_prediction_from_raw(r.get("llm_raw_output"))

        if actual is None or pred is None:
            continue

        y_true.append(float(actual))
        y_pred.append(float(pred))

    if not y_true:
        raise ValueError("No valid records found.")

    mae = compute_mae(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    mean_actual = float(np.mean(y_true))

    summary = {
        "mae": mae,
        "mape": mape,
        "mean_actual_case_duration": mean_actual,
        "n_traces": len(y_true)
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    log_name, provider, model, configuration = get_input()
    evaluate(configuration, log_name, provider)
