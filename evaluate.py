import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from utils.io_utils import get_input


def load_all_runs(results_dir: Path) -> List[Dict]:
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
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def evaluate(configuration: str, log_name: str, provider: str):
    BASE_DIR = Path(__file__).resolve().parent
    results_dir = BASE_DIR / "results" / configuration / log_name / provider

    records = load_all_runs(results_dir)

    # --- compute MAE overall ---
    y_true = []
    y_pred = []
    for r in records:
        if r.get("llm_answer") is None or r.get("actual_case_duration") is None:
            continue
        y_true.append(float(r["actual_case_duration"]))
        y_pred.append(float(r["llm_answer"]))

    if not y_true:
        raise ValueError("No valid records found.")

    overall_mae = compute_mae(y_true, y_pred)
    print("OVERALL MAE:", overall_mae)

    # --- produce table: one line per record ---
    lines = []
    for r in records:
        run = r.get("run", "unknown")
        prefix_length = r.get("prefix_length", None)
        pred = r.get("llm_answer", None)
        actual = r.get("actual_case_duration", None)

        if pred is None or actual is None:
            continue

        mae = abs(float(pred) - float(actual))
        lines.append({
            "run": run,
            "prefix_length": prefix_length,
            "predicted": float(pred),
            "actual": float(actual),
            "mae": mae
        })

    # print table
    for line in lines:
        print(line)

    return overall_mae

if __name__ == "__main__":
    log_name, provider, model, configuration = get_input()
    evaluate(configuration, log_name, provider)
