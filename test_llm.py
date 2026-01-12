import json
import random
from pathlib import Path
import pandas as pd
import re

from utils.io_utils import get_input
from utils.send_query import send_query
from utils.prompt_filler import fill_prompt
from utils.features.global_features import (compute_global_features, format_global_features)

BASE_DIR = Path(__file__).resolve().parent

# Utilities
def extract_llm_answer(llm_response: str) -> float:
    """
    Extracts the predicted total time from the LLM response.
    Assumes it's on the line immediately after [[ ## answer ## ]]
    """
    lines = llm_response.splitlines()
    for i, line in enumerate(lines):
        if "[[ ## answer ## ]]" in line:
            return float(lines[i + 1].strip())
    raise ValueError("Could not find [[ ## answer ## ]] in LLM response.")

def compute_actual_total_time_from_csv(
    csv_path: Path,
    case_id: str,
    case_col="case",
    time_col="timestamp"
) -> float:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    trace = df[df[case_col] == int(case_id)].sort_values(time_col)
    if len(trace) < 2:
        return 0.0

    return (trace[time_col].iloc[-1] - trace[time_col].iloc[0]).total_seconds()

# Main function
def test_llm(
    log_name: str,
    provider: str,
    model: str,
    configuration: str,      # single | global_only | inter-case_only
    n_runs: int = 1,
    print_only: bool = True
):
    
    prompt_file = f"{log_name}_{configuration}.txt"
    
    RESULTS_DIR = BASE_DIR / "results" / log_name / configuration / provider
    log_dir = Path("logs") / log_name
    raw_csv_path = log_dir / f"{log_name}.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"{re.escape(log_name)}_results_(\d+)\.json")
    existing_numbers = [
        int(m.group(1))
        for p in RESULTS_DIR.glob(f"{log_name}_results_*.json")
        if (m := pattern.fullmatch(p.name))
    ]
    next_idx = max(existing_numbers, default=0) + 1
    results_file = RESULTS_DIR / f"{log_name}_results_{next_idx}.json"

    # Global features (only if needed)
    global_features_text = None
    if configuration in {"global_only", "inter-case_only"}:
        df_raw = pd.read_csv(raw_csv_path)
        global_features_text = format_global_features(
            compute_global_features(df_raw)
        )

    results = []
    maes = []
    mapes = []

    for run_idx in range(n_runs):
        # Build prompt
        prompt_text, case_id = fill_prompt(
            log_name=log_name,
            prompt_file=prompt_file,
            configuration = configuration,
            global_features_text=global_features_text,
            examples_count=5
        )

       # DEBUG: only printing the prompt
        if print_only:
            print(prompt_text)
            return

        # Query LLM
        response_text = send_query(provider, model, prompt_text)

        try:
            predicted_time = extract_llm_answer(response_text)
        except Exception as e:
            print(f"Failed to parse answer for case {case_id}: {e}")
            continue

        try:
            actual_time = compute_actual_total_time_from_csv(
                raw_csv_path, case_id
            )
        except Exception as e:
            print(f"Failed to compute actual time for case {case_id}: {e}")
            continue

        mae = abs(predicted_time - actual_time)
        maes.append(mae)

        mape = None
        if actual_time > 0:
            mape = abs(predicted_time - actual_time) / actual_time * 100
            mapes.append(mape)

        results.append({
            "case_id": case_id,
            "predicted": predicted_time,
            "actual": actual_time,
            "mae": mae,
            "mape": mape,
            "llm_response": response_text
        })

        print(
            f"Run {run_idx+1}/{n_runs} | case={case_id} | "
            f"pred={predicted_time:.1f} | actual={actual_time:.1f} | mae={mae:.1f}"
        )

    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    overall_mae = sum(maes) / len(maes) if maes else None
    overall_mape = sum(mapes) / len(mapes) if mapes else None

    print(f"\nOverall MAE: {overall_mae}")
    print(f"Overall MAPE: {overall_mape}")

    return results, overall_mae, overall_mape

# MAIN
if __name__ == "__main__":
    log_name, provider, model, configuration = get_input()

    test_llm(
        log_name = log_name,
        provider = provider,
        model = model,
        configuration = configuration,
        n_runs = 1,
        print_only = True   # set True to just inspect prompt
    )
