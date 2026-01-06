import json
import random
from pathlib import Path
import pandas as pd
import re
from utils.io_utils import get_input
from utils.prompt_filler import fill_prompt
from utils.send_query import send_query
from utils.inter_case_features import compute_inter_case_features, format_features_for_prompt

BASE_DIR = Path(__file__).resolve().parent

def extract_llm_answer(llm_response: str) -> float:
    """
    Extracts the predicted total time from the LLM response.
    Assumes it's on the line immediately after [[ ## answer ## ]]
    """
    lines = llm_response.splitlines()
    for i, line in enumerate(lines):
        if "[[ ## answer ## ]]" in line:
            answer_line = lines[i + 1].strip()
            return float(answer_line)
    raise ValueError("Could not find [[ ## answer ## ]] in LLM response.")


def compute_actual_total_time_from_csv(csv_path: Path, case_id: str, case_col="case", time_col="timestamp") -> float:
    """
    Computes the actual total duration of a case from the raw CSV.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    trace = df[df[case_col] == int(case_id)].sort_values(time_col)
    if len(trace) < 2:
        return 0.0
    return (trace[time_col].iloc[-1] - trace[time_col].iloc[0]).total_seconds()


def test_llm(log_name: str, provider, model_name, prompt, n_runs: int = 1):
    prompt_name = Path(prompt).stem
    prefix = f"{log_name}_"
    if prompt_name.startswith(prefix):
        prompt_name = prompt_name[len(prefix):]

    RESULTS_DIR = BASE_DIR / "results" / log_name / prompt_name / provider
    log_dir = Path("logs") / log_name
    train_path = log_dir / f"{log_name}_train.json"
    test_path = log_dir / f"{log_name}_test.json"
    raw_csv_path = log_dir / f"{log_name}.csv"
    prompt_template_path = Path("prompts") / log_name / prompt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    pattern = re.compile(rf"{re.escape(log_name)}_results_(\d+)\.json")

    # Name and index results file
    existing_numbers = [int(m.group(1)) for p in RESULTS_DIR.glob(f"{log_name}_results_*.json") if (m := pattern.fullmatch(p.name))]
    next_idx = max(existing_numbers, default=0) + 1
    results_file = RESULTS_DIR / f"{log_name}_results_{next_idx}.json"

    # Load train/test JSONs
    with open(train_path) as f:
        train_traces = json.load(f)
    with open(test_path) as f:
        test_traces = json.load(f)

    # Load prompt template
    with open(prompt_template_path) as f:
        prompt_template = f.read()

    # Compute inter-case features once from raw CSV
    df_raw = pd.read_csv(raw_csv_path)
    features = compute_inter_case_features(df_raw)
    features_str = format_features_for_prompt(features)

    results = []
    maes = []
    mapes = []

    for run_idx in range(n_runs):
        # Pick a random test trace
        case_id = random.choice(list(test_traces.keys()))
        truncated_trace = test_traces[case_id]

        # Build truncated "running" version for prompt
        truncated_trace_prompt = {
            "trace_attributes": truncated_trace["trace_attributes"],
            "events": truncated_trace["events"][:],
            "total_duration": "RUNNING"
        }
        if truncated_trace_prompt["events"]:
            truncated_trace_prompt["events"][-1]["activity"] = "RUNNING"

        # Fill prompt including inter-case features
        prompt_text = fill_prompt(log_name, prompt, features_str, examples_count=5)

        # IF YOU JUST WANT TO READ THE PROMPT, UNCOMMENT THIS!!!
        print(prompt_text)
        import sys
        sys.exit(0)

        # Send to LLM
        response_text = send_query(provider, model_name, prompt_text)

        # Extract predicted time
        try:
            predicted_time = extract_llm_answer(response_text)
        except Exception as e:
            print(f"Error extracting LLM answer for case {case_id}: {e}")
            continue

        # Get actual total time from CSV
        try:
            actual_time = compute_actual_total_time_from_csv(raw_csv_path, case_id)
        except Exception as e:
            print(f"Skipping case {case_id}, error computing actual total_duration: {e}")
            continue

        # Compute metrics
        mae = abs(predicted_time - actual_time)
        maes.append(mae)

        mape = None
        if actual_time != 0:
            mape = abs(predicted_time - actual_time) / actual_time * 100
            mapes.append(mape)

        # Store result
        results.append({
            "case_id": case_id,
            "predicted": predicted_time,
            "actual": actual_time,
            "mae": mae,
            "mape": mape,
            "llm_response": response_text
        })

        print(f"Run {run_idx+1}/{n_runs}: case {case_id}, predicted={predicted_time}, "
              f"actual={actual_time}, mae={mae}, mape={mape}")

    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    overall_mae = sum(maes) / len(maes) if maes else None
    overall_mape = sum(mapes) / len(mapes) if mapes else None
    print(f"Overall MAE over {len(maes)} runs: {overall_mae}")
    print(f"Overall MAPE over {len(mapes)} runs: {overall_mape}")

    return results, overall_mae, overall_mape


if __name__ == "__main__":
    log_name, provider, model_name, encoding, prompt = get_input()
    test_llm(log_name, provider, model_name, prompt, n_runs=1)
