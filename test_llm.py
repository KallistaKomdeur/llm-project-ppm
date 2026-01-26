import json
from pathlib import Path
from datetime import datetime, timezone

from utils.io_utils import get_input
from utils.send_query import send_query
from utils.prompt_filler import fill_prompt
from utils.preprocessing import preprocess_log
from utils.llm_parsing import parse_llm_output
from utils.prompt_splitter import split_prompt_by_markers

BASE_DIR = Path(__file__).resolve().parent

# ======================
# HELPER FUNCTIONS
# ======================
def ensure_preprocessed(log_name: str) -> Path:
    """
    Ensure the preprocessed JSONL exists for the given log. If it does not already exist, the log is preprocessed.
    """
    log_dir = BASE_DIR / "logs" / log_name
    preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    if not preprocessed_path.exists():
        print(f"Preprocessed file missing. Running preprocessing for '{log_name}'")
        preprocess_log(log_name)

    if not preprocessed_path.exists():
        raise RuntimeError("Preprocessing failed: output file not found.")

    return preprocessed_path

def get_results_path(configuration: str, log_name: str, provider: str) -> Path:
    """
    Returns the JSONL file where query logs are appended.
    Each run is stored in a new file: run_1.jsonl, run_2.jsonl, etc.
    """
    results_dir = BASE_DIR / "results" / configuration / log_name / provider
    results_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [
        f for f in results_dir.iterdir()
        if f.is_file() and f.name.startswith("run_") and f.suffix == ".jsonl"]

    run_nums = [
        int(f.stem.split("_")[-1]) for f in existing_files
        if f.stem.split("_")[-1].isdigit()]

    next_run = max(run_nums, default=0) + 1
    return results_dir / f"run_{next_run}.jsonl"

# ======================
# MAIN FUNCTION
# ======================
def test_llm(log_name: str, provider: str, model: str | None, configuration: str, n_runs: int = 1, print_only: bool = True):
    """
    Runs LLM predictions and logs each query.
    """
    # Ensure log is preprocessed
    preprocessed_path = ensure_preprocessed(log_name)
    results_path = get_results_path(configuration, log_name, provider)

    for run_idx in range(n_runs):
        # Build prompt
        prompt_text, true_total_time, prefix_length = fill_prompt(log_name=log_name, configuration=configuration, examples_count=5)

        # Optional splitting
        prompt_parts = (
            split_prompt_by_markers(prompt_text)
            if "split" in configuration
            else [prompt_text]
        )

        # If debugging, only print prompt
        if print_only:
            for i, part in enumerate(prompt_parts, start=1):
                print(f"\n--- PROMPT PART {i}/{len(prompt_parts)} ---\n")
                print(part)
            return

        llm_outputs = []

        for i, part in enumerate(prompt_parts, start=1):
            print(f"Sending part {i}/{len(prompt_parts)}")
            output = send_query(provider, model, part)
            llm_outputs.append(output)
        
        final_output = llm_outputs[-1]
        # Parse response
        reasoning, answer = parse_llm_output(final_output)

        # Build log record
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "configuration": configuration,
            "log_name": log_name,
            "prefix_length": prefix_length,

            "prompt_parts": prompt_parts,
            "llm_raw_output": llm_outputs,

            "llm_reasoning": reasoning,
            "llm_answer": answer,

            "actual_case_duration": true_total_time
        }

        # Append to JSONL
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"Run {run_idx + 1}/{n_runs} | logged"
        )

    print(f"Done, results saved to {results_path}")
    return results_path


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    log_name, provider, model, configuration = get_input()

    test_llm(
        log_name=log_name,
        provider=provider,
        model=model,
        configuration=configuration,
        n_runs=1,
        print_only=True  # set False to actually query, True to just print the prompt
    )
