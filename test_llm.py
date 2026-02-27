import json
from pathlib import Path
from datetime import datetime, timezone

from utils.general_utils import get_input
from utils.send_query import LLMSession
from utils.prompt_filler import fill_prompt
from utils.preprocessing import preprocess_log
from utils.llm_parsing import parse_llm_output
from utils.prompt_splitter import split_prompt_by_markers
from utils.load_config import load_config
from utils.clean_log import clean_log

BASE_DIR = Path(__file__).resolve().parent

def ensure_preprocessed(log_name, clean_first):
    """
    Ensure the preprocessed JSONL exists for the given log. If it does not already exist, the log is preprocessed.
    """
    log_dir = BASE_DIR / "logs" / log_name

    if clean_first:
        # If needs to be cleaned, define paths and clean
        original_csv_path = log_dir / f"{log_name}.csv"
        clean_csv_path = log_dir / f"{log_name}_clean.csv"
        preprocessed_path = log_dir / f"{log_name}_clean_preprocessed.jsonl"

        if not clean_csv_path.exists():
            print(f"Creating clean version of {log_name}")
            clean_log(original_csv_path, clean_csv_path, log_name)

        if not preprocessed_path.exists():
            print(f"Preprocessed clean file missing. Running preprocessing for clean {log_name}.")
            preprocess_log(log_name, clean_version = True)

    else:
        # If doesn't need to be cleaned, define path and preprocessed
        preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

        if not preprocessed_path.exists():
            print(f"Preprocessed file missing. Running preprocessing for {log_name}")
            preprocess_log(log_name, clean_version = False)

    return preprocessed_path

def get_results_path(log_name):
    """
    Returns the JSONL file where query logs are added.
    """
    results_dir = BASE_DIR / "results" / log_name
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [f for f in results_dir.iterdir() if f.is_file() and f.name.startswith("run_") and f.suffix == ".jsonl"]
    run_nums = [int(f.stem.split("_")[-1]) for f in existing_files if f.stem.split("_")[-1].isdigit()]  # Check how many runs have already been
    next_run = max(run_nums, default=0) + 1                                                             # Find number for next run
    return results_dir / f"run_{next_run}.jsonl"

def test_llm(log_name, provider, model, configuration, n_runs, print_only, examples_count):
    """
    Runs LLM predictions and logs each query.
    """
    clean_first = config.get("clean_first", False)

    # Ensure log is preprocessed
    preprocessed_path = ensure_preprocessed(log_name, clean_first)
    results_path = get_results_path(log_name)

    session = LLMSession()      # Create session for caching

    for run_idx in range(n_runs):
        session.reset()     # Clear cache each individual run to prevent information flow!!!
        prompt_text, true_total_time, prefix_length, include_case_attr, include_log_info, inter_case_attr, true_total_length = fill_prompt(log_name=log_name, configuration=configuration, examples_count=examples_count, clean_first = clean_first)

        # Optional splitting
        prompt_parts = (split_prompt_by_markers(prompt_text) if "split" in configuration else [prompt_text])

        # If debugging, only print prompt
        if print_only:
            for i, part in enumerate(prompt_parts, start=1):
                print("Split here\n")
                print(part)
            return

        llm_outputs = []

        for i, part in enumerate(prompt_parts, start=1):
            print(f"Sending part {i}/{len(prompt_parts)}")
            output = session.send_query(provider, model, part)
            llm_outputs.append(output)
        
        final_output = llm_outputs[-1]                  # Get final output
        answer = parse_llm_output(final_output)         # Parse response

        # Build log record for later analysis
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "configuration": configuration,
            "log_name": log_name,
            "llm_answer": answer,
            "actual_case_duration": true_total_time, 

            "prefix_length": prefix_length,
            "true_total_length": true_total_length,
            "case_attributes_included": include_case_attr,
            "log_info_included": include_log_info,
            "included_inter-case_attributes": inter_case_attr,
            "clean_first": clean_first,

            "prompt_parts": prompt_parts,
            "llm_raw_output": llm_outputs
        }

        # Append to results file
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Logged run {run_idx + 1}/{n_runs}")

    return results_path

if __name__ == "__main__":
    log_name, provider, model, configuration = get_input()
    config = load_config()

    test_llm(
        log_name=log_name,
        provider=provider,
        model=model,
        configuration=configuration,
        n_runs= config.get("n_runs", 1),
        print_only= config.get("print_only", True),
        examples_count=config.get("examples_count", 1)
    )
