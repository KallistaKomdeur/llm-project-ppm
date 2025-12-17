from pathlib import Path
import pandas as pd
import json
from utils.preprocessing import preprocess
from utils.split import train_test_split_by_case, truncate_test_traces
from utils.io_utils import get_input
from utils.prompt_filler import fill_prompt

def load_or_preprocess(log_name: str, max_test_events: int = 5) -> dict:
    log_dir = Path("logs") / log_name
    raw_path = log_dir / f"{log_name}.csv"
    train_path = log_dir / f"{log_name}_train.json"
    test_path = log_dir / f"{log_name}_test.json"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Load existing JSONs
    if train_path.exists() and test_path.exists():
        print(f"Loading preprocessed train/test logs from {log_dir}")
        with open(train_path) as f:
            train_traces = json.load(f)
        with open(test_path) as f:
            test_traces = json.load(f)
        return {"train": train_traces, "test": test_traces}

    # Load raw CSV
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw log not found at {raw_path}")
    print(f"Processing raw log {raw_path}")
    df = pd.read_csv(raw_path)

    # Split train/test by case
    train_df, test_df = train_test_split_by_case(df)

    # Preprocess
    train_traces = preprocess(train_df, truncate=False)
    test_traces = preprocess(test_df, truncate=True, max_events=max_test_events)

    # Save JSONs
    with open(train_path, "w") as f:
        json.dump(train_traces, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_traces, f, indent=2)

    print(f"Saved preprocessed train/test logs to {log_dir}")
    return {"train": train_traces, "test": test_traces}

if __name__ == "__main__":
    log_name, provider, model_name, encoding = get_input()
    load_or_preprocess(log_name)
    prompt_text = fill_prompt(log_name)
    print(prompt_text)
