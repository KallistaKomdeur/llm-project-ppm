### IMPORTS
from pathlib import Path
from utils.send_query import send_query
from utils.preprocessing import preprocess
from utils.io_utils import get_input
import pandas as pd
import json
from typing import Dict, Any

### HELPER FUNCTIONS
def load_or_preprocess(log_name: str, processed_dir: str = "logs/processed", raw_dir: str = "logs/raw") -> Dict[str, Any]:
    """
    Loads preprocessed JSON if exists, otherwise reads raw CSV and preprocesses it.
    Automatically saves the JSON after preprocessing.
    """

    processed_path = Path(processed_dir) / f"{log_name}_processed.json"
    raw_path = Path(raw_dir) / f"{log_name}.csv"

    # Check whether either the processed path or raw path exist
    if processed_path.exists():
        print(f"Loading preprocessed log from {processed_path}")
        with open(processed_path, "r") as f:
            return json.load(f)
    elif raw_path.exists():
        print(f"Processing raw log from {raw_path}")
        df = pd.read_csv(raw_path)
        traces = preprocess(df)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(processed_path, "w") as f:
            json.dump(traces, f, indent=2)
        print(f"Saved preprocessed log to {processed_path}")
        return traces
    else:
        raise FileNotFoundError(f"Neither processed nor raw log found for {log_name}")

if __name__ == "__main__":
    log_name, provider, model_name, encoding = get_input()
    result = load_or_preprocess(log_name)
    print(result)
    #prompt = "Hi! Can you tell me who you are?"
    #response = send_query(provider, model_name, prompt)
    #print(response)