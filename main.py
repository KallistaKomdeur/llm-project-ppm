### IMPORTS
import random
import argparse
from pathlib import Path
from utils.send_query import send_query
from utils.preprocessing import preprocess
import pandas as pd

### CONSTANTS
SEED = 42
BASE_DIR = Path(__file__).resolve().parent

### PRE-CONFIGURATION
random.seed(SEED) 

### HELPER FUNCTIONS
def get_input():
    parser = argparse.ArgumentParser()
    # TODO Add default value
    parser.add_argument("log_name", type=str)
    parser.add_argument("provider", type=str)
    parser.add_argument("--model_name", type=str, default = "2.5-flash")
    parser.add_argument("--encoding", type=str, default = "seq")
    args = parser.parse_args()

    log_name = args.log_name
    provider = args.provider
    model_name = args.model_name
    encoding = args.encoding

    return log_name, provider, model_name, encoding

### MAIN
def main():
    log_name, provider, model_name, encoding = get_input()

    log_raw_dir = BASE_DIR / "logs/raw"
    log_processed_dir = BASE_DIR / "logs/processed"
    log_raw_dir.mkdir(parents=True, exist_ok=True)
    log_processed_dir.mkdir(parents=True, exist_ok=True)

    log_raw = log_raw_dir / f"{log_name}.csv"
    log_processed = log_processed_dir / f"{log_name}_preprocessed.csv"

    if not log_raw.exists():
        raise FileNotFoundError(f"Raw log not found: {log_name}")

    if log_processed.exists():
        print(f"Loading preprocessed log: {log_processed}")
        df = pd.read_csv(str(log_processed))
    else:
        print(f"Preprocessed log not found, preprocessing {log_name}.")
        df = preprocess(log_raw)
        df.to_csv(log_processed, index=False)
        print(f"Saved preprocessed log: {log_processed}")
    
    prompt = "Hi! Can you tell me who you are?"
    
    #response = send_query(provider, model_name, prompt)
    #print(response)

if __name__ == "__main__":
    main()