### IMPORTS
import random
import argparse
from pathlib import Path
from utils.send_query import send_query

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
    parser.add_argument("model_name", type=str)
    parser.add_argument("encoding", type=str)
    args = parser.parse_args()

    log_name = args.log_name
    provider = args.provider
    model_name = args.model_name
    encoding = args.encoding

    return log_name, provider, model_name, encoding

### MAIN
def main():
    log_name, provider, model_name, encoding = get_input()
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log = log_dir / f"{log_name}.csv"
    if not log.exists():
        raise FileNotFoundError(f"Log not found: {log}")
    
    prompt = "Hi! Can you tell me who you are?"
    
    response = send_query(provider, model_name, prompt)
    print(response)

if __name__ == "__main__":
    main()