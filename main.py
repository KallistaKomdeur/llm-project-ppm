### IMPORTS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import random
import argparse
from pathlib import Path

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
    parser.add_argument("model_type", type=str)
    parser.add_argument("encoding", type=str)
    args = parser.parse_args()

    log_name = args.log_name
    provider = args.provider
    model_type = args.model_type
    encoding = args.encoding

    return log_name, provider, model_type, encoding

def get_api_key(provider):
    load_dotenv() 
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    api_key = GEMINI_API_KEY

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Check your .env file and path.")

### MAIN
def main():
    # Configure parameters
    log_name, provider, model_type, encoding = get_input()
    log = BASE_DIR / "logs" / f"{log_name}.csv"
    if not log.exists():
        raise FileNotFoundError(f"Log not found: {log}")
    
    # Configure API key
    api_key = get_api_key(provider)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content("Say hello in a language of your choice.")
    
if __name__ == "__main__":
    main()