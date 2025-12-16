### IMPORTS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import random
import argparse

### CONSTANTS
SEED = 42
random.seed(SEED)

### MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_name", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("encoding", type=str)
    args = parser.parse_args()

    log_name = args.log_name
    model = args.model
    encoding = args.encoding

    print(log_name, model, encoding)

if __name__ == "__main__":
    main()