import argparse

def get_input() -> tuple:
    parser = argparse.ArgumentParser(description="Run LLM experiment on event log")
    # Required
    parser.add_argument("log_name", type=str, help="Name of the event log")
    parser.add_argument("provider", type=str, help="LLM provider (e.g. gemini, openai)")
    parser.add_argument("configuration", type=str, choices=["single", "global_only", "inter-case_only"], help="Which input configuration to use")
    # Optional
    parser.add_argument("--model", type=str, default = "2.5-flash", help="Model name for the provider (e.g. 2.5-flash)")
    args = parser.parse_args()

    return args.log_name, args.provider, args.model, args.configuration