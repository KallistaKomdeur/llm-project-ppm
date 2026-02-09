import argparse

# ======================
# MAIN FUNCTION
# ======================
def get_input() -> tuple:
    """
    Parser to get variable values from input command. Some choices are fixed (see README). 
    """
    parser = argparse.ArgumentParser(description="Run LLM experiment on event log")
    # Required
    parser.add_argument("log_name", type=str, help="Name of the event log")
    parser.add_argument("provider", type=str, choices=["gemini", "openai", "anthropic"], help="LLM provider")
    parser.add_argument("configuration", type=str, choices=["single_1", "inter-case_1", "single_split_1", "inter-case_split_1"], help="Which input configuration to use")
    # Optional
    parser.add_argument("--model", type=str, default = "2.5-flash", help="Model name for the provider (e.g. 2.5-flash)")
    args = parser.parse_args()

    return args.log_name, args.provider, args.model, args.configuration