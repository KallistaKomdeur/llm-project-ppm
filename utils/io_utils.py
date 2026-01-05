import argparse

def get_input() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_name", type=str)
    parser.add_argument("provider", type=str)
    parser.add_argument("--model_name", type=str, default = "2.5-flash")
    parser.add_argument("--encoding", type=str, default = "seq")
    parser.add_argument("--prompt", type=str, default = "bpic2020_template_paper.txt")
    args = parser.parse_args()

    log_name = args.log_name
    provider = args.provider
    model_name = args.model_name
    encoding = args.encoding
    prompt = args.prompt

    return log_name, provider, model_name, encoding, prompt