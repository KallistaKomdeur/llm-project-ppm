from utils.preprocessing import load_or_preprocess
from utils.io_utils import get_input
from utils.prompt_filler import fill_prompt

if __name__ == "__main__":
    log_name, provider, model_name, encoding = get_input()
    load_or_preprocess(log_name)
    prompt_text = fill_prompt(log_name)
    print(prompt_text)
