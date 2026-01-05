from utils.preprocessing import load_or_preprocess
from utils.io_utils import get_input
from utils.prompt_filler import fill_prompt
from utils.inter_case_features import compute_inter_case_features, format_features_for_prompt
import pandas as pd


if __name__ == "__main__":
    log_name, provider, model_name, encoding, prompt = get_input()
    load_or_preprocess(log_name)
    features = compute_inter_case_features(pd.read_csv(f"logs/{log_name}/{log_name}.csv"))
    features_text = format_features_for_prompt(features)
    prompt_text = fill_prompt(log_name, features_text)
    print(prompt_text)
