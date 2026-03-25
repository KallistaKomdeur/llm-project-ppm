import yaml
import subprocess
from pathlib import Path

settings_path = Path("config/settings.yaml")

commands = [
    "python -m test_llm traffic_fines gemini single_ref",
    "python -m test_llm traffic_fines gemini single_split_ref",
    "python -m test_llm traffic_fines gemini single_reasoning",
    "python -m test_llm traffic_fines gemini inter-case_ref",
    "python -m test_llm traffic_fines gemini inter-case_split_ref",
    "python -m test_llm traffic_fines gemini inter-case_self_select_split",
    "python -m test_llm traffic_fines gemini inter-case_reasoning",
    "python -m test_llm traffic_fines gemini inter-case_explanations",
]

# Settings combinations
combinations = [{"truncate_training_examples": True}, {"truncate_training_examples": False}]

for combo in combinations:
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    settings.update(combo)

    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f)

    for cmd in commands:
        print(f"Executing {cmd}")
        subprocess.run(cmd, shell=True, check=True)
