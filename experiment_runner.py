import yaml
import subprocess
from pathlib import Path

settings_path = Path("config/settings.yaml")

commands = [
    "python -m test_llm bpic2011 gemini single_ref",
    "python -m test_llm bpic2012 gemini single_ref",
    "python -m test_llm bpic2015 gemini single_ref",
    "python -m test_llm bpic2020_domestic_declarations gemini single_ref",
    "python -m test_llm bpic2020_international_declarations gemini single_ref",
    "python -m test_llm bpic2020_prepaid_travel_costs gemini single_ref",
    "python -m test_llm bpic2020_request_for_payment gemini single_ref",
    "python -m test_llm bpic2020_travel_permit_data gemini single_ref",
    "python -m test_llm helpdesk gemini single_ref",
    "python -m test_llm hospital_billing gemini single_ref",
    "python -m test_llm gen_baseline gemini single_ref",
    "python -m test_llm gen_drifted gemini single_ref",
   
]

# Settings combinations
combinations = [{"selection_mode": "similar_prefix"}]

for combo in combinations:
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    settings.update(combo)

    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f)

    for cmd in commands:
        print(f"Executing {cmd}")
        subprocess.run(cmd, shell=True, check=True)
