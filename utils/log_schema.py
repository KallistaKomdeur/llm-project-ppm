from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "config" / "log_schemas"

class LogSchema:
    def __init__(self, schema_dict: dict):
        self.log_name = schema_dict["log_name"]
        self.case_id = schema_dict["columns"]["case_id"]
        self.activity = schema_dict["columns"]["activity"]
        self.resource = schema_dict["columns"]["resource"]
        self.timestamp = schema_dict["columns"]["timestamp"]

def load_log_schema(log_name: str) -> LogSchema:
    path = SCHEMA_DIR / f"{log_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"No schema found for log '{log_name}' ({path})"
        )

    with open(path, "r", encoding="utf-8") as f:
        schema_dict = yaml.safe_load(f)

    return LogSchema(schema_dict)
