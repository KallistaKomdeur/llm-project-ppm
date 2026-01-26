from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "config" / "log_schemas"

class LogSchema:
    def __init__(self, schema_dict: dict):
        self.log_name = schema_dict["log_name"]
        cols = schema_dict["columns"]
        self.case_id = cols["case_id"]          # mandatory
        self.activity = cols["activity"]        # mandatory
        self.resource = cols.get("resource")    # optional
        self.timestamp = cols.get("timestamp")  # optional

def load_log_schema(log_name: str) -> LogSchema:
    path = SCHEMA_DIR / f"{log_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"No schema found for log '{log_name}' ({path})"
        )

    with open(path, "r", encoding="utf-8") as f:
        schema_dict = yaml.safe_load(f)

    return LogSchema(schema_dict)
