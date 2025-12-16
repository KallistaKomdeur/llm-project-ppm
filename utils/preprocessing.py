import pandas as pd
from pathlib import Path

def preprocess(log_raw: Path) -> pd.DataFrame:
    df = pd.read_csv(log_raw)
    return df