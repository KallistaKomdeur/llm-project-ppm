from sklearn.model_selection import train_test_split
import pandas as pd

def train_test_split_by_case(df: pd.DataFrame, case_col: str = "case", test_size: float = 0.2, random_state: int = 42):
    """Split a log into train/test sets by case ID."""
    cases = df[case_col].unique()
    train_cases, test_cases = train_test_split(cases, test_size=test_size, random_state=random_state)
    train_df = df[df[case_col].isin(train_cases)].copy()
    test_df = df[df[case_col].isin(test_cases)].copy()
    return train_df, test_df

def truncate_test_traces(df: pd.DataFrame, case_col: str = "case", time_col: str = "timestamp", max_events: int = 5):
    """Truncate each trace to the first max_events events (for test set)."""
    truncated = df.groupby(case_col).head(max_events).reset_index(drop=True)
    return truncated
