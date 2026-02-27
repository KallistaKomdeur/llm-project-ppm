import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent

def load_results(log_name):
    """Load results across individual files/runs"""
    results_dir = BASE_DIR / "results" / log_name
    if not results_dir.exists():
        print(f"Results directory not found")
        return pd.DataFrame()

    all_records = []
    files = list(results_dir.glob("run_*.jsonl"))
    
    # Aggregate results across files
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return pd.DataFrame(all_records)

def evaluate(log_name):
    """ 
    Evaluate all results for a log. MAE and R^2 grouped by:
    - Provider (Gemini, Claude, etc.)
    - Model
    - Configuration (single/inter-case, split or not)
    - Are case attributes and/or log info included?
    - Was the log cleaned first
    
    """
    df = load_results(log_name)
    if df.empty:
        print(f"No results found for {log_name}.")
        return

    # Convert to numeric and drop invalid rows
    df['llm_answer'] = pd.to_numeric(df['llm_answer'], errors='coerce')
    df['actual_case_duration'] = pd.to_numeric(df['actual_case_duration'], errors='coerce')
    df = df.dropna(subset=['llm_answer', 'actual_case_duration'])

    # Grouping keys
    group_cols = ['provider', 'model','configuration', 'case_attributes_included', 'log_info_included', 'clean_first']
    grouped = df.groupby(group_cols)

    summary_results = []
    
    # Compute metrics for all groups
    for names, group in grouped:
        y_true = group['actual_case_duration']
        y_pred = group['llm_answer']
        
        # Create group identifier dictionary
        group_meta = dict(zip(group_cols, names))
        
        # Calculate metrics
        metrics = {
            "parameters": group_meta,
            "n_samples": int(len(group)),
            "actual_avg": float(y_true.mean()),
            "predicted_avg": float(y_pred.mean()),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(group) > 1 else None
        }
        summary_results.append(metrics)

    final_output = {"log_name": log_name, "evaluation_groups": summary_results}

    # Save to the results directory
    output_path = BASE_DIR / "results" / log_name / "evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance and output results as file")
    parser.add_argument("log_name", type=str, help="The folder name within results/")
    args = parser.parse_args()
    evaluate(args.log_name)