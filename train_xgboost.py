import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from utils.log_schema import load_log_schema

def train_xgb(log_name: str, logs_dir: str = "logs", results_dir: str = "results"):
    """
    Train XGBoost model to predict total case duration and save metrics as JSON.
    """
    # Load schema and CSV
    schema = load_log_schema(log_name)
    case_col = schema.case_id
    activity_col = schema.activity
    resource_col = schema.resource
    timestamp_col = schema.timestamp
    case_attrs = schema.case_attributes

    csv_path = Path(logs_dir) / log_name / f"{log_name}.csv"
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    # Aggregate features
    case_features = []

    for case_id, group in df.groupby(case_col):
        group = group.sort_values(timestamp_col)
        features = {}

        # Target
        duration = (group[timestamp_col].max() - group[timestamp_col].min()).total_seconds() / 60.0
        features['total_duration'] = duration

        # Simple features
        features['num_events'] = len(group)
        features['num_unique_activities'] = group[activity_col].nunique()
        if resource_col and resource_col in group.columns:
            features['num_unique_resources'] = group[resource_col].nunique()

        # Frequency encoding
        activity_counts = group[activity_col].value_counts(normalize=True)
        features['top_activity_ratio'] = activity_counts.iloc[0] if not activity_counts.empty else 0.0

        # Case attributes
        for attr in case_attrs:
            if attr in group.columns:
                features[attr] = group[attr].iloc[0]

        case_features.append(features)

    feature_df = pd.DataFrame(case_features)

    # Split features and target
    X = feature_df.drop(columns=['total_duration'])
    y = feature_df['total_duration']

    # One-hot encode categorical
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    xgb_model = XGBRegressor(objective='reg:squarederror', tree_method='hist', n_jobs=-1, random_state=42)

    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Predictions & metrics
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": mae,
        "r2": r2,
        "num_train_cases": len(y_train),
        "num_test_cases": len(y_test),
        "best_params": search.best_params_
    }

    # Save metrics JSON
    output_dir = Path(results_dir) / log_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"xgboost_{log_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")
    print(metrics)

    return best_model, metrics

# Terminal entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_xgboost.py <log_name>")
        sys.exit(1)

    log_name = sys.argv[1]
    train_xgb(log_name)
