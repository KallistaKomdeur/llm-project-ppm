import pandas as pd
import matplotlib.pyplot as plt
from log_schema import load_log_schema

def descriptives(log_name: str):
    schema = load_log_schema(log_name)
    df = pd.read_csv(f"logs/{log_name}/{log_name}.csv")
    df[schema.timestamp] = pd.to_datetime(df[schema.timestamp])
    case_groups = df.groupby(schema.case_id)[schema.timestamp]
    
    summary = pd.DataFrame({'completion_time': case_groups.max(), 'duration': (case_groups.max() - case_groups.min()).dt.total_seconds() / 60.0})
    summary = summary.sort_values(by='completion_time')
    daily_mean = summary.set_index('completion_time')['duration'].resample('D').mean()
    split_idx = int(len(summary) * 0.8)
    train_set = summary.iloc[:split_idx]
    test_set = summary.iloc[split_idx:]
    t_split = summary.iloc[split_idx - 1]['completion_time']

    def get_stats(data):
        return {
            "Mean": data['duration'].mean(),
            "Median": data['duration'].median(),
            "Std Dev": data['duration'].std(),
            "Max": data['duration'].max(),
            "Count": len(data)
        }

    train_stats = get_stats(train_set)
    test_stats = get_stats(test_set)

    print(pd.DataFrame({"First 80%": train_stats, "Last 20%": test_stats}).to_string())

    plt.figure(figsize=(14, 7))
    plt.plot(train_set['completion_time'], train_set['duration'], alpha=0.3, label='Train', linewidth=0.5)
    plt.plot(test_set['completion_time'], test_set['duration'], alpha=0.3, label='Test', linewidth=0.5)
    plt.plot(daily_mean.index, daily_mean.values, linewidth=2, label='Mean duration per day', zorder=5)
    plt.axvline(x=t_split, linestyle='--', linewidth=1, label=f'80%', zorder=6)

    plt.title(f'{log_name}', fontsize=14)
    plt.xlabel('Completion date', fontsize=12)
    plt.ylabel('Total duration (minutes)', fontsize=12)
    plt.gcf().autofmt_xdate() 
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    descriptives("traffic_fines")