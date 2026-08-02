import pandas as pd
from pathlib import Path

input_dir = Path("StockAnalyticaData/v0")
output_dir = Path("feature_repo/stock_feature_store/feature_repo/data")
output_dir.mkdir(parents=True, exist_ok=True)

dfs = []

for csv_file in input_dir.glob("*.csv"):
    stock = csv_file.stem.split("__")[0]

    print(f"Processing {stock}...")

    df = pd.read_csv(csv_file)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # IMPORTANT: Do not assume chronological order
    df = df.sort_values("timestamp")

    # Set timestamp as index for time-based rolling
    df = df.set_index("timestamp")

    # Feature 1: Rolling average of close price over last 10 minutes
    df["rolling_avg_10"] = (
        df["close"]
        .rolling(window="10min", min_periods=1)
        .mean()
    )

    # Feature 2: Rolling sum of volume over last 10 minutes
    df["volume_sum_10"] = (
        df["volume"]
        .rolling(window="10min", min_periods=1)
        .sum()
    )

    # Reset index so timestamp becomes a column again
    df = df.reset_index()

    # Create target
    df["close_5min_future"] = df["close"].shift(-5)
    df["target"] = (df["close_5min_future"] > df["close"]).astype(int)

    # Add stock name
    df["stock_name"] = stock

    # Drop rows where future price is unavailable
    df = df.dropna(subset=["close_5min_future"])

    dfs.append(df)

# Merge all stocks
feature_df = pd.concat(dfs, ignore_index=True)

# Save parquet
output_file = output_dir / "stock_features_v0.parquet"
feature_df.to_parquet(output_file, index=False)

print(f"\nSaved features to: {output_file}")
print(feature_df.head())
print(feature_df.columns.tolist())
