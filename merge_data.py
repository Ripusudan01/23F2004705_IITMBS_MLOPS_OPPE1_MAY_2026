import pandas as pd
from pathlib import Path

v0_dir = Path("StockAnalyticaData/v0")
v1_dir = Path("StockAnalyticaData/v1")

output_dir = Path("merged_data")
output_dir.mkdir(exist_ok=True)

dfs = []

# Read all CSVs from v0 and v1
for folder in [v0_dir, v1_dir]:
    for csv_file in folder.glob("*.csv"):
        print(f"Processing {csv_file.name}")

        stock = csv_file.stem.split("__")[0]

        df = pd.read_csv(csv_file)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Rolling features using last 10 rows
        df["rolling_avg_10"] = (
            df["close"]
            .rolling(window=10, min_periods=1)
            .mean()
        )

        df["volume_sum_10"] = (
            df["volume"]
            .rolling(window=10, min_periods=1)
            .sum()
        )

        # Target
        df["close_5min_future"] = df["close"].shift(-5)
        df["target"] = (
            df["close_5min_future"] > df["close"]
        ).astype(int)

        df["stock_name"] = stock

        dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

merged_df.to_csv(
    output_dir / "merged_stock_data.csv",
    index=False,
)

print("Merged dataset shape:", merged_df.shape)
print("Saved to:", output_dir / "merged_stock_data.csv")
