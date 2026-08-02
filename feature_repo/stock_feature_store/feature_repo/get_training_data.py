import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

df = pd.read_parquet("data/stock_features_v0.parquet")

# Use a sample instead of the full dataset
df = df.head(1000)

entity_df = df[
    ["stock_name", "timestamp"]
].rename(columns={"timestamp": "event_timestamp"})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
    ],
).to_df()

print(training_df.head())

training_df.to_csv("training_dataset_sample.csv", index=False)

print(training_df.shape)
