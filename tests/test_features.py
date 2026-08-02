import pandas as pd


DATA_PATH = "feature_repo/stock_feature_store/feature_repo/data/stock_features_v0.parquet"


def test_features_exist():
    df = pd.read_parquet(DATA_PATH)

    assert "rolling_avg_10" in df.columns
    assert "volume_sum_10" in df.columns
    assert "target" in df.columns


def test_feature_values():
    df = pd.read_parquet(DATA_PATH)

    # Rolling average should never be negative
    assert (df["rolling_avg_10"].dropna() >= 0).all()

    # Volume sum should never be negative
    assert (df["volume_sum_10"].dropna() >= 0).all()
