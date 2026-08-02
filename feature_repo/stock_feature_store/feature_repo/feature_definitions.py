from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, String

stock = Entity(
    name="stock_name",
    join_keys=["stock_name"],
)

stock_source = FileSource(
    name="stock_source",
    path="data/stock_features_v0.parquet",
    timestamp_field="timestamp",
)

stock_feature_view = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=365),
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
    ],
    source=stock_source,
    online=True,
)
