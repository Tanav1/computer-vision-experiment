from util import transform

from .training import (
    TRANSFORM_SETS
)

from . import (
    X_val,
    y_val,
    VAL_AUG_PCT,
    VAL_BATCH_SIZE,
    VAL_NUM_BATCHES,
)

dst = transform.DatasetTransform(X_val, y_val)

## Pass this into Model.fit(x=training_data), this contains both X(data) and y(labels) values.
data = transform.BatchIter(VAL_BATCH_SIZE, VAL_NUM_BATCHES, dst,
                           transform_sets=TRANSFORM_SETS,
                           pct_transform=VAL_AUG_PCT)


baseline_dst = transform.DatasetTransform(X_val, y_val)

## Pass this into Model.fit(x=training_data), this contains both X(data) and y(labels) values.
baseline_data = transform.BatchIter(VAL_BATCH_SIZE, VAL_NUM_BATCHES,
                                    baseline_dst,
                                    pct_transform=0)
