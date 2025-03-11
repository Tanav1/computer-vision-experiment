import numpy as np

from . import (
    X_test,
    y_test,
)

from util.transform import (
    center_rotation,
    BatchIter,
    TransformSet,
    DatasetTransform
)

NUM_STEPS = 90
NUM_BATCHES = 350
BATCH_SIZE = int(X_test.shape[0] / NUM_BATCHES)

START = 0
END = -90

args_list = np.linspace(START, END, NUM_STEPS)


datasets = []

for arg in args_list:
    ds = DatasetTransform(X_test, y_test)
    ts = TransformSet(center_rotation)
    ts.append_args_kwargs(arg.copy())
    batch = BatchIter(BATCH_SIZE, NUM_BATCHES, ds, pct_transform=1.0, transform_sets=[ts])
    datasets.append(batch)
