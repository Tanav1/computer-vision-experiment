import numpy as np

from . import (
    X_test,
    y_test,
)

from util.transform import (
    affine_transform,
    BatchIter,
    TransformSet,
    DatasetTransform
)

NUM_STEPS = 75
NUM_BATCHES = 350
BATCH_SIZE = int(X_test.shape[0] / NUM_BATCHES)

start_end = np.array([
    [
        np.array([
            [0, 0], # top left
            [0, 1], # bottom left
            [1, 1]  # bottom right
        ]),
        np.zeros((3, 2))
    ]
])

affine_transforms = np.repeat(start_end, NUM_STEPS, axis=0)

end_points = np.linspace(
    np.array([
        [.01, .01],
        [0.01, .99],
        [1.01, 1.01]
    ]),
    np.array([
        [.15, .25],
        [0.25, .7],
        [.3, .8]]),
    NUM_STEPS
)
affine_transforms[:, 1, :] = end_points


datasets = []

for args in affine_transforms:
    ds = DatasetTransform(X_test, y_test)
    ts = TransformSet(affine_transform)
    ts.append_args_kwargs(*(args.copy()))
    batch = BatchIter(BATCH_SIZE, NUM_BATCHES, ds, pct_transform=1.0, transform_sets=[ts])
    datasets.append(batch)
