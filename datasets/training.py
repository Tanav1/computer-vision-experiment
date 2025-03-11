import numpy as np

from util import transform

from . import (
    X_train,
    y_train,
    TRAIN_AUG_PCT,
    TRAIN_BATCH_SIZE,
    TRAIN_NUM_BATCHES,
)

TRANSFORM_SETS = []

AFFINE_TRANSFORM = [
    (
        np.array([
            [0, 0],
            [1, 0],
            [1, 1]
        ]),
        np.array([
            [0, .1],
            [.8, 0],
            [1, .8]
        ])
    ),
    (
        np.array([
            [0, 0],
            [1, 0],
            [1, 1]
        ]),
        np.array([
            [.1, 0],
            [.9, .1],
            [.8, 1]
        ])
    ),
]
affine_transform_ts = transform.TransformSet(transform.affine_transform)
for args in AFFINE_TRANSFORM:
    affine_transform_ts.append_args_kwargs(*args)
TRANSFORM_SETS.append(affine_transform_ts)

SCALE_AFFINE = [
    (1.05,),
    (1.1,),

    (.95,),
    (.9,),
]
scale_transform_ts = transform.TransformSet(transform.scale_affine)
for args in SCALE_AFFINE:
    scale_transform_ts.append_args_kwargs(*args)
TRANSFORM_SETS.append(scale_transform_ts)


OFFSET_SHIFT = [
    (
        (.2, 0),
    ),
    (
        (0, .1),
    ),
    (
        (0, .2),
    ),

    (
        (-.2, 0),
    ),
    (
        (0, -.1),
    ),
    (
        (0, -.2),
    ),
]
offset_shift_ts = transform.TransformSet(transform.offset_shift)
for args in OFFSET_SHIFT:
    offset_shift_ts.append_args_kwargs(*args)
TRANSFORM_SETS.append(offset_shift_ts)

ADJUST_CONTRAST = [
    (10.,),
    (30.,),
    (-10.,),
    (-30.,),
]
adjust_contrast_ts = transform.TransformSet(transform.adjust_contrast)
for args in ADJUST_CONTRAST:
    adjust_contrast_ts.append_args_kwargs(*args)
TRANSFORM_SETS.append(adjust_contrast_ts)

ADJUST_BRIGHTNESS = [
    (.4,),
    (-.4,)
]
adjust_brightness_ts = transform.TransformSet(transform.adjust_brightness)
for args in ADJUST_BRIGHTNESS:
    adjust_brightness_ts.append_args_kwargs(*args)
TRANSFORM_SETS.append(adjust_brightness_ts)


dst = transform.DatasetTransform(X_train, y_train)

## Pass this into Model.fit(x=training_data), this contains both X(data) and y(labels) values.
data = transform.BatchIter(TRAIN_BATCH_SIZE, TRAIN_NUM_BATCHES, dst,
                           transform_sets=TRANSFORM_SETS,
                           pct_transform=TRAIN_AUG_PCT)

## Create a baseline dataset with the stock, non-augmented data.
baseline_dst = transform.DatasetTransform(X_train, y_train)
baseline_data = transform.BatchIter(TRAIN_BATCH_SIZE, TRAIN_NUM_BATCHES,
                                    baseline_dst,
                                    pct_transform=0)
