import os
from util import imgs
from util import transform

## Global dataset constants

RANDOM_STATE = os.environ.get('MIDS_207_RANDOM_STATE', 1000)
RANDOM_STATE = int(RANDOM_STATE)

DATA_PATH = os.environ.get('MIDS_207_DATA_PATH', './data')

# max images per class
MAX_PER_CLASS = os.environ.get('MIDS_207_MAX_PER_CLASS', 500)
MAX_PER_CLASS = int(MAX_PER_CLASS)

DATA_TRAIN_PCT = .6
DATA_VAL_PCT = .2
DATA_TEST_PCT = .2

## Training dataset constants
TRAIN_AUG_PCT = .8
TRAIN_BATCH_SIZE = 100
TRAIN_NUM_BATCHES = 100

VAL_AUG_PCT = .8
VAL_BATCH_SIZE = 32
VAL_NUM_BATCHES = 20




print(f'loading data from {DATA_PATH}...')
data = imgs.PathLoader(DATA_PATH, max_per_class=MAX_PER_CLASS)

print(f'splitting data into train, val, test...')
X_train, y_train, X_val, y_val, X_test, y_test = data.split_train_val_test(
    DATA_TRAIN_PCT,
    DATA_VAL_PCT,
    DATA_TEST_PCT
)
