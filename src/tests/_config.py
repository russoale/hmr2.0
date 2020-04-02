import os

from main.config import Config


class TestConfig(Config):
    BATCH_SIZE = 2
    DATA_DIR = os.path.abspath('files')
    SMPL_DATA_DIR = os.path.abspath('files')
    DATASETS = ['dataset']
    SMPL_DATASETS = ['smpl']
    SEED = 1
