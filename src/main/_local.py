from datetime import datetime

import os

from main.config import Config
from main.model import Model


class LocalConfig(Config):
    ROOT_DATA_DIR = os.path.abspath('../../')

    LOG_DIR = os.path.join(ROOT_DIR, 'logs', datetime.now().strftime("%d%m%Y-%H%M%S"))
    DATA_DIR = os.path.join(ROOT_DIR, 'src', 'tests', 'files')
    SMPL_DATA_DIR = os.path.join(ROOT_DIR, 'src', 'tests', 'files')
    SMPL_MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'neutral_smpl_coco_regressor.pkl')
    SMPL_MEAN_THETA_PATH = os.path.join(ROOT_DIR, 'models', 'neutral_smpl_mean_params.h5')

    DATASETS = ['dataset']
    SMPL_DATASETS = ['smpl']
    BATCH_SIZE = 2
    SEED = 1


if __name__ == '__main__':
    LocalConfig()

    model = Model()
    model.train()
