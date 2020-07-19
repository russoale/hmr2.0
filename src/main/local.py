import os
from datetime import datetime

from main.config import Config
from main.model import Model


class LocalConfig(Config):
    ROOT_DATA_DIR = os.path.abspath(os.path.join(__file__, '..', '..', '..'))

    LOG_DIR = os.path.join(ROOT_DATA_DIR, 'logs', datetime.now().strftime("%d%m%Y-%H%M%S"))
    DATA_DIR = os.path.join(ROOT_DATA_DIR, 'src', 'tests', 'files')
    SMPL_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'src', 'tests', 'files')
    SMPL_MODEL_PATH = os.path.join(ROOT_DATA_DIR, 'models', 'neutral_smpl_coco_regressor.pkl')
    SMPL_MEAN_THETA_PATH = os.path.join(ROOT_DATA_DIR, 'models', 'neutral_smpl_mean_params.h5')
    CUSTOM_REGRESSOR_PATH = os.path.join(ROOT_DATA_DIR, 'src', 'tests', 'files', 'regressors')
    CUSTOM_REGRESSOR_IDX = {
        0: 'regressor_test.npy',
    }

    DATASETS = ['dataset']
    SMPL_DATASETS = ['smpl']
    BATCH_SIZE = 2
    JOINT_TYPE = 'cocoplus'
    NUM_KP2D = 19
    NUM_KP3D = 14

    def __init__(self):
        super(LocalConfig, self).__init__()
        self.SEED = 1
        self.NUM_TRAINING_SAMPLES = 1
        self.NUM_TRAIN_SMPL_SAMPLES = 4
        self.NUM_VALIDATION_SAMPLES = 1
        self.NUM_TEST_SAMPLES = 1


if __name__ == '__main__':
    LocalConfig()

    model = Model()
    model.train()
