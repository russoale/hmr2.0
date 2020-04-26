# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

import json
import os
from datetime import datetime


class Config(object):
    __instance = None

    def __new__(cls):
        if Config.__instance is None:
            Config.__instance = object.__new__(cls)
            Config.__instance.__initialized = False

        return Config.__instance

    # ------Directory settings:------
    #
    # root directory
    ROOT_DATA_DIR = os.path.join('/', 'data', 'ssd1', 'russales')

    # path to save training models to
    LOG_DIR = os.path.join(ROOT_DATA_DIR, 'logs', datetime.now().strftime("%d%m%Y-%H%M%S"))

    # path to specific checkpoint to be restored
    # if LOG_DIR is set to specific training and RESTORE_PATH is None
    # per default last saved checkpoint will be restored
    # subclass config to override, see example in evaluate.ipynb
    RESTORE_PATH = None

    # path to saved dataset in tf record format
    # folder names should be same as defined in DATASET config (see below):
    # e.g. DATASETS = ['coco', 'mpii_3d', 'h36m']
    DATA_DIR = os.path.join(ROOT_DATA_DIR, 'new_records')

    # path to saved smpl data in tf record format
    # folder names should be same as defined in SMPL_DATASETS config (see below):
    # e.g. SMPL_DATASETS = ['cmu', 'joint_lim']
    SMPL_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'new_records', 'smpl')

    # path to the neutral smpl model
    SMPL_MODEL_PATH = os.path.join(ROOT_DATA_DIR, 'models', 'neutral_smpl_coco_regressor.pkl')

    # path to mean theta h5 file
    SMPL_MEAN_THETA_PATH = os.path.join(ROOT_DATA_DIR, 'models', 'neutral_smpl_mean_params.h5')

    # ------HMR parameters:------
    #
    # input image size to the encoder network after preprocess
    ENCODER_INPUT_SHAPE = (224, 224, 3)

    # number of iterations for regressor feedback loop
    ITERATIONS = 3

    # cocoplus (19 keypoints) or lsp 14 keypoints, returned by SMPL
    JOINT_TYPE = 'cocoplus'

    # cocoplus (19 keypoints) or lsp 14 keypoints, returned by SMPL
    NUM_KP2D = 19 if JOINT_TYPE == 'cocoplus' else 14
    NUM_KP3D = 14

    # number of epochs to train
    EPOCHS = 55

    # effective batch size
    BATCH_SIZE = 64

    # number of parallel read tf record files
    NUM_PARALLEL = 16

    # seed for random shuffle
    SEED = 42

    # list of datasets to use for training
    # DATASETS = ['lsp', 'lsp_ext', 'mpii', 'coco', 'mpii_3d', 'h36m']  # paired setting, except h36m mosh not available
    DATASETS = ['lsp', 'lsp_ext', 'coco', 'mpii']  # unpaired setting

    # datasets to use for adversarial prior training
    SMPL_DATASETS = ['cmu', 'joint_lim']  # , 'h36m']

    # if set to True, no adversarial prior is trained = monsters
    ENCODER_ONLY = False

    # set True to use 3D labels
    USE_3D = False

    # ------Hyper parameters:------
    #
    # generator learning rate
    GENERATOR_LEARNING_RATE = 1e-5
    # generator weight decay
    GENERATOR_WEIGHT_DECAY = 1e-4
    # weight on generator 2d loss
    GENERATOR_2D_LOSS_WEIGHT = 60.
    # weight on generator 3d loss
    GENERATOR_3D_LOSS_WEIGHT = 60.

    # adversarial prior learning rate
    DISCRIMINATOR_LEARNING_RATE = 1e-4
    # adversarial prior weight decay
    DISCRIMINATOR_WEIGHT_DECAY = 1e-4
    # weight on discriminator
    DISCRIMINATOR_LOSS_WEIGHT = 1

    # ------Data augmentation:------
    #
    # value to jitter translation
    TRANS_MAX = 20

    # max value of scale jitter
    SCALE_MAX = 1.23

    # min value of scale jitter
    SCALE_MIN = 0.8

    # ------SMPL settings:------
    #
    # number of smpl joints
    NUM_JOINTS = 23
    NUM_JOINTS_GLOBAL = NUM_JOINTS + 1

    # number of cameras parameters [scale, tx, ty]
    NUM_CAMERA_PARAMS = 3

    # The pose (theta) is modeled by relative 3D rotation
    # of K joints in axis-angle representation (rotation matrix)
    # K joints + 1 (global rotation)
    # override this according to smpl representation
    NUM_POSE_PARAMS = NUM_JOINTS_GLOBAL * 3

    # number of shape (beta) parameters
    # override this according to smpl representation
    NUM_SHAPE_PARAMS = 10

    # total number of smpl params
    NUM_SMPL_PARAMS = NUM_CAMERA_PARAMS + NUM_POSE_PARAMS + NUM_SHAPE_PARAMS

    # total number of vertices
    NUM_VERTICES = 6890

    def __init__(self):
        if self.__initialized:
            return

        self.__initialized = True

        # number of training samples, use `test_count_all_samples` in test_dataset.py to determine
        self.NUM_TRAINING_SAMPLES = self.count_samples_of(self.DATASETS, 'train')

        # number of smpl training samples
        self.NUM_TRAIN_SMPL_SAMPLES = self.count_samples_of(self.SMPL_DATASETS, 'train')

        # number of validation samples, use `test_count_all_samples` in test_dataset.py to determine
        # For validation coco val set for 2d and for 3d subject 9 from h36m was used
        # (all sequences but only first camera perspective)
        self.NUM_VALIDATION_SAMPLES = self.count_samples_of(self.DATASETS, 'val')

        # number of test samples, use `test_count_all_samples` in test_dataset.py to determine
        self.NUM_TEST_SAMPLES = self.count_samples_of(self.DATASETS, 'test')

    @staticmethod
    def count_samples_of(datasets, split):
        """Numbers need to be provided after tf record generation (see `inspect.ipynb`)
        Args:
            datasets: list of dataset names used for training
            split: train|val, define which split to use
        Returns:
            total_number: int, total number of training samples
        """
        train_samples_per_dataset = {
            'lsp': 999,
            'lsp_ext': 9984,
            'mpii': 17537,
            'coco': 116601,
            'mpii_3d': 166982,
            'h36m': 113231,

            # SMPL/MOSH:
            'cmu': 3934266,
            'joint_lim': 181967,
        }

        val_samples_per_dataset = {
            'lsp': 999,
            'coco': 4802,
            'h36m': 6120,
        }

        test_samples_per_dataset = {
            'mpii_3d': 1955,
            'h36m': 40414,
        }

        if split == 'train':
            samples = train_samples_per_dataset
        elif split == 'val':
            samples = val_samples_per_dataset
        elif split == 'test':
            samples = test_samples_per_dataset
        else:
            raise Exception('unknown split')

        return sum([samples[d] for d in datasets if d in samples.keys()])

    def save_config(self):
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        print('Saving logs to {}'.format(self.LOG_DIR))

        config_path = os.path.join(self.LOG_DIR, "config.json")
        config_dict = dict([(a, getattr(self, a)) for a in dir(self)
                            if not a.startswith("_") and not callable(getattr(self, a))])
        with open(config_path, 'w') as fp:
            json.dump(config_dict, fp, indent=4, sort_keys=True)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
