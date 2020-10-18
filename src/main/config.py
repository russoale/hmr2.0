# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

import json
from datetime import datetime

import os


class Config(object):
    __instance = None

    def __new__(cls):
        if Config.__instance is None:
            Config.__instance = object.__new__(cls)
            Config.__instance.__initialized = False

        return Config.__instance

    # ------Directory settings:------
    #
    # root project directory
    ROOT_PROJECT_DIR = os.path.abspath(os.path.join(__file__, '..', '..', '..'))

    # root data directory
    ROOT_DATA_DIR = os.path.join('/', 'data', 'ssd1', 'russales')

    # path to save training models to
    LOG_DIR = os.path.join(ROOT_DATA_DIR, 'logs', datetime.now().strftime("%d%m%Y-%H%M%S"))

    # path to specific checkpoint to be restored
    # if LOG_DIR is set to specific training and RESTORE_PATH is None
    # per default last saved checkpoint will be restored
    # subclass config to override, see example in evaluate.ipynb
    RESTORE_PATH = None
    RESTORE_EPOCH = None

    # path to saved dataset in tf record format
    # folder names should be same as defined in DATASET config (see below):
    # e.g. DATASETS = ['coco', 'mpii_3d', 'h36m']
    # ['tfrecords', 'tfrecords_with_toes', 'tfrecords_balanced_3d]
    DATA_DIR = os.path.join(ROOT_DATA_DIR, 'tfrecords')

    # path to saved smpl data in tf record format
    # folder names should be same as defined in SMPL_DATASETS config (see below):
    # e.g. SMPL_DATASETS = ['cmu', 'joint_lim']
    SMPL_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'tfrecords', 'smpl')

    # path to the neutral smpl model
    # ['neutral_smpl_coco_regressor.pkl', 'neutral_smpl_coco_regressor_tool_shoulders.pkl']
    SMPL_MODEL_PATH = os.path.join(ROOT_PROJECT_DIR, 'models', 'neutral_smpl_coco_regressor.pkl')

    # path to mean theta h5 file
    SMPL_MEAN_THETA_PATH = os.path.join(ROOT_PROJECT_DIR, 'models', 'neutral_smpl_mean_params.h5')

    # path to the custom regressors
    CUSTOM_REGRESSOR_PATH = os.path.join(ROOT_PROJECT_DIR, 'models', 'regressors')

    # ------HMR parameters:------
    #
    # input image size to the encoder network after preprocess
    ENCODER_INPUT_SHAPE = (224, 224, 3)

    # number of iterations for regressor feedback loop
    ITERATIONS = 3

    # define joint type returned by SMPL
    # any of [cocoplus, lsp, custom, coco_custom]
    JOINT_TYPE = 'cocoplus'

    # cocoplus: 19 keypoints
    # lsp:  14 keypoints
    # custom: set keypoints according to generated regressors
    DS_KP2D = {
        'lsp': 14,
        'cocoplus': 19,
        'custom': 21
    }
    DS_KP3D = {
        'lsp': 14,
        'cocoplus': 14,
        'custom': 16
    }

    # indices where custom regressors should be places in to joint regressor
    # this depends on how universal keypoints are definded in tfrecord_converter, e.g.:
    # toes left/right have been added accordingly to the scheme of lsp order.
    # Therefore being inserted at index 0 (toes_r) and 7 (toes_l) as lsp is ordered
    # from bottom to top of the body with right side of the body first and then left.
    CUSTOM_REGRESSOR_IDX = {
        0: 'regressor_toes_right.npy',
        7: 'regressor_toes_left.npy'
    }

    # if you want to run inference or evaluation with a pretrained standard lsp or cocoplus model
    # but still regress for the new keypoints set this to True
    INITIALIZE_CUSTOM_REGRESSOR = False

    # number of epochs to train
    EPOCHS = 55

    # effective batch size
    BATCH_SIZE = 64

    # number of parallel read tf record files
    NUM_PARALLEL = 16

    # seed for random shuffle
    SEED = 42

    # list of datasets to use for training
    DATASETS = ['lsp', 'lsp_ext', 'mpii', 'coco', 'mpii_3d', 'h36m', 'total_cap']

    # datasets to use for adversarial prior training
    SMPL_DATASETS = ['cmu', 'joint_lim']  # , 'h36m']

    # if set to True, no adversarial prior is trained = monsters
    ENCODER_ONLY = False

    # set True to use 3D labels
    USE_3D = True

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
        self.NUM_TRAINING_SAMPLES = self.count_samples_of(self.DATASETS, self.DATA_DIR, 'train')

        # number of smpl training samples
        self.NUM_TRAIN_SMPL_SAMPLES = self.count_samples_of(self.SMPL_DATASETS, self.DATA_DIR, 'train')

        # number of validation samples, use `test_count_all_samples` in test_dataset.py to determine
        # For validation coco val set for 2d and for 3d subject 9 from h36m was used
        # (all sequences but only first camera perspective)
        self.NUM_VALIDATION_SAMPLES = self.count_samples_of(self.DATASETS, self.DATA_DIR, 'val')

        # number of test samples, use `test_count_all_samples` in test_dataset.py to determine
        self.NUM_TEST_SAMPLES = self.count_samples_of(self.DATASETS, self.DATA_DIR, 'test')

        self.NUM_KP2D = self.DS_KP2D.get(self.JOINT_TYPE)
        self.NUM_KP3D = self.DS_KP3D.get(self.JOINT_TYPE)

    @staticmethod
    def count_samples_of(datasets, datadir, split):
        """Numbers need to be provided after tf record generation (see `inspect.ipynb`)
        Args:
            datasets: list of dataset names used for training
            split: train|val, define which split to use
        Returns:
            total_number: int, total number of training samples
        """
        train_samples_per_dataset = {
            'tfrecords': {
                'lsp': 999,
                'lsp_ext': 9896,
                'mpii': 16125,
                'coco': 98101,

                'mpii_3d': 166311,
                'h36m': 311950,
                'total_cap': 75060,

                'cmu': 3934266,
                'joint_lim': 181967,
            },
            'tfrecords_with_toes': {
                'lsp': 999,
                'lsp_ext': 9896,
                'mpii': 16125,
                'coco': 98101,

                'mpii_3d': 173126,
                'h36m': 118955,
                'total_cap': 81617,

                'cmu': 3934266,
                'joint_lim': 181967,
            },
            'tfrecords_balanced_3d': {
                'lsp': 999,
                'lsp_ext': 9896,
                'mpii': 16125,
                'coco': 98101,

                'mpii_3d': 166031,
                'h36m': 99552,
                'total_cap': 75060,

                'cmu': 3934266,
                'joint_lim': 181967,
            }
        }

        val_samples_per_dataset = {
            'tfrecords': {
                'lsp': 997,
                'coco': 3984,
                'h36m': 6482,
            },
            'tfrecords_with_toes': {
                'lsp': 997,
                'coco': 3984,
                'h36m': 15883,
            },
            'tfrecords_balanced_3d': {
                'lsp': 997,
                'coco': 3984,
                'h36m': 6120,
            }
        }

        test_samples_per_dataset = {
            'tfrecords': {
                'mpii_3d': 2874,
                'h36m': 110128,
                'total_cap': 73871,
            },
            'tfrecords_with_toes': {
                'h36m': 110128,
                'total_cap': 74273,
            },
            'tfrecords_balanced_3d': {
                'h36m': 110128,
                'total_cap': 74273,
            }
        }

        if split == 'train':
            samples = train_samples_per_dataset
        elif split == 'val':
            samples = val_samples_per_dataset
        elif split == 'test':
            samples = test_samples_per_dataset
        else:
            raise Exception('unknown split')

        if os.path.basename(datadir) not in samples:
            # return 0 for LocalConfig
            return 0

        samples = samples[os.path.basename(datadir)]
        return sum([samples[d] for d in datasets if d in samples.keys()])

    def save_config(self):
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        print('Saving logs to {}'.format(self.LOG_DIR))

        config_path = os.path.join(self.LOG_DIR, "config.json")
        if not os.path.exists(config_path):
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

    def read_config(self):
        config_path = os.path.join(self.LOG_DIR, "config.json")
        if os.path.exists(config_path):
            return json.load(open(config_path))
        else:
            return None

    def reset(self):
        Config.__instance = None
