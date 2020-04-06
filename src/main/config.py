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

    # if set to True, no adversarial prior is trained = monsters
    ENCODER_ONLY = False

    # set True to use 3D labels
    USE_3D = True

    # ------Training settings:------
    #
    # set default for training mode
    TRAINING = True

    # number of epochs to train
    EPOCHS = 55

    # number of training samples, use `test_count_all_samples` in test_dataset.py to determine
    NUM_SAMPLES = 422445

    # number of validation samples, use `test_count_all_samples` in test_dataset.py to determine
    # For validation coco val set for 2d and for 3d subject 9 from h36m was used
    # (all sequences but only first camera perspective)
    NUM_VALIDATION_SAMPLES = 10922

    # number of test samples, use `test_count_all_samples` in test_dataset.py to determine
    NUM_TEST_SAMPLES = 42369

    # effective batch size
    BATCH_SIZE = 64

    # number of parallel read tf record files
    NUM_PARALLEL = 16

    # seed for random shuffle
    SEED = 42

    # list of datasets to use for training
    # DATASETS = ['lsp', 'lsp_ext', 'mpii', 'coco', 'mpii_3d', 'h36m']
    DATASETS = ['lsp', 'lsp_ext', 'mpii', 'mpii_3d', 'h36m']  # skip coco due to missing extremities annotation

    # datasets to use for adversarial prior training
    SMPL_DATASETS = ['cmu', 'joint_lim']  # , 'H3.6']

    # ------Hyper parameters:------
    #
    # encoder learning rate
    ENCODER_LEARNING_RATE = 1e-5
    # encoder weight decay
    ENCODER_WEIGHT_DECAY = 1e-4
    # weight on encoder loss
    ENCODER_LOSS_WEIGHT = 60.

    # adversarial prior learning rate
    DISCRIMINATOR_LEARNING_RATE = 1e-4
    # adversarial prior weight decay
    DISCRIMINATOR_WEIGHT_DECAY = 1e-4
    # weight on discriminator
    DISCRIMINATOR_LOSS_WEIGHT = 1

    # weight on theta regressor
    REGRESSOR_LOSS_WEIGHT = 60.

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
