import pickle
import random

import numpy as np


class PoseLoader:

    def __init__(self):
        super(PoseLoader, self).__init__()

    def init_poses(self, file_name):
        with open(file_name, "rb") as f:
            res = pickle.load(f, encoding='latin-1')

        self.num_poses = res['poses'].shape[0]
        self.poses = res['poses']
        self.shapes = np.tile(np.reshape(res['betas'], (10, 1)), self.num_poses).T
        self.transforms = res['trans']

    def sample_poses(self, k=10):
        ids = random.sample(range(0, self.num_poses), k)
        ids = [187, 1, 351, 681, 2052, 2077, 2402, 2431, 2439, 75]

        poses = self.poses[ids].reshape([k, -1, 3])
        shapes = self.shapes[ids]
        transforms = self.transforms[ids].reshape([k, -1, 3])
        return poses, shapes, transforms
