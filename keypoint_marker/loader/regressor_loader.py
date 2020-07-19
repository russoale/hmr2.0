from glob import glob

import numpy as np


class RegressorLoader:

    def __init__(self):
        super(RegressorLoader, self).__init__()
        self.joint_regressor = None

    def init_regressors(self, path):
        files = glob(path)
        if len(files) > 0:
            regressors = []
            for file in files:
                regressor = np.load(file)
                regressors.append(regressor)

            self.joint_regressor = np.concatenate(regressors, 1).T
