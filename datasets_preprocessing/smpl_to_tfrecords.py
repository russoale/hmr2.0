import glob
import pickle
from time import time

import numpy as np
from os import path

from converter.smpl_tfrecord_converter import SmplTFRecordConverter, SmplDataSetConfig, SmplDataSetSplit


class NeutralSmplConverter(SmplTFRecordConverter):

    def prepare_data(self):
        pkl_files = sorted([f for f in glob.glob(path.join(self.data_dir, '*/*.pkl'))])

        poses, shapes = [], []
        idx = 0
        for pkl in pkl_files:
            if len(pkl_files) % 100 == 0:
                print('convert pkl {}/{}'.format(pkl_files.index(pkl) + 1, len(pkl_files)))

            with open(pkl, 'rb') as f:
                res = pickle.load(f, encoding='latin-1')

            key = 'poses' if 'poses' in res.keys() else 'new_poses'
            pose = res[key]
            shape = np.tile(np.reshape(res['betas'], (10, 1)), pose.shape[0]).T
            num_poses = res[key].shape[0]

            idx += 1
            for i in range(num_poses):
                poses.append(pose[i])
                shapes.append(shape[i])

        poses = np.asarray(poses)
        shapes = np.asarray(shapes)

        config = SmplDataSetConfig('train')
        self.smpl_data_set_splits = [SmplDataSetSplit(config, poses, shapes)]


if __name__ == '__main__':
    t0 = time()
    neutral_smpl_converter = NeutralSmplConverter()
    print('Done (t={})\n\n'.format(time() - t0))
