from glob import glob
from os.path import join
from time import time

import numpy as np
import scipy.io as sio

from converter.tfrecord_converter import DataSetConfig, DataSetSplit, TFRecordConverter


class LspConverter(TFRecordConverter):

    def __init__(self):
        self.lsp_order = ['ankle_r', 'knee_r', 'hip_r', 'hip_l', 'knee_l', 'ankle_l', 'wrist_r',
                          'elbow_r', 'shoulder_r', 'shoulder_l', 'elbow_l', 'wrist_l', 'neck', 'brain']

        super().__init__()

    def prepare_data(self):
        image_paths = np.array(sorted([f for f in glob(join(self.data_dir, 'images/*.jpg'))]))
        joints = sio.loadmat(join(self.data_dir, 'joints.mat'), squeeze_me=True)['joints'].astype(np.float32)
        if self.args.dataset_name == 'lsp_ext':
            joints = np.transpose(joints, (2, 0, 1))
            kps_2d = joints[:, :, :2]
            vis = joints[:, :, 2].astype(np.int64)

            lsp_config = DataSetConfig('train', has_3d=False, reorder=self.lsp_order)
            self.data_set_splits.append(DataSetSplit(lsp_config, image_paths, kps_2d, vis))

        elif self.args.dataset_name == 'lsp':
            joints = np.transpose(joints, (2, 1, 0))
            kps_2d = joints[:, :, :2]
            vis = (1 - joints[:, :, 2]).astype(np.int64)

            lsp_config_train = DataSetConfig('train', has_3d=False, reorder=self.lsp_order)
            self.data_set_splits.append(DataSetSplit(lsp_config_train, image_paths[:1000], kps_2d[:1000], vis[:1000]))

            lsp_config_test = DataSetConfig('val', has_3d=False, reorder=self.lsp_order)
            self.data_set_splits.append(DataSetSplit(lsp_config_test, image_paths[1000:], kps_2d[1000:], vis[1000:]))
        else:
            raise Exception('unknown LSP dataset name')


if __name__ == '__main__':
    t0 = time()
    lsp_converter = LspConverter()
    print('Done (t={})\n\n'.format(time() - t0))
