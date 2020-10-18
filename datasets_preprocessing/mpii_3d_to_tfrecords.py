from os.path import join
from time import time

import numpy as np

from converter.tfrecord_converter import TFRecordConverter, DataSetConfig, DataSetSplit


class Mpii3dConverter(TFRecordConverter):

    def __init__(self):
        # when using toes
        # self.mpii_3d_ids = np.array([8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7, 23, 28]) - 1
        # self.mpii_3d_order = ['brain', 'neck', 'shoulder_r', 'elbow_r', 'wrist_r', 'shoulder_l', 'elbow_l', 'wrist_l',
        #                       'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l', 'pelvis', 'spine', 'head',
        #                       'toes_l', 'toes_r']

        self.mpii_3d_ids = np.array([8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]) - 1
        self.mpii_3d_order = ['brain', 'neck', 'shoulder_r', 'elbow_r', 'wrist_r', 'shoulder_l', 'elbow_l', 'wrist_l',
                              'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l', 'pelvis', 'spine', 'head']

        self.split_dict = {
            'train': {
                'sub_ids': [1, 2, 3, 4, 5, 6, 7, 8],
                'seq_ids': [1, 2],
                'cam_ids': [0, 1, 2, 4, 5, 6, 7, 8]
            },
            # test set does not contain toes annotation
            'test': {
                'sub_ids': [1, 2, 3, 4, 5, 6],
                'seq_ids': None,
                'cam_ids': None
            }
        }
        super().__init__()

    def prepare_data(self):
        for split_name, value in self.split_dict.items():
            image_paths, kps_2d, kps_3d, sequences = [], [], [], []

            sub_ids = value['sub_ids']
            seq_ids = value['seq_ids']
            cam_ids = value['cam_ids']
            idx = 0
            for sub_id in sub_ids:
                print('convert subject {} {}/{}'.format(sub_id, sub_ids.index(sub_id) + 1, len(sub_ids)))
                if seq_ids is None and cam_ids is None:
                    img_dir, image_ids, kp2d, kp3d = self.convert_test(sub_id)
                    for i, image_id in enumerate(image_ids):
                        image_paths.append(img_dir % image_id)
                        kps_2d.append(kp2d[i])
                        kps_3d.append(kp3d[i])
                        sequences.append('mpii_ts{}'.format(sub_id))
                else:
                    for seq_id in seq_ids:
                        for cam_id in cam_ids:
                            img_dir, num_images, kp2d, kp3d = self.convert_train(sub_id, seq_id, cam_id)

                            idx += 1
                            for i in range(num_images):
                                image_paths.append(img_dir % (i + 1))
                                kps_2d.append(kp2d[i])
                                kps_3d.append(kp3d[i])

            image_paths = np.asarray(image_paths)
            kps_2d = np.asarray(kps_2d)
            kps_3d = np.asarray(kps_3d)
            sequences = np.asarray(sequences) if split_name == 'test' else None

            mpii_config = DataSetConfig(split_name, True, self.mpii_3d_order, lsp_only=True)
            self.data_set_splits.append(DataSetSplit(mpii_config, image_paths, kps_2d, kps_3d=kps_3d, seqs=sequences))

    def convert_train(self, sub_id, seq_id, cam_id):
        # prepare paths
        seq_dir = join(self.data_dir, 'S%d' % sub_id, 'Seq%d' % seq_id)
        ann_path = join(seq_dir, 'annot.mat')
        img_dir = join(seq_dir, 'imageFrames', 'video_%d' % cam_id, 'frame_%06d.jpg')

        from scipy.io import loadmat
        res = loadmat(ann_path, squeeze_me=True, variable_names=['frames', 'annot2', 'annot3'])

        num_images = res['frames'].shape[0]
        kp2d = res['annot2'][cam_id].astype(np.float32).reshape(num_images, -1, 2)
        kp3d = res['annot3'][cam_id].astype(np.float32).reshape(num_images, -1, 3)

        kp2d = kp2d[:, self.mpii_3d_ids, :]
        kp3d = kp3d[:, self.mpii_3d_ids, :]
        kp3d = np.divide(kp3d, np.float32(1000.), dtype=np.float32)  # Fix units: mm -> meter

        return img_dir, num_images, kp2d, kp3d

    def convert_test(self, sub_id):
        # prepare paths
        seq_dir = join(self.data_dir, 'test', 'TS%d' % sub_id)
        ann_path = join(seq_dir, 'annot_data.mat')
        img_dir = join(seq_dir, 'imageSequence', 'img_%06d.jpg')

        import h5py
        with h5py.File(ann_path, 'r') as f:
            valid_frames = f.get('valid_frame')[()].flatten().astype(bool)
            kp2d = f.get('annot2')[()].astype(np.float32)[valid_frames]
            kp3d = f.get('annot3')[()].astype(np.float32)[valid_frames]
            image_ids = np.arange(1, valid_frames.shape[0] + 1)[valid_frames]

        kp2d = np.squeeze(kp2d)
        kp3d = np.squeeze(kp3d)
        kp3d = np.divide(kp3d, np.float32(1000.), dtype=np.float32)  # Fix units: mm -> meter

        return img_dir, image_ids, kp2d, kp3d


if __name__ == '__main__':
    t0 = time()
    mpii3d_converter = Mpii3dConverter()
    print('Done (t={})\n\n'.format(time() - t0))
