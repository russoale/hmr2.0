import csv
from os import listdir
from os.path import join
from time import time

import numpy as np

from converter.helpers import CameraInfo
from converter.tfrecord_converter import TFRecordConverter, DataSetConfig, DataSetSplit


class TotalCaptureConverter(TFRecordConverter):

    def __init__(self):
        # using ankle instead of heel
        self.tc_order = ['shoulder_r', 'spine', 'hip_l', 'knee_r', 'shoulder_l', 'wrist_l', 'spine2', 'ankle_r',
                         'collarbone_r', 'thumb_r', 'thumb_end_r', 'ankle_l', 'elbow_l', 'head', 'brain',
                         'spine3', 'foot_r', 'toes_r', 'spine1', 'hip_r', 'neck', 'elbow_r', 'knee_l', 'wrist_r',
                         'thumb_l', 'thumb_end_l', 'collarbone_l', 'pelvis', 'fingers_l', 'fingers_end_l',
                         'fingers_r', 'fingers_end_r', 'foot_l', 'toes_l']

        self.split_dict = {
            'train': {
                'sub_ids': [1, 2, 3, 4, 5],
                'seq_ids': ['acting1', 'acting2', 'walking1', 'walking3',
                            'freestyle1', 'freestyle2', 'rom1', 'rom2', 'rom3'],
                'skip_frames': -1
            },
            'test': {
                'sub_ids': [1, 2, 3, 4, 5],
                'seq_ids': ['acting3', 'freestyle3', 'walking2'],
                'skip_frames': 5
            }
        }
        super().__init__()

    def prepare_data(self):
        for split_name, value in self.split_dict.items():
            sub_ids = value['sub_ids']
            seq_ids = value['seq_ids']

            image_paths, kps_2d, kps_3d, sequences = [], [], [], []

            idx = 0
            for sub_id in sub_ids:
                print('convert subject {} {}/{}'.format(sub_id, sub_ids.index(sub_id) + 1, len(sub_ids)))
                ann_path = join(self.data_dir, 'S%d' % sub_id, 'Positions_3D')
                seqs = [str(s.rsplit('.', 1)[0]) for s in listdir(ann_path)]

                for seq in seqs:
                    if seq not in seq_ids:
                        continue

                    with open(join(ann_path, seq + '.csv')) as file:
                        reader = csv.reader(file, delimiter=';')
                        content = {}
                        cam_info = []
                        for row in reader:
                            if row[0] == '#camerainfo':
                                key = 'cam' + row[1]
                                cam_info.append(key)
                            else:
                                key = row[0]
                            content[key] = row[1:]
                        content['#camerainfo'] = cam_info

                    cam_infos = {cam: CameraInfo.from_line(content[cam]) for cam in content['#camerainfo']}

                    num_frames = int(content['#number_of_frames'][0])
                    num_points = len(content['#point_names'])
                    num_dimens = int(content['#dimensions'][0])
                    shape = (num_frames, num_points, num_dimens)

                    data = np.empty(shape, dtype=np.float32)
                    for f in range(num_frames):
                        kp3d = np.asarray(content[str(f)]).reshape((-1, 3))
                        data[f] = kp3d[:num_points]

                    for cam, info in cam_infos.items():
                        kp3d = np.add(np.tensordot(data, info.R, axes=(2, 1)).reshape(shape), info.T)
                        kp2d = kp3d[:, :, :2] * info.f / np.expand_dims(kp3d[:, :, 2], -1) + info.o

                        sub_dir = 'S%d' % sub_id
                        scene_dir = 'TC_{}_{}_{}'.format(sub_dir, seq, cam)
                        img_dir = join(self.data_dir, sub_dir, 'imageFrames', scene_dir, 'frame_%06d.jpg')

                        idx += 1
                        for i in range(kp2d.shape[0]):
                            image_paths.append(img_dir % (i + 1))
                            kps_2d.append(kp2d[i])
                            kps_3d.append(kp3d[i])

                            if split_name == 'test':
                                sequences.append(scene_dir)

            image_paths = np.asarray(image_paths)
            kps_2d = np.asarray(kps_2d)
            kps_3d = np.asarray(kps_3d)
            sequences = np.asarray(sequences) if split_name == 'test' else None

            if value['skip_frames'] > 1:
                skip_frames = value['skip_frames']
                image_paths = image_paths[::skip_frames]
                kps_2d = kps_2d[::skip_frames]
                kps_3d = kps_3d[::skip_frames]
                sequences = sequences[::skip_frames] if split_name == 'test' else None

            tc_config = DataSetConfig(split_name, has_3d=True, reorder=self.tc_order, lsp_only=True)
            self.data_set_splits.append(DataSetSplit(tc_config, image_paths, kps_2d, kps_3d=kps_3d, seqs=sequences))


if __name__ == '__main__':
    t0 = time()
    h36m_converter = TotalCaptureConverter()
    print('Done (t={})\n\n'.format(time() - t0))
