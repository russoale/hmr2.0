import csv
from time import time

import numpy as np
from os import listdir
from os.path import join

from converter.tfrecord_converter import TFRecordConverter, DataSetConfig, DataSetSplit


class H36MConverter(TFRecordConverter):

    def __init__(self):
        # using ankle instead of heel
        self.h36m_order = ['pelvis', 'hip_r', 'knee_r', 'ankle_r', 'foot_r', 'toes_r', 'hip_l', 'knee_l', 'ankle_l',
                           'foot_l', 'toes_l', 'torso', 'neck', 'head', 'brain', 'shoulder_l', 'elbow_l', 'wrist_l',
                           'thumb_l', 'fingers_l', 'shoulder_r', 'elbow_r', 'wrist_r', 'thumb_r', 'fingers_r']

        self.split_dict = {
            'train': {
                'sub_ids': [1, 5, 6, 7, 8]
            },
            'val': {
                'sub_ids': [9]
            },
            'test': {
                'sub_ids': [9, 11]
            },
        }

        super().__init__()

    def prepare_data(self):
        for split_name, value in self.split_dict.items():
            sub_ids = value['sub_ids']

            image_paths, kps_2d, kps_3d, sequences = [], [], [], []

            idx = 0
            for sub_id in sub_ids:
                print('convert subject {} {}/{}'.format(sub_id, sub_ids.index(sub_id) + 1, len(sub_ids)))
                ann_path = join(self.data_dir, 'S%d' % sub_id, 'Positions_3D')
                seqs = [str(s.rsplit('.', 1)[0]) for s in listdir(ann_path)]

                for seq in seqs:
                    with open(join(ann_path, seq + '.csv')) as file:
                        reader = csv.reader(file, delimiter=';')
                        content = {}
                        cam_info = []
                        for row in reader:
                            if row[0] == '#camerainfo':
                                key = row[1]
                                cam_info.append(key)
                            else:
                                key = row[0]
                            value = row[1:]
                            content[key] = value
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

                        img_dir = join(self.data_dir, 'S%d' % sub_id, 'imageFrames', seq + info.suffix,
                                       'frame_%06d.jpg')

                        idx += 1
                        for i in range(kp2d.shape[0]):
                            image_paths.append(img_dir % (i + 1))
                            kps_2d.append(kp2d[i])
                            kps_3d.append(kp3d[i])

                            if split_name == 'test':
                                sequences.append('h36m_sub{}_{}_cam{}'.format(sub_id, seq, cam))

                        if split_name == 'val':
                            # for val set only use first camera perspective
                            break

            image_paths = np.asarray(image_paths)
            kps_2d = np.asarray(kps_2d)
            kps_3d = np.asarray(kps_3d)
            sequences = np.asarray(sequences) if split_name == 'test' else None

            h36m_config = DataSetConfig(split_name, has_3d=True, reorder=self.h36m_order, lsp_only=True)
            self.data_set_splits.append(DataSetSplit(h36m_config, image_paths, kps_2d, kps_3d=kps_3d, seqs=sequences))


class CameraInfo:

    def __init__(self):
        self.name = ''
        self.suffix = ''

        # extrinsic parameters
        self.R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.T = np.array([0, 0, 0], dtype=np.float32)

        # intrinsic parameters
        self.f = np.array([1000, 1000], dtype=np.float32)
        self.o = np.array([500, 500], dtype=np.float32)

    @staticmethod
    def from_line(camera_info_line):
        """Read method to be used by the annotation class while reading a file
        Args:
            camera_info_line: The line specifying the camera info as taken from the #Camerainfo section,
                            with the leftmost placeholder removed
        Returns:
            A camera info class with the data loaded from the camera_info_line
        """
        camera_info = CameraInfo()
        camera_info.name = camera_info_line[0]
        camera_info.suffix = camera_info_line[1]
        camera_info.R = np.array(camera_info_line[2:11], dtype=np.float32).reshape((3, 3))
        camera_info.T = np.array(camera_info_line[11:14], dtype=np.float32)
        camera_info.f = np.array(camera_info_line[14:16], dtype=np.float32)
        camera_info.o = np.array(camera_info_line[16:18], dtype=np.float32)
        return camera_info


if __name__ == '__main__':
    t0 = time()
    h36m_converter = H36MConverter()
    print('Done (t={})\n\n'.format(time() - t0))
