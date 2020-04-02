from time import time

import numpy as np
import scipy.io as sio
from os.path import join
from tqdm import tqdm

from converter.tfrecord_converter import TFRecordConverter, DataSetConfig, DataSetSplit


class MpiiConverter(TFRecordConverter):

    def __init__(self):
        self.num_kps = 16
        self.mpii_order = ['ankle_r', 'knee_r', 'hip_r', 'hip_l', 'knee_l', 'ankle_l', 'pelvis', 'thorax', 'neck',
                           'brain', 'wrist_r', 'elbow_r', 'shoulder_r', 'shoulder_l', 'elbow_l', 'wrist_l']

        super().__init__()

    def prepare_data(self):
        print('loading annotations into memory...')
        ann_path = join(self.data_dir, 'annotations', 'mpii_human_pose_v1_u12_1.mat')
        annotations = sio.loadmat(ann_path, struct_as_record=False, squeeze_me=True)['RELEASE']

        ids = np.array(range(len(annotations.annolist)))

        train_indices = annotations.img_train.astype(bool)
        train_ids = ids[train_indices]
        self.data_set_splits.append(self.create_dataset_split(annotations, train_ids, 'train'))

        # can't use testing set because ground truth is not available, only generate for submission purpose
        # val_indices = np.logical_not(train_indices)
        # val_ids = ids[val_indices]
        # self.data_set_splits.append(self.create_dataset_split(annotations, val_ids, 'val'))

    def create_dataset_split(self, annotations, img_ids, name):
        def convert_vis(value):
            if type(value) == int or (type(value) == str and value in ['1', '0']):
                return int(value)
            elif isinstance(value, np.ndarray):
                return int(value.size != 0)
            else:
                return 0

        images, kps_2d, vis = [], [], []
        img_dir = join(self.data_dir, 'images')
        print("prepare {} mpii annotations for conversion".format(name))
        for img_id in tqdm(img_ids):
            try:
                ann_info = annotations.annolist[img_id]

                single_persons = annotations.single_person[img_id]
                if not isinstance(single_persons, np.ndarray):
                    single_persons = np.array([single_persons])

                if single_persons.size == 0:
                    continue

                rects = ann_info.annorect
                if not isinstance(rects, np.ndarray):
                    rects = np.array([rects])

                persons = rects[single_persons - 1]
                for person in persons:
                    points = person.annopoints.point
                    if not isinstance(points, np.ndarray):
                        # There is only one! so ignore this image
                        continue

                    kp2d = np.zeros((self.num_kps, 2), np.float32)
                    v = np.zeros((self.num_kps,), np.float32)

                    for p in points:
                        kp2d[p.id] = [p.x, p.y]
                        v[p.id] = convert_vis(p.is_visible)

                    images.append(join(img_dir, ann_info.image.name))
                    kps_2d.append(kp2d)
                    vis.append(v)

            except (AttributeError, TypeError):
                print('error')
                continue

        images = np.asarray(images)
        kps_2d = np.asarray(kps_2d, dtype=np.float32)
        vis = np.asarray(vis, dtype=np.int64)
        mpii_config = DataSetConfig(name, False, self.mpii_order)
        return DataSetSplit(mpii_config, images, kps_2d, vis)


if __name__ == '__main__':
    t0 = time()
    mpii_converter = MpiiConverter()
    print('Done (t={})\n\n'.format(time() - t0))
