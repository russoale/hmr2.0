from time import time

import numpy as np
from os.path import join
from pycocotools.coco import COCO
from tqdm import tqdm

from converter.tfrecord_converter import TFRecordConverter, DataSetSplit, DataSetConfig


class CocoConverter(TFRecordConverter):

    def __init__(self):
        self.coco_order = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r', 'shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r',
                           'wrist_l', 'wrist_r', 'hip_l', 'hip_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r']

        self.face_and_shoulder = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r', 'shoulder_l', 'shoulder_r']

        super().__init__()

    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument('--year', required=False, default='2017', metavar='<2014|2017>', help='data set year')

    def prepare_data(self):
        splits = ['train', 'val']
        for split in splits:
            image_paths, kps_2d, vis = [], [], []

            data_type = split + self.args.year
            img_dir = join(self.data_dir, data_type)

            # load coco annotations
            self.coco = COCO(join(self.data_dir, 'annotations', 'person_keypoints_{}.json'.format(data_type)))
            # get id for category person
            cat_ids = self.coco.getCatIds(catNms=['person'])
            # get all image id's containing instances of person category
            img_ids = self.coco.getImgIds(catIds=cat_ids)

            # np.random.shuffle(img_ids)
            # img_ids = img_ids[:1000]

            print("prepare coco annotations for conversion")
            for img_id in tqdm(img_ids):
                img_path = join(img_dir, '%012d.jpg' % img_id)

                ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
                img_annotations = self.coco.loadAnns(ann_ids)
                for ann in img_annotations:
                    keypoints = np.array(ann['keypoints'])
                    keypoints = np.reshape(keypoints, (-1, 3))

                    image_paths.append(img_path)
                    kps_2d.append(keypoints[:, :2])
                    vis.append(keypoints[:, 2] == 2)

            image_paths = np.asarray(image_paths)
            kps_2d = np.asarray(kps_2d, dtype=np.float32)
            vis = np.asarray(vis, dtype=np.int64)
            # generate zero placeholders

            coco_config = DataSetConfig(split, False, self.coco_order, self.face_and_shoulder)
            self.data_set_splits.append(DataSetSplit(coco_config, image_paths, kps_2d, vis))


if __name__ == '__main__':
    t0 = time()
    coco_converter = CocoConverter()
    print('Done (t={})\n\n'.format(time() - t0))
