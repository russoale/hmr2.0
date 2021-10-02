import argparse
from os import path, makedirs, listdir, environ

import abc
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# tf INFO and WARNING messages are not printed
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from converter.helpers import check_np_array, check_type, int64_feature, float_feature, bytes_feature, resize_img


class TFRecordConverterConfig:
    # if generating datasets for toes set: num_kp2d=21, num_kp3d=16
    def __init__(self, num_kp2d=19, num_kp3d=14, margin=150, min_vis=6, min_height=60, min_3d_mov=.2, max_scale=150.):
        self.num_kp2d = num_kp2d
        self.num_kp3d = num_kp3d
        self.margin = margin
        self.min_vis = min_vis
        self.min_height = min_height
        self.min_3d_mov = min_3d_mov
        self.max_scale = max_scale


class TFRecordConverter(abc.ABC):

    def __init__(self):
        self.config = TFRecordConverterConfig()

        self.parser = argparse.ArgumentParser()
        self.add_arguments()
        self.args, _ = self.parser.parse_known_args()

        self.data_dir = path.join(self.args.data_directory, self.args.dataset_name)
        self.output_dir = path.join(self.args.output_directory, self.args.dataset_name)

        if not path.exists(self.output_dir):
            makedirs(self.output_dir)
        print('Saving results to {}'.format(self.output_dir))

        # create data
        self.__data_set_splits = []
        self.__examples = []

        self.prepare_data()  # this needs to be implemented by subclass

        print('\n-----TFRecordConverter-----')
        self.check_data()
        self.filter_data()
        self.convert_data()

    def add_arguments(self):
        self.parser.add_argument('--data_directory', required=False, metavar='/path/to/data',
                                 help='Directory containing the original dataset')
        self.parser.add_argument('--output_directory', required=False, metavar='/path/to/output/data',
                                 help='Directory where to store the generated TFRecord')
        self.parser.add_argument('--dataset_name', required=False, metavar='<dataset name>',
                                 help='Name of the dataset to be converted')
        self.parser.add_argument('--num_shards', required=False, default=500, metavar='<int>',
                                 help='Number of shards in TFRecord files')

    def check_data(self):
        """Check if data has been passed correctly to TFRecordConverter.
           If vis or 3d keypoints are not passed this will create placeholders.
           If a dataset specific reorder is set in dataset config this will reorder
           given keypoints 2D, 3D and visibility according to it.
        """
        print('check data and reorder if necessary...')

        def _reorder(value, reorder=None, lsp_only=False, num_kp3d=None):
            if reorder is None:
                return value

            reorder = reorder[:num_kp3d] if lsp_only else reorder
            # this sets missing kp annotations to [0, 0]
            zero = np.zeros_like(value[:, :1])
            return np.concatenate([value, zero], axis=1)[:, reorder]

        for d in self.__data_set_splits:
            count = d.image_paths.shape[0]

            d.kps_2d = _reorder(d.kps_2d, d.config.reorder)

            if d.vis is None:
                # if vis is not available then assuming all keypoints are visible
                d.vis = np.ones((count, d.kps_2d.shape[1]), dtype=np.int64)
                # check if any keypoints have been added by reorder
                # if so set those to not visible
                not_visible = np.where(d.kps_2d == 0.0)
                d.vis[not_visible[0], not_visible[1]] = 0
            else:
                d.vis = _reorder(d.vis, d.config.reorder)

            if d.kps_3d is None:
                # if dataset has no 3D keypoints create placeholders
                d.kps_3d = np.zeros((count, self.config.num_kp3d, 3), dtype=np.float32)
            else:
                d.kps_3d = _reorder(d.kps_3d, d.config.reorder, lsp_only=True)

            if self.config.num_kp2d > d.kps_2d.shape[1]:
                # Padding is necessary
                p = self.config.num_kp2d - d.kps_2d.shape[1]
                d.kps_2d = np.pad(d.kps_2d, [[0, 0], [0, p], [0, 0]], mode='constant', constant_values=0.0)
                d.vis = np.pad(d.vis, [[0, 0], [0, p]], mode='constant', constant_values=0.0)

            if d.kps_2d.min() < 0:
                # some 3D datasets contain negative 2D coordinates
                # due to 3D re-projection the keypoints appear outside the camera view
                frame_ids, kp_ids = np.where(np.any(d.kps_2d < 0, axis=2))
                d.kps_2d[frame_ids, kp_ids, :] = np.float32(0.)
                d.vis[frame_ids, kp_ids] = np.int64(0)

            if d.seqs is None:
                d.seqs = np.zeros(d.image_paths.shape[0])

            assert d.image_paths.shape[0] == d.kps_2d.shape[0] == d.vis.shape[0] == d.kps_3d.shape[0] == \
                   d.seqs.shape[0], "DataSetSplit parameters all need to have same length"

            d.image_paths = check_np_array('image_paths', d.image_paths, (count,))
            d.kps_2d = check_np_array('2d keypoints', d.kps_2d, (count, self.config.num_kp2d, 2), dtype=np.float32)
            d.vis = check_np_array('visibility', d.vis, (count, self.config.num_kp2d,), dtype=np.int64)
            d.kps_3d = check_np_array('keypoints_3d', d.kps_3d, (count, self.config.num_kp3d, 3), dtype=np.float32)
            d.seqs = check_np_array('sequences', d.seqs, (count,))

    def filter_data(self):
        """Filter data given rules:
            - 3D frames from extracted videos are too similar given the body movement
            - not enough 2D keypoints are available
            - no body 2D keypoints are available
            - not enough keypoints visible in image frame
        """
        print('filter data...')

        for d in self.__data_set_splits:
            if d.config.has_3d and d.config.name != 'test':
                use_these = self._filter_3d_frames(d.kps_3d)
                d.image_paths = d.image_paths[use_these]
                d.kps_2d = d.kps_2d[use_these]
                d.vis = d.vis[use_these]
                d.kps_3d = d.kps_3d[use_these]
                d.seqs = d.seqs[use_these]

            use_these = np.zeros(d.image_paths.shape[0], bool)
            indices = range(d.image_paths.shape[0])
            for idx, img, kp2d, vis in zip(indices, d.image_paths, d.kps_2d, d.vis):
                # only face kps visible
                if np.all(vis[d.config.body_idx] == 0):
                    continue

                if sum(vis) <= self.config.min_vis:
                    continue

                vis_kps = kp2d[vis.astype(bool)]
                min_pt = np.min(vis_kps, axis=0)
                max_pt = np.max(vis_kps, axis=0)
                kp_bbox = [max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
                if max(kp_bbox) < self.config.min_height:
                    continue

                use_these[idx] = True

            d.image_paths = d.image_paths[use_these]
            d.kps_2d = d.kps_2d[use_these]
            d.vis = d.vis[use_these]
            d.kps_3d = d.kps_3d[use_these]
            d.seqs = d.seqs[use_these]

    def _filter_3d_frames(self, kps_3d):
        use_these = np.zeros(kps_3d.shape[0], bool)

        use_these[0] = True  # Always use_these first frame.
        prev_kp3d = kps_3d[0]
        indices = range(1, kps_3d.shape[0])
        for idx, kp3d in zip(indices, kps_3d):
            # Check if any joint moved more than 200mm.
            if not np.any(np.linalg.norm(prev_kp3d - kp3d, axis=1) >= self.config.min_3d_mov):
                continue
            use_these[idx] = True
            prev_kp3d = kp3d

        return use_these

    def convert_data(self):
        print('convert data...')
        for d in self.__data_set_splits:
            universal_order = d.config.universal_order
            len_images = d.image_paths.shape[0]
            save_total = int(np.ceil(len_images / self.args.num_shards))
            save_points = np.arange(1, save_total) * self.args.num_shards
            save_points = np.append(save_points, len_images - 1)  # save rest when arrived at last element

            # to create a file for unit tests with less than num_shades samples start from 0
            total = tqdm(zip(range(1, len_images), d.image_paths, d.kps_2d, d.vis, d.kps_3d, d.seqs), total=len_images)
            total.set_description_str('processing')
            for i, image_path, kp2d, vis, kp3d, seq in total:
                scale_and_crop = self._scale_and_crop(universal_order, image_path, kp2d, vis, kp3d)
                if scale_and_crop is None:
                    if i in save_points:
                        self._save(d.config, lambda message: total.write(message))
                    continue

                image, kp2d, vis, kp3d = scale_and_crop
                self._create_and_add_example(d.config, image, kp2d, vis, kp3d, seq)

                if i in save_points:
                    self._save(d.config, lambda message: total.write(message))

    def _scale_and_crop(self, universal_order, img_path, kp2d, vis, kps_3d):
        """Scale image and keypoints and crop image given TFRecordConverterConfig"""
        if not path.exists(img_path):
            return

        image = tf.image.decode_image(open(img_path, 'rb').read(), channels=3).numpy()
        scale, center = self._calc_scale_and_center(kp2d, vis, universal_order)
        image, scale = resize_img(image, scale)

        kp2d[:, 0] *= scale[0]
        kp2d[:, 1] *= scale[1]
        center = np.round(center * scale).astype(np.int)

        # crop image (2 * margin) x (2 * margin) around the center
        top_left = np.maximum(center - self.config.margin, 0).astype(int)
        bottom_right = (center + self.config.margin).astype(int)

        # make sure image won't be upscaled
        height, width = image.shape[:2]
        bottom_right[0] = np.minimum(bottom_right[0], width)
        bottom_right[1] = np.minimum(bottom_right[1], height)

        image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        kp2d[:, 0] -= top_left[0]
        kp2d[:, 1] -= top_left[1]

        if not self._check_min_vis(image, kp2d, vis):
            return

        return image, kp2d, vis, kps_3d

    def _check_min_vis(self, img, kp2d, vis):
        height, width = img.shape[:2]

        x_in = np.logical_and(kp2d[:, 0] < width, kp2d[:, 0] >= 0)
        y_in = np.logical_and(kp2d[:, 1] < height, kp2d[:, 1] >= 0)
        kps_in = np.logical_and(x_in, y_in)

        kps_out = np.logical_not(kps_in)
        kp2d[kps_out, :] = np.float32(0.)
        vis[kps_out] = np.int64(0)

        return np.sum(kps_in) >= self.config.min_vis

    def _calc_scale_and_center(self, kp2d, vis, universal_order):
        """Calculates scale based on given keypoints and max scale from Config.
            - if ankles are visible use full person height
            - if torso is visible doubled torso height
            - else use tippled person height
        """
        # Scale person to be roughly max scale height
        vis = vis.astype(bool)
        min_pt = np.min(kp2d[vis], axis=0)
        max_pt = np.max(kp2d[vis], axis=0)
        center = (min_pt + max_pt) / 2.
        person_height = np.linalg.norm(max_pt - min_pt)

        # If ankles are visible
        ankle_l = universal_order.index('ankle_l')
        ankle_r = universal_order.index('ankle_r')
        if vis[ankle_l] or vis[ankle_r]:
            return self.config.max_scale / person_height, center

        shoulder_l = universal_order.index('shoulder_l')
        shoulder_r = universal_order.index('shoulder_r')
        hip_l = universal_order.index('hip_l')
        hip_r = universal_order.index('hip_r')

        # Torso points left shoulder, right shoulder, left hip, right hip
        torso_heights = []
        if vis[shoulder_l] and vis[hip_l]:
            torso_heights.append(np.linalg.norm(kp2d[shoulder_l] - kp2d[hip_l]))
        if vis[shoulder_r] and vis[hip_r]:
            torso_heights.append(np.linalg.norm(kp2d[shoulder_r] - kp2d[hip_r]))

        if len(torso_heights) > 0:
            return self.config.max_scale / (np.mean(torso_heights) * 2), center
        else:  # No torso!
            return self.config.max_scale / (person_height * 3), center

    def _create_and_add_example(self, config, img, kp2d, vis, kp3d, seq):
        image_string = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[1].tobytes()
        kp2d_vis = np.column_stack([kp2d, vis])
        feat_dict = {
            'image_raw': bytes_feature(tf.compat.as_bytes(image_string)),
            'keypoints_2d': float_feature(kp2d_vis),
            'keypoints_3d': float_feature(kp3d),
            'has_3d': int64_feature(config.has_3d),
        }

        if config.name == 'test':
            as_bytes = tf.compat.as_bytes(seq)
            if as_bytes == b'':
                print('not value!')
            feat_dict.update({'sequence': bytes_feature(as_bytes)})

        self.__examples.append(tf.train.Example(features=tf.train.Features(feature=feat_dict)))

    def _save(self, config, print_saving):
        num_record = 0
        if not (len(listdir(self.output_dir)) == 0):
            import re
            num_record = int(max([re.findall(r"\d+", f)[0] for f in listdir(self.output_dir)])) + 1

        record_name = path.join(self.output_dir, '%03d_{}.tfrecord'.format(config.name))
        tf_record_name = record_name % num_record
        print_saving('saving {}'.format(tf_record_name))
        with tf.io.TFRecordWriter(tf_record_name) as writer:
            for x in self.__examples[:self.args.num_shards]:
                writer.write(x.SerializeToString())
        self.__examples = self.__examples[self.args.num_shards:]

    @abc.abstractmethod
    def prepare_data(self):
        raise NotImplementedError('prepare_data method not yet implemented')

    @property
    def data_set_splits(self):
        return self.__data_set_splits

    @data_set_splits.setter
    def data_set_splits(self, value):
        if not isinstance(value, list) and value == []:
            raise ValueError('data set splits should be of type List and not empty!')

        if any(not isinstance(x, DataSetSplit) for x in value):
            raise ValueError('data set splits must be a list of type DataSetSplits!')

        self.__data_set_splits = value


class DataSetConfig:

    def __init__(self, name, has_3d, reorder=None, face_and_shoulder=None, lsp_only=False):
        # for custom regressors add the corresponding keypoints to universal order
        # this will force all datasets to add the given keypoint if available else
        # will pad it with [0, 0] and visibility 0
        """
        when generating toes use following universal order:
         ['toes_r', 'ankle_r', 'knee_r', 'hip_r', 'hip_l', 'knee_l', 'ankle_l', 'toes_l',
            'wrist_r', 'elbow_r', 'shoulder_r', 'shoulder_l', 'elbow_l', 'wrist_l', 'neck', 'brain']
        """

        self.universal_order = ['ankle_r', 'knee_r', 'hip_r', 'hip_l', 'knee_l', 'ankle_l', 'wrist_r', 'elbow_r',
                                'shoulder_r', 'shoulder_l', 'elbow_l', 'wrist_l', 'neck', 'brain']

        default_face_and_shoulder = ['shoulder_l', 'shoulder_r', 'neck', 'brain']
        if face_and_shoulder is not None:
            # merge face and shoulder kps without duplicate
            face_and_shoulder = list(set(face_and_shoulder + default_face_and_shoulder))
        else:
            face_and_shoulder = default_face_and_shoulder

        self.body_idx = [i for i, kp in enumerate(self.universal_order) if kp not in face_and_shoulder]
        self.lsp_only = lsp_only

        self.name = check_type('name', name, str)
        self.has_3d = check_type('has_3d', has_3d, int)
        self.reorder = self.__create_reorder_idx(reorder) if reorder is not None else None

    def __create_reorder_idx(self, reorder):
        final_order = []
        for kp in self.universal_order:
            final_order.append(reorder.index(kp) if kp in reorder else -1)
        if not self.lsp_only:
            for idx, kp in enumerate(reorder):
                if kp not in self.universal_order:
                    final_order.append(idx)

        return final_order


class DataSetSplit:

    def __init__(self, config: DataSetConfig, image_paths, kps_2d, vis=None, kps_3d=None, seqs=None):
        self.config = config
        self.image_paths = check_type('image_paths', image_paths, np.ndarray)
        self.kps_2d = check_type('2d keypoints', kps_2d, np.ndarray)
        self.vis = check_type('visibility', vis, np.ndarray) if vis is not None else None
        self.kps_3d = check_type('3d keypoints', kps_3d, np.ndarray) if kps_3d is not None else None
        self.seqs = check_type('sequences', seqs, np.ndarray) if seqs is not None else None
