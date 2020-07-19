from glob import glob
from os.path import join
from time import time

import tensorflow as tf

from main.config import Config


class Dataset(object):

    def __init__(self):
        self.config = Config()
        if self.config.JOINT_TYPE == 'cocoplus':
            # flipping ids for lsp with coco
            self.flip_ids_kp2d = tf.constant([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 16, 15, 18, 17])
        else:
            # flipping ids for lsp with coco including custom added toes
            self.flip_ids_kp2d = tf.constant([7, 6, 5, 4, 3, 2, 1, 0, 13, 12, 11, 10, 9, 8, 14, 15, 16, 18, 17, 20, 19])
        self.flip_ids_kp3d = self.flip_ids_kp2d[:self.config.NUM_KP3D]

    ############################################################
    #  Train/Val Dataset Loader
    ############################################################

    def get_train(self):
        start = time()
        print('initialize train dataset...')
        dataset = self.create_dataset('train', self._parse, self._random_jitter)
        dataset = dataset.shuffle(10000, seed=self.config.SEED, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.config.BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        print('Done (t={})\n'.format(time() - start))
        return dataset

    def get_val(self):
        start = time()
        print('initialize val dataset...')
        val_dataset = self.create_dataset('val', self._parse, self._convert_and_scale)
        val_dataset = val_dataset.batch(self.config.BATCH_SIZE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        print('Done (t={})\n'.format(time() - start))
        return val_dataset

    def _parse(self, example_proto):
        feature_map = {
            'image_raw': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            'keypoints_2d': tf.io.VarLenFeature(dtype=tf.float32),
            'keypoints_3d': tf.io.VarLenFeature(dtype=tf.float32),
            'has_3d': tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        features = tf.io.parse_single_example(example_proto, feature_map)
        image_data = features['image_raw']
        kp2d = tf.reshape(tf.sparse.to_dense(features['keypoints_2d']), (self.config.NUM_KP2D, 3))
        kp3d = tf.reshape(tf.sparse.to_dense(features['keypoints_3d']), (self.config.NUM_KP3D, 3))
        has_3d = features['has_3d']

        return image_data, kp2d, kp3d, has_3d

    def _convert_and_scale(self, image_data, kp2d, kp3d, has_3d):
        vis = tf.cast(kp2d[:, 2], tf.float32)
        image = tf.image.decode_jpeg(image_data, channels=3)
        # helpers.show_image(image.numpy(), kp2d.numpy()[:, :2], vis.numpy())

        # convert to [0, 1].
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image_size = tf.cast(tf.shape(image)[:2], tf.float32)

        encoder_img_size = self.config.ENCODER_INPUT_SHAPE[:2]
        image_resize = tf.image.resize(image, encoder_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        actual_factor = tf.cast(tf.shape(image_resize)[:2], tf.float32) / image_size

        kp2d_x = kp2d[:, 0] * actual_factor[1]
        kp2d_y = kp2d[:, 1] * actual_factor[0]
        kp2d_resize = tf.stack([kp2d_x, kp2d_y], axis=1)
        # helpers.show_tf_float_image(image_crop, kp2d_resize[:, :2], kp2d_resize[:, 2])

        # Normalize kp to [-1, 1]
        vis_final = tf.expand_dims(vis, axis=-1)
        kp2d_final = tf.concat([2.0 * (kp2d_resize / encoder_img_size) - 1.0, vis_final], axis=-1)
        # Preserving non_vis to be 0.
        kp2d_final = kp2d_final * vis_final

        # Normalize image to [-1, 1]
        image_final = tf.subtract(image_resize, 0.5)
        image_final = tf.multiply(image_final, 2.0)

        return image_final, kp2d_final, kp3d, has_3d

    def _random_jitter(self, image_data, kp2d, kp3d, has_3d):
        vis = tf.cast(kp2d[:, 2], tf.int32)
        center = self._random_transform_image(kp2d, vis)

        image = tf.image.decode_jpeg(image_data, channels=3)
        # helpers.show_image(image.numpy(), kp2d.numpy()[:, :2], vis.numpy())

        # convert to [0, 1].
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        image_scaled, kp2d_scaled, center_scaled = self._random_scale_image(image, kp2d, center)
        # helpers.show_tf_float_image(image_scaled, kp2d_scaled, vis)

        image_pad, kp2d_pad, center_pad = self._pad_image(image_scaled, kp2d_scaled, center_scaled)
        # helpers.show_tf_float_image(image_pad, kp2d_pad, vis)

        image_crop, kp2d_crop = self._center_crop_image(image_pad, kp2d_pad, center_pad)
        # helpers.show_tf_float_image(image_crop, kp2d_crop, vis)

        image_flipped, kp2d_flipped, vis_flipped, kp3d_flipped = self._random_flip_image(image_crop, kp2d_crop, vis,
                                                                                         kp3d)
        # helpers.show_tf_float_image(image_flipped, kp2d_flipped, vis)

        # Normalize kp to [-1, 1]
        vis_final = tf.expand_dims(tf.cast(vis_flipped, tf.float32), axis=-1)
        kp2d_final = tf.concat([2.0 * (kp2d_flipped / self.config.ENCODER_INPUT_SHAPE[:2]) - 1.0, vis_final], axis=-1)
        # Preserving non_vis to be 0.
        kp2d_final = kp2d_final * vis_final

        # Normalize image to [-1, 1]
        image_final = tf.subtract(image_flipped, 0.5)
        image_final = tf.multiply(image_final, 2.0)

        return image_final, kp2d_final, kp3d_flipped, has_3d

    def _random_scale_image(self, image, kp2d, center):
        """Scale image with min and max scale defined in config
        Args:
           image:  [height, width, channel]
           kp2d:   [num_kp, 3], currently assumes 2d coco+ keypoints (num_kp=19)
           center: [x, y], center from which to scale the image
        """
        scale_min = self.config.SCALE_MIN
        scale_max = self.config.SCALE_MAX
        scale_factor = tf.random.uniform([1], minval=scale_min, maxval=scale_max, dtype=tf.float32)
        image_size = tf.cast(tf.shape(image)[:2], tf.float32)

        new_image_size = tf.cast(image_size * scale_factor, tf.int32)
        image_resize = tf.image.resize(image, new_image_size)
        actual_factor = tf.cast(tf.shape(image_resize)[:2], tf.float32) / image_size

        kp2s_x = kp2d[:, 0] * actual_factor[1]
        kp2s_y = kp2d[:, 1] * actual_factor[0]
        new_kp2d = tf.stack([kp2s_x, kp2s_y], axis=1)

        center_x = tf.cast(center[0] * actual_factor[1], tf.int32)
        center_y = tf.cast(center[1] * actual_factor[0], tf.int32)
        new_center = tf.stack([center_x, center_y])

        return image_resize, new_kp2d, new_center

    def _random_transform_image(self, kp2d, vis):
        """Transform center and based on 2d keypoints and trans max defined in config
        Args:
            kp2d: [num_kp, 3], currently assumes 2d coco+ keypoints (num_kp=19)
            vis:  [num_kp,], valued between [0, 1]
        """
        min_pt = tf.reduce_min(tf.boolean_mask(kp2d[:, :2], vis), axis=0)
        max_pt = tf.reduce_max(tf.boolean_mask(kp2d[:, :2], vis), axis=0)
        center = (min_pt + max_pt) / 2.
        trans_max = self.config.TRANS_MAX
        rand_trans = tf.random.uniform([2], minval=-trans_max, maxval=trans_max, dtype=tf.float32)
        center = center + rand_trans
        return center

    def _random_flip_image(self, image, kp2d, vis, kp3d):
        """Flipping image and keypoints
        Args:
            image: [height, width, channel]
            kp2d:  [num_kp, 2], currently assumes 2d coco+ keypoints (num_kp=19)
            vis:   [num_kp,], valued between [0, 1]
            kp3d:  [num_kp, 3], currently assumes 3d LSP keypoints (num_kp=14)
        """
        rand_flip = tf.random.uniform([], 0, 1.0)
        should_flip = tf.less(rand_flip, .5)

        def flip(_image, _kp2d, _vis, _kp3d):
            image_flipped = tf.image.flip_left_right(_image)

            image_width = tf.cast(tf.shape(image_flipped)[0], dtype=_kp2d.dtype)
            kp2d_x = image_width - _kp2d[:, 0] - 1
            kp2d_y = _kp2d[:, 1]
            kp2d_flipped = tf.stack([kp2d_x, kp2d_y], -1)
            kp2d_flipped = tf.gather(kp2d_flipped, self.flip_ids_kp2d)
            vis_flipped = tf.gather(_vis, self.flip_ids_kp2d)

            kp3d_flipped = tf.gather(_kp3d, self.flip_ids_kp3d)
            flip_mat = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
            kp3d_flipped = tf.transpose(tf.matmul(flip_mat, kp3d_flipped, transpose_b=True))
            kp3d_flipped = kp3d_flipped - tf.reduce_mean(kp3d_flipped, axis=0)

            return image_flipped, kp2d_flipped, vis_flipped, kp3d_flipped

        return tf.cond(should_flip, lambda: flip(image, kp2d, vis, kp3d), lambda: (image, kp2d, vis, kp3d))

    def _center_crop_image(self, image, kp2d, center):
        """Crop image to the input size of the specified encoder backbone net defined in config
        Args:
            image:  [height, width, channel]
            kp2d:   [num_kp, 3], currently assumes 2d coco+ keypoints (num_kp=19)
            center: [x, y], center from which to crop image
        """
        center = tf.squeeze(center)
        bbox_begin = tf.stack([center[1], center[0], 0])
        bbox_size = tf.stack(self.config.ENCODER_INPUT_SHAPE)
        image_crop = tf.slice(image, bbox_begin, bbox_size)
        x_crop = kp2d[:, 0] - tf.cast(center[0], tf.float32)
        y_crop = kp2d[:, 1] - tf.cast(center[1], tf.float32)
        kp2d_crop = tf.stack([x_crop, y_crop], axis=1)
        return image_crop, kp2d_crop

    def _pad_image(self, image, kp2d, center):
        """Pad image with safe margin
        Args:
            image:  [height, width, channel]
            kp2d:   [num_kp, 3], currently assumes 2d coco+ keypoints (num_kp=19)
            center: [x, y], center from which to pad image by safe margin
        """
        margin = tf.cast(self.config.ENCODER_INPUT_SHAPE[0] / 2, tf.int32)
        margin_safe = margin + self.config.TRANS_MAX + 50  # Extra 50 for safety.

        def repeat_col(col, num_repeat):
            # col is N x 3, ravels
            # i.e. to N*3 and repeats, then put it back to num_repeat x N x 3
            return tf.reshape(tf.tile(tf.reshape(col, [-1]), [num_repeat]), [num_repeat, -1, 3])

        top = repeat_col(image[0, :, :], margin_safe)
        bottom = repeat_col(image[-1, :, :], margin_safe)
        image_pad = tf.concat([top, image, bottom], 0)
        # Left requires another permute bc how img[:, 0, :]->(h, 3)
        left = tf.transpose(repeat_col(image_pad[:, 0, :], margin_safe), perm=[1, 0, 2])
        right = tf.transpose(repeat_col(image_pad[:, -1, :], margin_safe), perm=[1, 0, 2])
        image_pad = tf.concat([left, image_pad, right], 1)

        kp2d_pad = kp2d + tf.cast(margin_safe, tf.float32)
        center_pad = center + margin_safe
        center_pad = center_pad - margin
        return image_pad, kp2d_pad, center_pad

    ############################################################
    #  SMPL Dataset Loader
    ############################################################

    def get_smpl(self):
        start = time()
        print('initialize smpl dataset...')
        smpl_dataset = self.create_dataset('train', self._parse_smpl,
                                           data_dir=self.config.SMPL_DATA_DIR,
                                           datasets=self.config.SMPL_DATASETS)
        smpl_dataset = smpl_dataset.shuffle(10000, seed=self.config.SEED, reshuffle_each_iteration=True)
        # keep batch size * iterations as discriminator runs over all predictions from IF loop
        smpl_dataset = smpl_dataset.batch(self.config.BATCH_SIZE * self.config.ITERATIONS, drop_remainder=True)
        smpl_dataset = smpl_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        smpl_dataset = smpl_dataset.repeat()

        print('Done (t={})\n'.format(time() - start))
        return smpl_dataset

    def _parse_smpl(self, example_proto):
        feature_map = {
            'pose': tf.io.VarLenFeature(dtype=tf.float32),
            'shape': tf.io.VarLenFeature(dtype=tf.float32),
        }
        features = tf.io.parse_single_example(example_proto, feature_map)
        pose = tf.reshape(tf.sparse.to_dense(features['pose']), (self.config.NUM_POSE_PARAMS,))
        shape = tf.reshape(tf.sparse.to_dense(features['shape']), (self.config.NUM_SHAPE_PARAMS,))

        return tf.concat([pose, shape], axis=-1)

    ############################################################
    #  Test Dataset Loader
    ############################################################

    def get_test(self):
        start = time()
        print('initialize test dataset...')
        tf_record_dirs = [join(self.config.DATA_DIR, dataset, '*_test.tfrecord') for dataset in self.config.DATASETS]
        tf_records = [tf_record for tf_records in sorted([glob(f) for f in tf_record_dirs]) for tf_record in tf_records]

        drop_remainder = False
        if self.config.NUM_TEST_SAMPLES % self.config.BATCH_SIZE > 0:
            drop_remainder = True

        test_dataset = tf.data.TFRecordDataset(tf_records, num_parallel_reads=self.config.NUM_PARALLEL * 2) \
            .map(self._parse_test, num_parallel_calls=self.config.NUM_PARALLEL * 2) \
            .map(self._convert_and_scale_test, num_parallel_calls=self.config.NUM_PARALLEL * 2) \
            .batch(self.config.BATCH_SIZE, drop_remainder=drop_remainder) \
            .prefetch(self.config.NUM_PARALLEL * 2)

        print('Done (t={})\n'.format(time() - start))
        return test_dataset

    def _parse_test(self, example_proto):
        feature_map = {
            'image_raw': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            'keypoints_3d': tf.io.VarLenFeature(dtype=tf.float32),
            'sequence': tf.io.FixedLenFeature([], dtype=tf.string)
        }
        features = tf.io.parse_single_example(example_proto, feature_map)
        image_data = features['image_raw']
        kp3d = tf.reshape(tf.sparse.to_dense(features['keypoints_3d']), (self.config.NUM_KP3D, 3))
        sequence = features['sequence']

        return image_data, kp3d, sequence

    def _convert_and_scale_test(self, image_data, kp3d, sequence):
        image = tf.image.decode_jpeg(image_data, channels=3)
        # helpers.show_image(image.numpy(), kp2d.numpy()[:, :2], vis.numpy())

        # convert to [0, 1].
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        encoder_img_size = self.config.ENCODER_INPUT_SHAPE[:2]
        image_resize = tf.image.resize(image, encoder_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Normalize image to [-1, 1]
        image_final = tf.subtract(image_resize, 0.5)
        image_final = tf.multiply(image_final, 2.0)

        return image_final, kp3d, sequence

    ############################################################
    #  Inference Dataset Loader
    ############################################################

    def get_data_for(self, example):
        if not isinstance(example, list):
            example = [example]

        return tf.data.TFRecordDataset(example).map(self._parse_inference).map(self._convert_and_scale_all)

    def _parse_inference(self, example_proto):
        feature_map = {
            'image_raw': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            'keypoints_2d': tf.io.VarLenFeature(dtype=tf.float32),
            'keypoints_3d': tf.io.VarLenFeature(dtype=tf.float32),
            'has_3d': tf.io.FixedLenFeature([], dtype=tf.int64),
            'sequence': tf.io.FixedLenFeature([], dtype=tf.string, default_value='train')
        }
        features = tf.io.parse_single_example(example_proto, feature_map)

        image_data = features['image_raw']
        kp2d = tf.reshape(tf.sparse.to_dense(features['keypoints_2d']), (self.config.NUM_KP2D, 3))
        kp3d = tf.reshape(tf.sparse.to_dense(features['keypoints_3d']), (self.config.NUM_KP3D, 3))
        has_3d = features['has_3d']
        sequence = features['sequence']

        return image_data, kp2d, kp3d, has_3d, sequence

    def _convert_and_scale_all(self, image_data, kp2d, kp3d, has_3d, sequence):
        image_final, kp2d_final, kp3d, has_3d = self._convert_and_scale(image_data, kp2d, kp3d, has_3d)
        return image_final, kp2d_final, kp3d, has_3d, sequence

    ############################################################
    #  Helper
    ############################################################

    def create_dataset(self, ds_type, parse_func, map_func=None, data_dir=None, datasets=None):
        if data_dir is None:
            data_dir = self.config.DATA_DIR
        if datasets is None:
            datasets = self.config.DATASETS

        tf_record_dirs = [join(data_dir, dataset, '*_{}.tfrecord'.format(ds_type)) for dataset in datasets]
        tf_records = [tf_record for tf_records in sorted([glob(f) for f in tf_record_dirs]) for tf_record in tf_records]

        dataset = tf.data.Dataset.from_tensor_slices(tf_records)
        dataset = dataset.shuffle(len(tf_records))
        dataset = dataset.interleave(map_func=lambda record: tf.data.TFRecordDataset(record),
                                     cycle_length=self.config.BATCH_SIZE * self.config.NUM_PARALLEL,
                                     block_length=self.config.BATCH_SIZE,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE
                                     )
        dataset = dataset.map(parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if map_func is not None:
            dataset = dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset
