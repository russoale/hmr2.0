import os
import sys

# to make run from console for module import
import pytest

sys.path.append(os.path.abspath('..'))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from main.dataset import Dataset
from main.local import LocalConfig


class TestDataset(tf.test.TestCase):

    @pytest.mark.xfail(strict=False)
    def test_dataset_train(self):
        config = LocalConfig()
        dataset = Dataset().get_train()
        for batch in dataset.take(1):
            image = ((batch[0].numpy()[0, :, :, :] + 1) / 2 * 255).astype(np.int32)
            output = np.sum(image)
            expected = np.array(6991299, dtype=np.int32)
            self.assertAllCloseAccordingToType(expected, output)  # this is flaky due to jitter of train dataset
            self.assertEqual(config.ENCODER_INPUT_SHAPE, image.shape)

            kp2d = ((batch[1].numpy()[0, :, :2] + 1) / 2 * image.shape[:2]).astype(np.int32)
            output = np.sum(kp2d)
            expected = np.array(3818, dtype=np.int32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual((config.NUM_KP2D, 2), kp2d.shape)

            vis = batch[1].numpy()[0, :, 2].astype(np.int32)
            output = np.sum(vis)
            expected = np.array(17, dtype=np.int32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual((config.NUM_KP2D,), vis.shape)

            kp3d = batch[2].numpy()[0, :, :]
            output = np.sum(kp3d)
            expected = np.array(4.11272e-06, dtype=np.float32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual((config.NUM_KP3D, 3), kp3d.shape)

            # check if has3d flag is included in correct shape
            has3d = batch[3].numpy()[0]
            self.assertEqual(tf.constant(1, tf.int64), has3d)
            self.assertEqual((config.BATCH_SIZE,), batch[3].shape)

    def test_dataset_val(self):
        config = LocalConfig()
        dataset = Dataset().get_val()
        for batch in dataset.take(1):
            image = ((batch[0].numpy()[0, :, :, :] + 1) / 2 * 255).astype(np.int32)
            output = np.sum(image)
            expected = np.array(8123856, dtype=np.int32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual(config.ENCODER_INPUT_SHAPE, image.shape)

            kp2d = ((batch[1].numpy()[0, :, :2] + 1) / 2 * image.shape[:2]).astype(np.int32)
            output = np.sum(kp2d)
            expected = np.array(4040, dtype=np.int32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual((config.NUM_KP2D, 2), kp2d.shape)

            vis = batch[1].numpy()[0, :, 2].astype(np.int32)
            output = np.sum(vis)
            expected = np.array(17, dtype=np.int32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual((config.NUM_KP2D,), vis.shape)

            kp3d = batch[2].numpy()[0, :, :]
            output = np.sum(kp3d)
            expected = np.array(56.9217948, dtype=np.float32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual((config.NUM_KP3D, 3), kp3d.shape)

            # check if has3d flag is included in correct shape
            has3d = batch[3].numpy()[0]
            self.assertEqual(tf.constant(1, tf.int64), has3d)
            self.assertEqual((config.BATCH_SIZE,), batch[3].shape)

    def test_dataset_test(self):
        config = LocalConfig()
        dataset = Dataset().get_test()
        for batch in dataset.take(1):
            image = ((batch[0].numpy()[0, :, :, :] + 1) / 2 * 255).astype(np.int32)
            output = np.sum(image)
            expected = np.array(10050903, dtype=np.int32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual(config.ENCODER_INPUT_SHAPE, image.shape)

            kp3d = batch[1].numpy()[0, :, :]
            output = np.sum(kp3d)
            expected = np.array(38780.2031, dtype=np.float32)
            self.assertAllCloseAccordingToType(expected, output)
            self.assertEqual((config.NUM_KP3D, 3), kp3d.shape)

            # check if sequence flag is included in correct shape
            sequence = batch[2].numpy()[0].decode("utf-8")
            self.assertEqual('TS1', sequence)

    def test_dataset_smpl(self):
        config = LocalConfig()
        dataset = Dataset().get_smpl()
        for batch in dataset.take(1):
            shape = (config.BATCH_SIZE * config.ITERATIONS, (config.NUM_POSE_PARAMS + config.NUM_SHAPE_PARAMS))
            self.assertEqual(shape, batch.shape)

            pose = batch[0].numpy()[:config.NUM_POSE_PARAMS:]
            mean = tf.reduce_mean(pose)
            expected = np.array(0.0411809, dtype=np.float32)
            self.assertAllCloseAccordingToType(expected, mean)

            shape = batch[0].numpy()[-config.NUM_SHAPE_PARAMS:]
            mean = tf.reduce_mean(shape)
            expected = np.array(0.12554605, dtype=np.float32)
            self.assertAllCloseAccordingToType(expected, mean)


if __name__ == '__main__':
    tf.test.main()
