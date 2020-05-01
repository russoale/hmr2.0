import sys

import os

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from main.discriminator import ShapeDiscriminator, CommonPoseDiscriminator, SingleJointDiscriminator, \
    FullPoseDiscriminator, Discriminator
from main.local import LocalConfig


class TestDiscriminator(tf.test.TestCase):

    def setUp(self):
        super(TestDiscriminator, self).setUp()
        self.config = LocalConfig()
        self.batch_size = self.config.BATCH_SIZE
        self.num_kp = self.config.NUM_JOINTS

    def test_shape_discriminator(self):
        inputs = tf.ones((self.batch_size, 10))
        shape_dis = ShapeDiscriminator()
        outputs = shape_dis(inputs)
        mean = tf.reduce_mean(outputs)
        expected = np.array(-0.20134258, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual(outputs.shape, (self.batch_size, 1))

    def test_common_pose_discriminator(self):
        inputs = tf.ones((self.batch_size, self.num_kp, 9))
        c_pose_dis = CommonPoseDiscriminator()
        outputs = c_pose_dis(inputs)
        mean = tf.reduce_mean(outputs)
        expected = np.array(0.131552681, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_kp, 1, 32))

    def test_single_joint_discriminator(self):
        inputs = tf.ones((self.batch_size, self.num_kp, 1, 32))
        s_joint_dis = SingleJointDiscriminator()
        outputs = s_joint_dis(inputs)
        mean = tf.reduce_mean(outputs)
        expected = np.array(0.4741633, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_kp))

    def test_full_pose_discriminator(self):
        inputs = tf.ones((self.batch_size, self.num_kp, 1, 32))
        full_pose_dis = FullPoseDiscriminator()
        outputs = full_pose_dis(inputs)
        mean = tf.reduce_mean(outputs)
        expected = np.array(0.2649544, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual(outputs.shape, (self.batch_size, 1))

    def test_discriminator(self):
        inputs = tf.ones((self.batch_size, (self.config.NUM_JOINTS * 9 + self.config.NUM_SHAPE_PARAMS)))
        dis = Discriminator()
        outputs = dis(inputs)
        mean = tf.reduce_mean(outputs)
        expected = np.array(0.19671753, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual(outputs.shape, (self.batch_size, 25))


def create_info():
    discriminator = Discriminator()
    config = discriminator.config
    inputs = tf.ones((2, (config.NUM_JOINTS * 9 + config.NUM_SHAPE_PARAMS)))
    _ = discriminator(inputs)
    discriminator.summary()
    print('\n\n\n')
    discriminator.common_pose_discriminator.summary()
    print('\n\n\n')
    discriminator.single_joint_discriminator.summary()
    print('\n\n\n')
    discriminator.full_pose_discriminator.summary()
    print('\n\n\n')
    discriminator.shape_discriminator.summary()

    tf.keras.utils.plot_model(discriminator, to_file='discriminator.png')


if __name__ == '__main__':
    tf.test.main()
    # create_info()
