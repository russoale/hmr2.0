import sys

import numpy as np
import os

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from main.model_util import batch_orthographic_projection, batch_skew_symmetric, batch_rodrigues, \
    batch_global_rigid_transformation, batch_align_by_pelvis, batch_compute_similarity_transform
from main.local import LocalConfig
from tests._helper import get_kp3d


class TestUtil(tf.test.TestCase):

    def setUp(self):
        super(TestUtil, self).setUp()
        self.config = LocalConfig()

    def test_batch_compute_similarity_transform(self):
        pred_3d, real_3d = get_kp3d(self.config)
        output = batch_compute_similarity_transform(real_3d, pred_3d)
        mean = tf.reduce_mean(output)
        expected = np.array(1.806683, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual((self.config.BATCH_SIZE, self.config.NUM_KP3D, 3), output.shape)

    def test_batch_align_by_pelvis(self):
        joints_3d = tf.ones((self.config.BATCH_SIZE, self.config.NUM_KP3D, 3))
        output = batch_align_by_pelvis(joints_3d)
        expected = tf.zeros((self.config.BATCH_SIZE, self.config.NUM_KP3D, 3))

        self.assertAllCloseAccordingToType(expected, output)
        self.assertEqual((self.config.BATCH_SIZE, self.config.NUM_KP3D, 3), output.shape)

    def test_batch_orthographic_projection(self):
        joints_3d = tf.ones((self.config.BATCH_SIZE, self.config.NUM_JOINTS, 3))
        camera = tf.ones(3)
        output = batch_orthographic_projection(joints_3d, camera)
        expected = tf.ones((self.config.BATCH_SIZE, self.config.NUM_JOINTS, 2)) + 1

        self.assertAllCloseAccordingToType(expected, output)
        self.assertEqual((self.config.BATCH_SIZE, self.config.NUM_JOINTS, 2), output.shape)

    def test_batch_skew_symmetric(self):
        inputs = tf.ones((self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 3))
        output = batch_skew_symmetric(inputs)
        skew = np.array([[[[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]]]], dtype=np.float32)
        expected = tf.tile(skew, (self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 1, 1))

        self.assertAllCloseAccordingToType(expected, output)
        self.assertEqual((self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 3, 3), output.shape)

    def test_batch_rodrigues(self):
        inputs = tf.ones((self.config.BATCH_SIZE, self.config.NUM_POSE_PARAMS))
        output = batch_rodrigues(inputs)
        rotation = np.array([[[0.22629566, -0.18300793, 0.95671225, 0.95671225, 0.22629566,
                               -0.18300793, - 0.18300793, 0.95671225, 0.22629566]]], dtype=np.float32)
        expected = tf.tile(rotation, (self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 1))

        self.assertAllCloseAccordingToType(expected, output)
        self.assertEqual((self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 9), output.shape)

    def test_batch_global_rigid_transformation___rotate_base_false(self):
        log = self.__test_batch_global_rigid_transformation(False)
        exp_rel_joints = np.array(7.143234, dtype=np.float32)
        self.assertAllCloseAccordingToType(exp_rel_joints, log)

    def test_batch_global_rigid_transformation___rotate_base_true(self):
        log = self.__test_batch_global_rigid_transformation(True)
        exp_rel_joints = np.array(7.671055, dtype=np.float32)
        self.assertAllCloseAccordingToType(exp_rel_joints, log)

    def __test_batch_global_rigid_transformation(self, rotate_base):
        rotations = tf.tile(np.array([[[[.1, .9, .1]]]], dtype=np.float32),
                            (self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 3, 1))
        joints = tf.ones((self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 3))
        ancestors = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
        new_joints, rel_joints = batch_global_rigid_transformation(rotations, joints, ancestors, rotate_base)

        exp_new_joints = tf.ones((self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 3))
        self.assertAllCloseAccordingToType(exp_new_joints, new_joints)
        self.assertEqual((self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 3), new_joints.shape)

        shape = rel_joints.shape
        log = tf.reduce_logsumexp(rel_joints)
        self.assertEqual((self.config.BATCH_SIZE, self.config.NUM_JOINTS_GLOBAL, 4, 4), shape)

        return log


if __name__ == '__main__':
    tf.test.main()
