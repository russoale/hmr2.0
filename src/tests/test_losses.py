import sys

import os

# to make run from console for module import
from tests._helper import get_kp2d, get_kp3d

sys.path.append(os.path.abspath('..'))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import numpy as np
from tests._config import TestConfig
from main.losses import batch_kp2d_l1_loss, batch_kp3d_l2_loss, batch_pose_l2_loss, batch_shape_l2_loss, \
    batch_generator_disc_l2_loss, batch_disc_l2_loss, mean_per_joint_position_error_2d, \
    batch_mean_mpjpe_3d, batch_mean_mpjpe_3d_aligned


class TestLosses(tf.test.TestCase):

    def setUp(self):
        super(TestLosses, self).setUp()
        self.config = TestConfig()

    def test_batch_kp2d_l1_loss(self):
        pred2d, real2d = get_kp2d(self.config)

        result = batch_kp2d_l1_loss(real2d, pred2d)
        expected = np.array(0.03604419, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_batch_kp3d_l2_loss(self):
        pred_3d, real_3d = get_kp3d(self.config)
        has3d = np.array([[0.], [1.]], dtype=np.float32)

        result = batch_kp3d_l2_loss(real_3d, pred_3d, has3d)
        expected = np.array(0.0010463116, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_batch_pose_l2_loss(self):
        real_pose = np.array([[-9.44920200e+01, -4.25263865e+01, -1.30050643e+01, -2.79970490e-01,
                               3.24995661e-01, 5.03083125e-01, -6.90573755e-01, -4.12994214e-01,
                               -4.21870093e-01, 5.98717416e-01, -1.48420885e-02, -3.85911139e-02,
                               1.13642605e-01, 2.30647176e-01, -2.11843286e-01, 1.31767149e+00,
                               -6.61596447e-01, 4.02174644e-01, 3.03129424e-02, 5.91100770e-02,
                               -8.04416564e-02, -1.12944653e-01, 3.15045050e-01, -1.32838375e-01,
                               -1.33748209e-01, -4.99408923e-01, 1.40508643e-01, 6.10867911e-02,
                               -2.22951915e-02, -4.73448564e-02, -1.48489055e-01, 1.47620442e-01,
                               3.24157346e-01, 7.78414851e-04, 1.70687935e-01, -1.54716815e-01,
                               2.95053507e-01, -2.91967776e-01, 1.26000780e-01, 8.09572677e-02,
                               1.54710846e-02, -4.21941758e-01, 7.44124075e-02, 1.17146423e-01,
                               3.16305389e-01, 5.04810448e-01, -3.65526364e-01, 1.31366428e-01,
                               -2.76658949e-02, -9.17315987e-03, -1.88285742e-01, 7.86409877e-03,
                               -9.41106758e-02, 2.08424367e-01, 1.62278709e-01, -7.98170265e-01,
                               -3.97403587e-03, 1.11321421e-01, 6.07793270e-01, 1.42215980e-01,
                               4.48185010e-01, -1.38429048e-01, 3.77056061e-02, 4.48877661e-01,
                               1.31445158e-01, 5.07427503e-02, -3.80920772e-01, -2.52292254e-02,
                               -5.27745375e-02, -7.43903887e-02, 7.22498075e-02, -6.35824487e-03]])
        real_pose = np.tile(real_pose, self.config.BATCH_SIZE).reshape((-1, 72))
        real_pose = tf.convert_to_tensor(real_pose, tf.float32)
        pred_pose = tf.ones((self.config.BATCH_SIZE, 72), tf.float32)
        has_smpl = tf.ones((self.config.BATCH_SIZE, 1), tf.float32)

        result = batch_pose_l2_loss(real_pose, pred_pose, has_smpl)
        expected = np.array(0.513418972, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_batch_shape_l2_loss(self):
        real_shape = np.array([[-3.54196257, 0.90870435, -1.0978663, -0.20436199, 0.18589762,
                                0.55789026, -0.18163599, 0.12002746, -0.09172286, 0.4430783]])
        real_shape = np.tile(real_shape, self.config.BATCH_SIZE).reshape((-1, 10))
        real_shape = tf.convert_to_tensor(real_shape, tf.float32)
        pred_shape = tf.ones((self.config.BATCH_SIZE, 10), tf.float32)
        has_smpl = tf.ones((self.config.BATCH_SIZE, 1), tf.float32)

        result = batch_shape_l2_loss(real_shape, pred_shape, has_smpl)
        expected = np.array(3.102015018, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_batch_encoder_disc_l2_loss(self):
        disc_output_generator = tf.tile(tf.range(85), (self.config.BATCH_SIZE,))
        disc_output_generator = tf.reshape(disc_output_generator, (-1, 85))

        result = batch_generator_disc_l2_loss(disc_output_generator)
        expected = np.array(194055, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_batch_disc_l2_loss(self):
        disc_output_generator = tf.tile(tf.range(85, dtype=tf.float32), (self.config.BATCH_SIZE,))
        disc_output_generator = tf.reshape(disc_output_generator, (-1, 85))

        result = batch_disc_l2_loss(disc_output_generator, disc_output_generator)[2]
        expected = np.array(395165, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_mean_per_joint_position_error_2d(self):
        pred2d, real2d = get_kp2d(self.config)

        result = mean_per_joint_position_error_2d(real2d, pred2d)
        expected = np.array(0.06046052, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_mean_per_joint_position_error_3d(self):
        pred_3d, real_3d = get_kp3d(self.config)

        result = batch_mean_mpjpe_3d(real_3d, pred_3d)
        expected = np.array(0.10350359, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)

    def test_mean_per_joint_position_error_3d_aligned(self):
        pred_3d, real_3d = get_kp3d(self.config)

        result = batch_mean_mpjpe_3d_aligned(real_3d, pred_3d)
        expected = np.array(0.07440712, dtype=np.float32)
        self.assertAllCloseAccordingToType(expected, result)


if __name__ == '__main__':
    tf.test.main()
