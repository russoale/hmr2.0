import os
import sys

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from main.local import LocalConfig
from main.smpl import Smpl


class TestSmpl(tf.test.TestCase):

    def setUp(self):
        super(TestSmpl, self).setUp()
        self.config = LocalConfig()
        self.inputs = tf.ones((self.config.BATCH_SIZE, (self.config.NUM_POSE_PARAMS + self.config.NUM_SHAPE_PARAMS)))

    def test_smpl_loaded_correctly(self):
        smpl = Smpl()
        self.assertEqual((6890, 3), smpl.vertices_template.shape)
        self.assertEqual((10, 20670), smpl.shapes.shape)
        self.assertEqual((6890, 24), smpl.smpl_joint_regressor.shape)
        self.assertEqual((207, 20670), smpl.pose.shape)
        self.assertEqual((6890, 24), smpl.lbs_weights.shape)
        self.assertEqual((6890, 19), smpl.joint_regressor.shape)

    def test_smpl_loaded_correctly_with_custom_regressors(self):
        self.config.JOINT_TYPE = 'custom'
        smpl = Smpl()
        self.assertEqual((6890, 20), smpl.joint_regressor.shape)
        # set back! important due to singleton pattern of config
        self.config.JOINT_TYPE = 'cocoplus'

    def test_smpl_output(self):
        smpl = Smpl()
        outputs = smpl(self.inputs)

        self.assertEqual(3, len(outputs))
        self.assertEqual((2, 6890, 3), outputs[0].shape)
        self.assertEqual((2, 19, 3), outputs[1].shape)
        self.assertEqual((2, 24, 3, 3), outputs[2].shape)


if __name__ == '__main__':
    tf.test.main()
