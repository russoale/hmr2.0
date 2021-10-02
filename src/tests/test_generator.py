import sys

import os

# to make run from console for module import
import pytest

sys.path.append(os.path.abspath(".."))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from main.generator import Generator, Regressor
from main.local import LocalConfig


class TestGenerator(tf.test.TestCase):

    def setUp(self):
        super(TestGenerator, self).setUp()
        self.batch_size = LocalConfig().BATCH_SIZE

    @pytest.mark.xfail(strict=False)
    def test_resnet(self):
        inputs = tf.ones((self.batch_size, 224, 224, 3))
        generator = Generator()
        outputs = generator.resnet50V2(inputs, training=True)
        mean = tf.reduce_mean(outputs)
        expected = np.array(0.1707771569, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual(outputs.shape, (self.batch_size, 2048))

    def test_regressor(self):
        inputs = tf.ones((self.batch_size, 2048))
        regressor = Regressor()
        outputs = regressor(inputs, training=True)
        mean = tf.reduce_mean(outputs)
        expected = np.array(0.0841677, dtype=np.float32)

        self.assertAllCloseAccordingToType(expected, mean)
        self.assertEqual(outputs.shape, (3, self.batch_size, 85))

    def test_generator(self):
        inputs = tf.ones((self.batch_size, 224, 224, 3))
        generator = Generator()
        outputs = generator(inputs)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0][0].shape, (self.batch_size, 6890, 3))
        self.assertEqual(outputs[0][1].shape, (self.batch_size, 19, 2))
        self.assertEqual(outputs[0][2].shape, (self.batch_size, 19, 3))
        self.assertEqual(outputs[0][3].shape, (self.batch_size, 24, 3, 3))
        self.assertEqual(outputs[0][4].shape, (self.batch_size, 10))
        self.assertEqual(outputs[0][5].shape, (self.batch_size, 3))


def create_info():
    inputs = tf.ones((64, 224, 224, 3))
    generator = Generator()
    _ = generator(inputs)
    generator.summary()
    print("\n\n\n")
    generator.resnet50V2.summary()
    print("\n\n\n")
    generator.regressor.summary()

    tf.keras.utils.plot_model(generator, to_file='generator.png')


if __name__ == '__main__':
    tf.test.main()
    # create_info()
