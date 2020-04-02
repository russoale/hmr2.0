import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

from main import util
from main.config import Config
from main.smpl import Smpl


class Regressor(tf.keras.Model):

    def __init__(self):
        super(Regressor, self).__init__(name='regressor')
        self.config = Config()

        self._build_model()

    def _build_model(self):
        self.mean_theta = tf.Variable(util.load_mean_theta(), name='mean_theta', trainable=True)

        # //@formatter:off
        self.fc_one = layers.Dense(1024, activation='relu', name='fc_0')
        self.dropout_one = layers.Dropout(0.5)
        self.fc_two = layers.Dense(1024, activation='relu', name='fc_1')
        self.dropout_two = layers.Dropout(0.5)
        variance_scaling = tf.initializers.VarianceScaling(.01, mode='fan_avg', distribution='uniform')
        self.fc_three = layers.Dense(85, activation=None, kernel_initializer=variance_scaling, name='fc_2')
        # //@formatter:on

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        batch_theta = self.mean_theta[:batch_size, :]

        thetas = tf.TensorArray(tf.float32, self.config.ITERATIONS)
        for i in range(self.config.ITERATIONS):
            total_inputs = tf.concat([inputs, batch_theta], axis=1)
            batch_theta = batch_theta + self._fc_blocks(total_inputs, **kwargs)
            thetas = thetas.write(i, batch_theta)

        return thetas.stack()

    def _fc_blocks(self, inputs, **kwargs):
        x = self.fc_one(inputs, **kwargs)
        x = self.dropout_one(x, **kwargs)
        x = self.fc_two(x, **kwargs)
        x = self.dropout_two(x, **kwargs)
        x = self.fc_three(x, **kwargs)
        return x


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__(name='generator')
        self.config = Config()

        input_shape = self.config.ENCODER_INPUT_SHAPE
        self.resnet50V2 = ResNet50V2(include_top=False, input_shape=input_shape, pooling='avg')
        self.regressor = Regressor()
        self.smpl = Smpl()

    def call(self, inputs, **kwargs):
        features = self.resnet50V2(inputs, **kwargs)
        thetas = self.regressor(features, **kwargs)

        outputs = []
        for i in range(self.config.ITERATIONS):
            theta = thetas[i, :]
            outputs.append(self._compute_output(theta, **kwargs))

        return outputs

    def _compute_output(self, theta, **kwargs):
        cam = theta[:, :self.config.NUM_CAMERA_PARAMS]
        pose_and_shape = theta[:, self.config.NUM_CAMERA_PARAMS:]
        vertices, joints_3d, rotations = self.smpl(pose_and_shape, **kwargs)
        joints_2d = util.batch_orthographic_projection(joints_3d, cam)

        return tf.tuple([theta, vertices, joints_2d, joints_3d, rotations])
