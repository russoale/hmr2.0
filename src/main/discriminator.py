import tensorflow as tf
from tensorflow.keras import layers

from main.config import Config


class CommonPoseDiscriminator(tf.keras.Model):
    """ For pose, theta is first converted to K many 3 × 3 rotation matrices via the Rodrigues formula.
        Each rotation matrix is sent to a common embedding network of two fully connected layers with
        32 hidden neurons.
    """

    def __init__(self):
        super(CommonPoseDiscriminator, self).__init__(name='common_pose_discriminator')
        self.config = Config()

        l2_regularizer = tf.keras.regularizers.l2(self.config.DISCRIMINATOR_WEIGHT_DECAY)
        conv_2d_params = {
            'filters': 32,
            'kernel_size': [1, 1],
            'padding': 'same',
            'data_format': 'channels_last',
            'kernel_regularizer': l2_regularizer
        }
        self.conv_2d_one = layers.Conv2D(**conv_2d_params, name='conv_2d_one')
        self.conv_2d_two = layers.Conv2D(**conv_2d_params, name='conv_2d_two')

    # [batch x K x 9]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_JOINTS, 9)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        x = tf.expand_dims(inputs, 2)  # to batch x K x 1 x 9 ('channels_last' default)
        x = self.conv_2d_one(x, **kwargs)
        x = tf.nn.relu(x)
        x = self.conv_2d_two(x, **kwargs)
        x = tf.nn.relu(x)
        return x  # to [batch x K x 1 x 32]

    def compute_output_shape(self, input_shape):
        return None, self.config.NUM_JOINTS, 1, 32


class SingleJointDiscriminator(tf.keras.Model):
    """The outputs of the common embedding network are sent to K different discriminators
        (single joint discriminators) that output 1-D values.
    """

    def __init__(self):
        super(SingleJointDiscriminator, self).__init__(name='single_joint_discriminator')
        self.config = Config()

        l2_regularizer = tf.keras.regularizers.l2(self.config.DISCRIMINATOR_WEIGHT_DECAY)
        self.joint_discriminators = []
        for i in range(self.config.NUM_JOINTS):
            self.joint_discriminators.append(layers.Dense(1, kernel_regularizer=l2_regularizer, name="fc_{}".format(i)))

    # [batch x K x 1 x 32]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_JOINTS, 1, 32)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        single_joint_outputs = []
        for i in range(self.config.NUM_JOINTS):
            single_joint_outputs.append(self.joint_discriminators[i](inputs[:, i, :, :], **kwargs))

        output = tf.squeeze(tf.stack(single_joint_outputs, 1))
        return output  # [batch x K]

    def compute_output_shape(self, input_shape):
        return None, self.config.NUM_JOINTS


class FullPoseDiscriminator(tf.keras.Model):

    def __init__(self):
        super(FullPoseDiscriminator, self).__init__(name='full_pose_discriminator')
        self.config = Config()

        l2_regularizer = tf.keras.regularizers.l2(self.config.DISCRIMINATOR_WEIGHT_DECAY)
        self.flatten = layers.Flatten()
        self.fc_one = layers.Dense(1024, kernel_regularizer=l2_regularizer, name="fc_0")
        self.fc_two = layers.Dense(1024, kernel_regularizer=l2_regularizer, name="fc_1")
        self.fc_out = layers.Dense(1, kernel_regularizer=l2_regularizer, name="fc_out")

    # [batch x K x 1 x 32]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_JOINTS, 1, 32)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        x = self.flatten(inputs)
        x = self.fc_one(x, **kwargs)
        x = tf.nn.relu(x)
        x = self.fc_two(x, **kwargs)
        x = tf.nn.relu(x)
        x = self.fc_out(x, **kwargs)
        return x  # [batch x 1]

    def compute_output_shape(self, input_shape):
        return None, 1


class ShapeDiscriminator(tf.keras.Model):
    def __init__(self):
        super(ShapeDiscriminator, self).__init__(name='shape_discriminator')
        self.config = Config()

        l2_regularizer = tf.keras.regularizers.l2(self.config.DISCRIMINATOR_WEIGHT_DECAY)
        self.fc_one = layers.Dense(10, kernel_regularizer=l2_regularizer, name="fc_0")
        self.fc_two = layers.Dense(5, kernel_regularizer=l2_regularizer, name="fc_1")
        self.fc_out = layers.Dense(1, kernel_regularizer=l2_regularizer, name="fc_out")

    # [batch x beta]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_SHAPE_PARAMS)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        x = self.fc_one(inputs, **kwargs)
        x = tf.nn.relu(x)
        x = self.fc_two(x, **kwargs)
        x = tf.nn.relu(x)
        x = self.fc_out(x, **kwargs)
        return x  # [batch x 1]

    def compute_output_shape(self, input_shape):
        return None, 1


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__(name='discriminator')
        self.config = Config()

        self.common_pose_discriminator = CommonPoseDiscriminator()
        self.single_joint_discriminator = SingleJointDiscriminator()
        self.full_pose_discriminator = FullPoseDiscriminator()
        self.shape_discriminator = ShapeDiscriminator()

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_JOINTS * 9 + self.config.NUM_SHAPE_PARAMS)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        # inputs batch x (pose: 207, shape: 10)
        poses = inputs[:, :self.config.NUM_JOINTS * 9]
        shapes = inputs[:, -self.config.NUM_SHAPE_PARAMS:]

        poses = tf.reshape(poses, [batch_size, self.config.NUM_JOINTS, 9])
        # compute common embedding features [batch x K x 1 x 32]
        common_pose_features = self.common_pose_discriminator(poses, **kwargs)

        # compute joint specific discriminators [batch x K]
        single_joint_outputs = self.single_joint_discriminator(common_pose_features, **kwargs)

        # compute full pose discriminator [batch x 1]
        full_pose_outputs = self.full_pose_discriminator(common_pose_features, **kwargs)

        # compute shape discriminators [batch x 1]
        shape_outputs = self.shape_discriminator(shapes, **kwargs)

        # [batch x (K + 1 + 1)]
        return tf.concat((single_joint_outputs, full_pose_outputs, shape_outputs), 1)
