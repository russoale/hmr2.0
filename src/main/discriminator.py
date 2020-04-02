import tensorflow as tf
from tensorflow.keras import layers

from main.config import Config
from main.util import batch_rodrigues


class CommonPoseDiscriminator(tf.keras.Model):
    """ For pose, theta is first converted to K many 3 × 3 rotation matrices via the Rodrigues formula.
        Each rotation matrix is sent to a common embedding network of two fully connected layers with
        32 hidden neurons.
    """

    def __init__(self):
        super(CommonPoseDiscriminator, self).__init__(name='common_pose_discriminator')
        self.config = Config()

        self.conv_2d_one = layers.Conv2D(32, [1, 1], name='conv_2d_one')
        self.conv_2d_two = layers.Conv2D(32, [1, 1], name='conv_2d_two')

    # [batch x K x 9]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_JOINTS, 9)
        assert inputs.shape == shape, "input dimension must be of shape {} but got {}".format(shape, inputs.shape)

        x = tf.expand_dims(inputs, 2)  # to batch x K x 1 x 9 ('channels_last' default)
        x = self.conv_2d_one(x, **kwargs)
        x = self.conv_2d_two(x, **kwargs)
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

        self.joint_discriminators = []
        for i in range(self.config.NUM_JOINTS):
            self.joint_discriminators.append(layers.Dense(1, name="disc_j{}".format(i)))

    # [batch x K x 1 x 32]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_JOINTS, 1, 32)
        assert inputs.shape == shape, "input dimension must be of shape {} but got {}".format(shape, inputs.shape)

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

        input_dim = self.config.NUM_JOINTS * 32
        self.fc_one = layers.Dense(1024, activation='relu', name="full_pose_fc_0", input_dim=input_dim)
        self.fc_two = layers.Dense(1024, activation='relu', name="full_pose_fc_1")
        self.fc_out = layers.Dense(1, activation=None, name="full_pose_fc_out")

    # [batch x K x 1 x 32]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_JOINTS, 1, 32)
        assert inputs.shape == shape, "input dimension must be of shape {} but got {}".format(shape, inputs.shape)

        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        x = tf.reshape(inputs, [batch_size, -1])
        x = self.fc_one(x, **kwargs)
        x = self.fc_two(x, **kwargs)
        x = self.fc_out(x, **kwargs)
        return x  # [batch x 1]

    def compute_output_shape(self, input_shape):
        return None, 1


class ShapeDiscriminator(tf.keras.Model):
    def __init__(self):
        super(ShapeDiscriminator, self).__init__(name='shape_discriminator')
        self.config = Config()

        self.fc_one = layers.Dense(5, activation='relu', name="shape_fc_0", input_dim=self.config.NUM_SHAPE_PARAMS)
        self.fc_two = layers.Dense(1, activation=None, name="shape_fc_1")

    # [batch x beta]
    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, self.config.NUM_SHAPE_PARAMS)
        assert inputs.shape == shape, "input dimension must be of shape {} but got {}".format(shape, inputs.shape)

        x = self.fc_one(inputs, **kwargs)
        x = self.fc_two(x, **kwargs)
        return x  # [batch x 1]

    def compute_output_shape(self, input_shape):
        return None, 1


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__(name='discriminator')

        self.common_pose_discriminator = CommonPoseDiscriminator()
        self.single_joint_discriminator = SingleJointDiscriminator()
        self.full_pose_discriminator = FullPoseDiscriminator()
        self.shape_discriminator = ShapeDiscriminator()

    def call(self, inputs, **kwargs):
        config = Config()
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, config.NUM_SMPL_PARAMS)
        assert inputs.shape == shape, "input dimension must be of shape {} but got {}".format(shape, inputs.shape)

        # inputs batch x thetas (cams: 3, pose: 72, shape: 10)
        _ = inputs[:, :config.NUM_CAMERA_PARAMS]  # camera is never used
        poses = inputs[:, config.NUM_CAMERA_PARAMS:(config.NUM_CAMERA_PARAMS + config.NUM_POSE_PARAMS)]
        shapes = inputs[:, -config.NUM_SHAPE_PARAMS:]

        # compute rotations matrices for [batch x K x 9] - ignore global rotation
        batch_poses_rot_mat = batch_rodrigues(poses)[:, 1:, :]

        # compute common embedding features [batch x K x 1 x 32]
        common_pose_features = self.common_pose_discriminator(batch_poses_rot_mat, **kwargs)

        # compute joint specific discriminators [batch x K]
        single_joint_outputs = self.single_joint_discriminator(common_pose_features, **kwargs)

        # compute full pose discriminator [batch x 1]
        full_pose_outputs = self.full_pose_discriminator(common_pose_features, **kwargs)

        # compute shape discriminators [batch x 1]
        shape_outputs = self.shape_discriminator(shapes, **kwargs)

        # [batch x (K + 1 + 1)]
        return tf.concat((single_joint_outputs, full_pose_outputs, shape_outputs), 1)
