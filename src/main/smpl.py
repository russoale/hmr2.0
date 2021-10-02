import pickle
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from main.config import Config
from main.model_util import batch_rodrigues, batch_global_rigid_transformation


class Smpl(layers.Layer):
    """smpl layer for models generation"""

    def __init__(self):
        super(Smpl, self).__init__()

        self.config = Config()
        if self.config.JOINT_TYPE not in ['cocoplus', 'lsp', 'custom']:
            raise Exception('unknow joint type: {}, it must be either cocoplus or lsp'.format(self.config.JOINT_TYPE))

        with open(self.config.SMPL_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        def tf_variable(value, name):
            converted = tf.convert_to_tensor(value=value, dtype=tf.float32)
            return tf.Variable(converted, name=name, trainable=False)

        # Mean template vertices: [6890 x 3]
        self.vertices_template = tf_variable(model['v_template'], name='vertices_template')

        # Shape blend shape basis: [6980 x 3 x 10]
        self.shapes = tf_variable(model['shapedirs'], name='shapes')
        self.num_betas = self.shapes.shape[-1]
        # [6980 x 3 x 10] -> [10 x (6980 * 3)]
        self.shapes = tf.transpose(tf.reshape(self.shapes, [-1, self.num_betas]))

        # Regressor for joint locations given [6890 x 24]
        self.smpl_joint_regressor = tf_variable(model['J_regressor'].T, name='smpl_joint_regressor')

        # Pose blend shape basis: [6890 x 3 x 207]
        self.pose = tf_variable(model['posedirs'], name='pose')
        # [(6890 * 3) x 207] -> [207 x (6890 * 3)]
        self.pose = tf.transpose(tf.reshape(self.pose, [-1, self.pose.shape[-1]]))

        # LBS weights: [6890 x 24]
        self.lbs_weights = tf_variable(model['weights'], name='lbs_weights')

        # load face vertices for rendering
        self.faces = tf.convert_to_tensor(model['f'], dtype=tf.float32)

        # This returns 19 coco keypoints: [6890 x 19]
        # if JOINT_TYPE == 'custom' this adds additional regressors given
        # the generated .npy files by keypoint maker
        self.joint_regressor = model['cocoplus_regressor']
        if self.config.JOINT_TYPE == 'custom':
            if len(self.config.CUSTOM_REGRESSOR_IDX) > 0:
                for index, file_name in self.config.CUSTOM_REGRESSOR_IDX.items():
                    file = join(self.config.CUSTOM_REGRESSOR_PATH, file_name)
                    regressor = np.load(file)
                    self.joint_regressor = np.insert(self.joint_regressor, index, np.squeeze(regressor), 0)
        else:
            if self.config.INITIALIZE_CUSTOM_REGRESSOR:
                self.joint_regressor_plus = tf.identity(self.joint_regressor)
                for index, file_name in self.config.CUSTOM_REGRESSOR_IDX.items():
                    file = join(self.config.CUSTOM_REGRESSOR_PATH, file_name)
                    regressor = np.load(file).astype(np.float32)
                    self.joint_regressor_plus = np.insert(self.joint_regressor_plus, index, np.squeeze(regressor), 0)

                self.joint_regressor_plus = tf_variable(self.joint_regressor_plus.T, name='joint_regressor_plus')

        self.joint_regressor = tf_variable(self.joint_regressor.T, name='joint_regressor')
        if self.config.JOINT_TYPE == 'lsp':  # 14 LSP joints!
            self.joint_regressor = self.joint_regressor[:, :14]

        self.ancestors = model['kintree_table'][0].astype(np.int32)
        self.identity = tf.eye(3)
        self.joint_transformed = None

    def call(self, inputs, **kwargs):
        """Obtain SMPL with pose (theta, with 3-D axis-angle rep) & shape (beta) inputs.
           Theta includes the global rotation.
        Args:
            inputs: [batch x 82] with pose = [batch, :72] and shape = [batch, 72:]
        Updates:
            self.joint_transformed: [batch x 24 x 3] joint location after shaping
                                                    & posing with beta and theta
        Returns:
            vertices: [batch x 6980 x 3]
            joints: [batch x (19 || 14) x 3] joint locations, depending on joint_type
            rotations: [batch x 24 x 3 x 3] rotation matrices by theta
        """
        _batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        _pose = inputs[:, :self.config.NUM_POSE_PARAMS]
        _shape = inputs[:, -self.config.NUM_SHAPE_PARAMS:]
        _reshape = [_batch_size, self.vertices_template.shape[0], self.vertices_template.shape[1]]  # [batch x 6890 x 3]

        # 1. Add shape blend shapes
        # [batch x 10] * [10 x (6890 * 3)] = [batch x 6890 x 3]
        v_shaped = tf.matmul(_shape, self.shapes)
        v_shaped = tf.reshape(v_shaped, _reshape) + self.vertices_template

        # 2. Infer shape-dependent smpl joint locations
        v_joints = self.compute_joints(v_shaped, self.smpl_joint_regressor)

        # 3. Add pose blend shapes
        # [batch x 24 x 3 x 3]
        rotations = tf.reshape(batch_rodrigues(_pose), [_batch_size, self.config.NUM_JOINTS_GLOBAL, 3, 3])
        # Ignore global rotation [batch x 23 x 3 x 3] -> [batch, 207]
        pose_feature = tf.reshape(rotations[:, 1:, :, :] - self.identity, [_batch_size, -1])
        # [batch, 207] x [207, 20670] -> N x 6890 x 3
        v_posed = tf.reshape(tf.matmul(pose_feature, self.pose), _reshape) + v_shaped

        # 4. Get the global joint location
        self.joint_transformed, rel_joints = batch_global_rigid_transformation(rotations, v_joints, self.ancestors)

        # 5. Do skinning:
        # [batch x 6890 x 24]
        weights = tf.reshape(tf.tile(self.lbs_weights, [_batch_size, 1]), [_batch_size, -1, 24])

        # compute vertices by homogeneous joint coordinates lbs weights
        # [batch x 6890 x 24] x [batch x 24 x 16] -> [batch x 6890 x 4 x 4]
        rel_joints = tf.reshape(rel_joints, [_batch_size, self.config.NUM_JOINTS_GLOBAL, 16])
        weighted_joints = tf.reshape(tf.matmul(weights, rel_joints), [_batch_size, -1, 4, 4])

        # -> [batch x 6890 x 4 x 1]
        ones = tf.ones([_batch_size, v_posed.shape[1], 1])
        v_posed_homo = tf.expand_dims(tf.concat([v_posed, ones], 2), -1)
        v_posed_homo = tf.matmul(weighted_joints, v_posed_homo)

        # [batch x 6890 x 3 x 1]
        vertices = v_posed_homo[:, :, :3, 0]

        # Get final coco or lsp or custom joints:
        if self.config.JOINT_TYPE != 'custom' and self.config.INITIALIZE_CUSTOM_REGRESSOR:
            joints = self.compute_joints(vertices, self.joint_regressor_plus)
        else:
            joints = self.compute_joints(vertices, self.joint_regressor)

        return vertices, joints, rotations

    def compute_output_shape(self, input_shape):
        return (None, 6890, 3), (None, self.config.NUM_KP2D, 3), (None, self.config.NUM_JOINTS_GLOBAL, 3, 3)

    def compute_joints(self, vertices, regressor):
        """computes joint location from vertices by regressor
        Args:
            vertices:  [batch x 6890 x 3] smpl vertices
            regressor: [6890 x J] with J regressor specific joint sets
        Returns:
            joints: [batch x J x 3] joint locations
        """
        joint_x = tf.matmul(vertices[:, :, 0], regressor)
        joint_y = tf.matmul(vertices[:, :, 1], regressor)
        joint_z = tf.matmul(vertices[:, :, 2], regressor)
        return tf.stack([joint_x, joint_y, joint_z], axis=2)

    def get_faces(self):
        return self.faces

    def save_obj(self, _vertices, file_name):
        """saves the smpl models vertices as '.obj' to current dir.
            Resulting file can be rendered with common render tools, e.g. blender
        Args:
            _vertices: [batch x 6890 x 3 x 1]
            file_name: string
        """
        file = './{}.obj'.format(file_name)
        with open(file, 'w') as fp:
            for v in _vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
