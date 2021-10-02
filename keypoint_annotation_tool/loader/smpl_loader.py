import pickle
from glob import glob

import cv2
import numpy as np


class SmplLoader:

    def __init__(self):
        super(SmplLoader, self).__init__()

    def init_model(self, file_name):
        with open(file_name, "rb") as f:
            model = pickle.load(f)

        self.faces = model['f']
        self.v_template = model["v_template"]
        self.pose_dirs = model["posedirs"]
        self.shape_dirs = model["shapedirs"]
        self.j_regressor = model["J_regressor"]
        self.weights = model['weights']
        self.parent_id = model['kintree_table'][0].astype(np.int32)
        self.identity = np.eye(3)

        if 'cocoplus_regressor' in model:
            self.coco_regressor = model["cocoplus_regressor"]

        self.custom_regressor = None

    def init_custom_regressors(self, path):
        files = glob(path)
        if len(files) > 0:
            regressors = []
            for file in files:
                regressor = np.load(file)
                regressors.append(regressor)

            self.custom_regressor = np.concatenate(regressors, 1).T

    def load_vertices(self, pose=None, shape=None, trans=None):

        if pose is None:
            pose = np.zeros([len(self.parent_id), 3])

        if shape is None:
            shape = np.zeros(self.shape_dirs.shape[-1])

        if trans is None:
            trans = np.zeros([1, 3])

        v_shaped = self.shape_dirs.dot(shape) + self.v_template

        x = np.matmul(self.j_regressor, v_shaped[:, 0])
        y = np.matmul(self.j_regressor, v_shaped[:, 1])
        z = np.matmul(self.j_regressor, v_shaped[:, 2])
        joints = np.vstack((x, y, z)).T

        rotation = self.relative_rotation(pose)
        v_posed = v_shaped + self.pose_dirs.dot(rotation)

        joints = self.global_rigid_transform(pose, joints)
        joints = joints.dot(self.weights.T)

        rest_shape_h = np.vstack((v_posed.T, np.ones((1, v_posed.shape[0]))))

        verts = (joints[:, 0, :] * rest_shape_h[0, :].reshape((1, -1))
                 + joints[:, 1, :] * rest_shape_h[1, :].reshape((1, -1))
                 + joints[:, 2, :] * rest_shape_h[2, :].reshape((1, -1))
                 + joints[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))
                 ).T

        verts = verts[:, :3] + trans
        return verts

    def relative_rotation(self, pose):
        pose = pose[1:, :]  # ignore global rotation
        pose = [cv2.Rodrigues(p)[0] - self.identity for p in pose]
        return np.concatenate(pose).ravel()

    def global_rigid_transform(self, pose, joints):
        homogeneous = np.array([[0.0, 0.0, 0.0, 1.0]])
        zeros = np.zeros([4, 3])

        def rotate_joint(pose_vec, joint):
            rot_joint = np.hstack([cv2.Rodrigues(pose_vec)[0], joint.reshape([3, 1])])
            return np.vstack([rot_joint, homogeneous])

        # create result list with root rotation
        result = [rotate_joint(pose[0, :], joints[0, :])]

        # joint rotations
        for i in range(1, len(self.parent_id)):
            joint = (joints[i, :] - joints[self.parent_id[i], :])
            rot_joint = rotate_joint(pose[i, :], joint)
            result.append(result[self.parent_id[i]].dot(rot_joint))

        # Skinning based on final_bone - init_bone
        for i in range(len(result)):
            joint = result[i].dot(np.concatenate([joints[i, :], [0]]))
            joint = np.hstack([zeros, joint.reshape([4, 1])])
            result[i] = result[i] - joint

        return np.dstack(result)
