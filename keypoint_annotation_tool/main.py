import sys
from os.path import abspath, join

import numpy as np
import scipy.optimize as so
import scipy.spatial as sp
import trimesh
from PySide2.QtWidgets import QApplication, QInputDialog
from PySide2.QtWidgets import QMainWindow
from PySide2.QtWidgets import QMessageBox

from loader.pose_loader import PoseLoader
from loader.settings_loader import SettingsLoader
from loader.smpl_loader import SmplLoader
from main_window import Ui_MainWindow

NEUTRAL = 'n'
NEUTRAL_COCO = 'c'
FEMALE = 'f'
MALE = 'm'

models = {
    # !only pick one of NEUTRAL and NEUTRAL_COCO
    # NEUTRAL: 'basic_model_neutral.pkl',
    NEUTRAL_COCO: 'basic_model_neutral_coco.pkl',
    FEMALE: 'basic_model_female.pkl',
    MALE: 'basic_model_male.pkl',
}

""" python snippet to modify original pkl file
    removes chumpy and scipy dependency
    converting csc_matrix to numpy array only increases file size about 2MB

from scipy.sparse import csc_matrix
from chumpy import Ch
new_dict = {}
for key, value in model.items():
    tmp = value
    if isinstance(value, csc_matrix):
        tmp = value.toarray()
    if isinstance(value, Ch):
        tmp = value.r
    new_dict[key] = tmp

with open(file_name, 'wb') as f:
    pickle.dump(new_dict, f)
"""


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.scene_chache = {}
        self.pose_loader = PoseLoader()
        self.smpl_loader = SmplLoader()
        self.settings_loader = SettingsLoader()

        self.neutral_button.toggled.connect(lambda: self._init_widget())
        self.female_button.toggled.connect(lambda: self._init_widget())
        self.male_button.toggled.connect(lambda: self._init_widget())

        self.female_button.setVisible(FEMALE in models)
        self.male_button.setVisible(MALE in models)

        self.poses_box.currentIndexChanged.connect(lambda: self._init_widget())

        self.regressor_name.textChanged.connect(lambda: self.check_convert_button())
        self.convert_button.clicked.connect(lambda: self.convert_scenes_to_regressor())

        self.reset_button.clicked.connect(lambda: self.reset())

        self.action_min_mesh.triggered.connect(lambda: self.open_min_mesh())
        self.action_min_vertices.triggered.connect(lambda: self.open_min_vertices())
        self.action_min_triangles.triggered.connect(lambda: self.open_min_triangles())
        self.menuBar().setNativeMenuBar(False)

        self._init_poses()
        self._init_regressors()
        self._init_widget()

    def _init_poses(self):
        poses_path = abspath(join(__file__, '..', 'smpl', 'poses', 'cmu_smpl_01_01.pkl'))
        self.pose_loader.init_poses(poses_path)
        poses, shapes, transforms = self.pose_loader.sample_poses()

        for gender, file_name in models.items():
            smpl_model_path = abspath(join(__file__, '..', 'smpl', 'models', file_name))
            self.smpl_loader.init_model(smpl_model_path)

            for i, (pose, shape, transform) in enumerate(zip(poses, shapes, transforms)):
                verts = self.smpl_loader.load_vertices(pose, shape, transform)
                # uncomment to load rest pose and comment both transformations
                # verts = self.smpl_loader.load_vertices()

                faces = self.smpl_loader.faces

                mesh = trimesh.Trimesh(vertices=verts,
                                       faces=faces,
                                       vertex_colors=[200, 200, 200, 255],
                                       face_colors=[0, 0, 0, 0],
                                       use_embree=False,
                                       process=False)

                transform = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1, 0, 0], mesh.centroid)
                mesh.apply_transform(transform)

                transform = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0], mesh.centroid)
                mesh.apply_transform(transform)

                key = gender + '_pose_' + str(i)
                self.scene_chache[key] = mesh.scene()

        for i in range(poses.shape[0]):
            self.poses_box.addItem('Pose ' + str(i))

    def _init_regressors(self):
        regressors_path = abspath(join(__file__, '..', 'regressors', '*.npy'))
        self.smpl_loader.init_custom_regressors(regressors_path)

    def _init_widget(self):

        if hasattr(self.openGLWidget, 'view'):
            # reset current widget to merge vertices to keep topology
            self.openGLWidget.unmerge_if_necessary()

        gender = self.get_checked_gender()
        index = self.poses_box.currentIndex()
        index = index if index >= 0 else 0
        key = gender + '_pose_' + str(index)
        scene = self.scene_chache[key]
        self.openGLWidget.initialize_scene(scene, self.settings_loader, self.smpl_loader)
        self.openGLWidget.updateGL()

    def get_checked_gender(self):
        if self.neutral_button.isChecked():
            gender = NEUTRAL
            if NEUTRAL_COCO in models:
                gender = NEUTRAL_COCO
        elif self.female_button.isChecked():
            gender = FEMALE
        elif self.male_button.isChecked():
            gender = MALE
        else:
            raise Exception('no button checked')

        return gender

    def check_convert_button(self):
        annotated_meshes = 0
        for scene in self.scene_chache.values():
            if 'joint_0' in scene.geometry:
                annotated_meshes += 1

        enough_meshes = annotated_meshes >= self.settings_loader.min_mesh_annotations
        enable_button = self.regressor_name.text() != ''
        self.convert_button.setEnabled(enable_button and enough_meshes)

    def convert_scenes_to_regressor(self):
        self.brute_force_closest_vertex_to_joints()

    def brute_force_closest_vertex_to_joints(self):
        scenes = self.get_valid_scenes()

        # calculate 50 nearest vertex ids for all meshes and joints
        closest_k = 50
        candidates = {}
        keep_searching = True
        while keep_searching:
            for key, scene in scenes.items():
                joint = np.squeeze(scene['joint'].vertices)
                distances, vertex_ids = scene['mesh'].kdtree.query(joint, closest_k)
                candidates[key] = {vertex_id: dist for vertex_id, dist in zip(vertex_ids, distances)}

            # only keep common ids
            from functools import reduce
            common_ids = reduce(np.intersect1d, [list(c.keys()) for c in candidates.values()])

            if len(common_ids) <= self.settings_loader.min_vertices:
                closest_k += 10
                continue
            else:
                keep_searching = False

            # calculate average distance per mesh/joint for valid ids
            mean_dist = [np.mean([c[common_id] for c in candidates.values()]) for common_id in common_ids]
            mean_dist = {common_id: dist for common_id, dist in zip(common_ids, mean_dist)}
            mean_dist = {k: v for k, v in sorted(mean_dist.items(), key=lambda item: item[1])}

        # pick closest vertex with min average distance to all joints per mesh
        closest_id = list(mean_dist)[0]
        final_vertices = [closest_id]
        mean_dist.pop(closest_id)

        while len(final_vertices) < self.settings_loader.min_vertices:
            # calculate all distance combinations between valid vertices
            vertex_ids = list(mean_dist)
            id_dist = [sp.distance.cdist(s['mesh'].vertices[final_vertices], s['mesh'].vertices[vertex_ids]) for s in
                       scenes.values()]
            id_dist = np.mean(id_dist, axis=0)

            # min the ratio between distances to joint and distance to all other vertices
            best_dist = list(mean_dist.values()) / id_dist
            best_id = np.argmin(best_dist)

            # max the difference between distance to all other vertices and distances to joint
            # best_dist = id_dist - list(mean_dist.values())
            # best_id = np.argmax(best_dist)

            n, m = np.unravel_index(best_id, best_dist.shape)
            best_id = vertex_ids[m]

            final_vertices.append(best_id)
            mean_dist.pop(best_id)

        vertices, joints = [], []
        for scene in scenes.values():
            verts = np.asarray(scene['mesh'].vertices).reshape([-1, 3])
            verts = verts[final_vertices]
            vertices.append(verts)
            joint = np.asarray(scene['joint'].vertices).reshape([-1, 3])
            joints.append(joint)

        vertex_weight = np.zeros([6890, ])

        vertices = np.stack(vertices).transpose([0, 2, 1]).reshape([-1, len(final_vertices)])
        joints = np.stack(joints).transpose([0, 2, 1]).reshape([-1])
        # constraint weights to sum up to 1
        vertices = np.concatenate([vertices, np.ones([1, vertices.shape[1]])], 0)
        joints = np.concatenate([joints, np.ones(1)])
        # solve with non negative least squares
        weights = so.nnls(vertices, joints)[0]
        vertex_weight[final_vertices] = weights

        file = join('regressors', 'regressor_{}.npy'.format(self.regressor_name.text()))
        with open(file, 'wb') as f:
            vertex_weight = vertex_weight.astype(np.float32)
            vertex_weight = np.expand_dims(vertex_weight, -1)
            np.save(f, vertex_weight)

            widget = QMessageBox(
                icon=QMessageBox.Information,
                text='Regressor file successfully saved to: {}\n\nClick Reset to start again'.format(file),
                buttons=[QMessageBox.Ok]
            )
            widget.exec_()

        vertex_weight = np.squeeze(vertex_weight)
        self.convert_button.setEnabled(False)
        self.regressor_name.setEnabled(False)

        for scene in self.scene_chache.values():
            mesh = scene.geometry['geometry_0']
            mesh.visual.vertex_colors = [200, 200, 200, 255]
            mesh.visual.vertex_colors[final_vertices] = [0, 255, 0, 255]

            x = np.matmul(vertex_weight, mesh.vertices[:, 0])
            y = np.matmul(vertex_weight, mesh.vertices[:, 1])
            z = np.matmul(vertex_weight, mesh.vertices[:, 2])
            joints = np.vstack((x, y, z)).T
            joints = trimesh.PointCloud(joints, colors=[0, 255, 0, 255])
            scene.add_geometry(joints, geom_name='new_joints')

    def get_valid_scenes(self):
        valid_scenes = {}

        scenes, scales = [], []
        for scene in self.scene_chache.values():
            if 'joint_0' not in scene.geometry:
                continue

            scenes.append(scene)
            scales.append(scene.scale)

        for i, scene in enumerate(scenes):
            mesh = scene.geometry['geometry_0']
            if mesh.vertices.shape[0] != 6890:
                mesh.merge_vertices()
                assert mesh.vertices.shape[0] == 6890, \
                    'something went wrong, mesh should contain 6890 vertices but has {}'.format(mesh.vertices.shape[0])

            valid_scenes[i] = {'mesh': mesh, 'joint': scene.geometry['joint_0']}

        return valid_scenes

    def reset(self):
        self.regressor_name.setText('')
        self.regressor_name.setEnabled(True)
        self.poses_box.clear()

        self._init_poses()
        self._init_regressors()
        self._init_widget()

    def open_min_mesh(self):
        num, ok = QInputDialog.getInt(self, 'Settings', 'Set minimum mesh annotations',
                                      self.settings_loader.min_mesh_annotations, minValue=3, maxValue=30)

        if ok:
            self.settings_loader.min_mesh_annotations = num

    def open_min_vertices(self):
        num, ok = QInputDialog.getInt(self, 'Settings', 'Set minimum vertices',
                                      self.settings_loader.min_vertices, minValue=10, maxValue=30)

        if ok:
            self.settings_loader.min_vertices = num

    def open_min_triangles(self):
        num, ok = QInputDialog.getInt(self, 'Settings', 'Set minimum triangles',
                                      self.settings_loader.min_triangles, minValue=3, maxValue=99)

        if ok:
            self.settings_loader.min_triangles = num


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    try:  # main loop
        sys.exit(app.exec_())
    except SystemExit:
        pass
