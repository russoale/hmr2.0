import collections

import numpy as np
import pyglet
from PySide2 import QtCore
from trimesh import PointCloud

from widget.gl_helper import gl_set_culling, gl_set_polygon_mode

pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False

from pyglet import gl

import trimesh.transformations as tf
import trimesh.util as tu

from trimesh.rendering import convert_to_vertexlist
from trimesh.viewer.trackball import Trackball

from widget.qt5_pyglet_widget import Qt5PygletWidget


class SmplWidget(Qt5PygletWidget):
    """ Create a Qt5PygletWidget that will display a trimesh.Scene object
        in an OpenGL context via pyglet.
    """

    def __init__(self, parent):
        super(SmplWidget, self).__init__(parent)

    # noinspection PyAttributeOutsideInit
    def initialize_scene(self, scene, settings_loader, smpl_loader, smooth=True, profile=False):
        lights = {}
        for i, light in enumerate(scene.lights[:7]):
            matrix = scene.graph.get(light.name)[0]
            color = light.color.astype(np.float64) / 255.0
            lights[i] = matrix, color

        w = self.frameGeometry().width()
        h = self.frameGeometry().height()
        scene.camera.resolution = [w, h]

        self.initialize_widget(lights=lights,
                               fov=scene.camera.fov,
                               z_near=scene.camera.z_near,
                               z_far=scene.camera.z_far)

        self._scene = scene
        self._smpl_loader = smpl_loader
        self._settings_loader = settings_loader
        self._smooth = smooth
        self._init_regressors()

        # save initial camera transform
        self._initial_camera_transform = scene.camera_transform.copy()
        self._initial_camera_scale = scene.scale.copy()
        self._initial_camera_centroid = scene.centroid.copy()

        self._line_offset = tf.translation_matrix([0, 0, scene.scale / 1000])
        self._reset_view()

        self._batch = pyglet.graphics.Batch()

        self._vertex_list = {}  # store scene geometry as vertex lists
        self._vertex_list_hash = {}  # store geometry hashes
        self._vertex_list_mode = {}  # store geometry rendering mode

        self._profile = bool(profile)
        if self._profile:
            from pyinstrument import Profiler
            self.Profiler = Profiler

    def _reset_view(self):
        self.view = {
            'cull': True,
            'wireframe': False,
            'merged_vertices': True,
            'show_joints': False,
            'rays': False,
            'ball': QtTrackball(pose=self._initial_camera_transform,
                                size=self._scene.camera.resolution,
                                scale=self._initial_camera_scale,
                                target=self._initial_camera_centroid)}

        self._scene.camera_transform = self.view['ball'].pose

    def _init_regressors(self):
        self._regressors = {
            'smpl_joints': self._smpl_loader.j_regressor,
            'lsp_joints': self._smpl_loader.coco_regressor[:14, :]
        }

        if self._smpl_loader.custom_regressor is not None:
            self._regressors.update({'custom_joints': self._smpl_loader.custom_regressor})

        self._joint_colors = {
            'smpl_joints': [255, 0, 0, 255],
            'lsp_joints': [0, 255, 0, 255],
            'custom_joints': [0, 0, 255, 255]
        }

    def on_init(self):
        self._update_vertex_list()

    def _update_vertex_list(self):
        for name, geom in self._scene.geometry.items():
            if geom.is_empty:
                continue
            if self.geometry_hash(geom) == self._vertex_list_hash.get(name):
                continue

            args = convert_to_vertexlist(geom, smooth=bool(self._smooth))
            self._vertex_list[name] = self._batch.add_indexed(*args)
            self._vertex_list_hash[name] = self.geometry_hash(geom)
            self._vertex_list_mode[name] = args[1]

    def geometry_hash(self, geometry):
        if hasattr(geometry, 'md5'):
            md5 = geometry.md5()
        elif hasattr(geometry, 'tostring'):
            md5 = str(hash(geometry.tostring()))

        if hasattr(geometry, 'visual'):
            md5 += str(geometry.visual.crc())

        return md5

    def on_resize(self, w, h):
        res = [w, h]
        self.view['ball'].resize(res)
        self._scene.camera.resolution = res
        self._scene.camera_transform = self.view['ball'].pose

    def on_draw(self):
        if self._profile:
            profiler = self.Profiler()
            profiler.start()

        self._update_vertex_list()

        transform_camera = np.linalg.inv(self._scene.camera_transform)
        transform_camera = np.asanyarray(transform_camera, dtype=np.float32)
        gl.glMultMatrixf((gl.GLfloat * 16)(*transform_camera.T.ravel()))

        node_names = collections.deque(self._scene.graph.nodes_geometry)
        while len(node_names) > 0:
            current_node = node_names.popleft()
            transform, geometry_name = self._scene.graph.get(current_node)

            if geometry_name is None:
                continue

            mesh = self._scene.geometry[geometry_name]
            if mesh.is_empty:
                continue

            mode = self._vertex_list_mode[geometry_name]
            if mode == gl.GL_LINES:
                # apply the offset in camera space
                arrays = [transform, np.linalg.inv(transform_camera), self._line_offset, transform_camera]
                transform = np.linalg.multi_dot(arrays)

            gl.glPushMatrix()
            transform = np.asanyarray(transform, dtype=np.float32)
            transform = (gl.GLfloat * 16)(*transform.T.ravel())
            gl.glMultMatrixf(transform)

            # draw the mesh with its transform applied
            self._vertex_list[geometry_name].draw(mode=mode)
            gl.glPopMatrix()

        if self._profile:
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))

    def on_mouse_scrolling(self, dy):
        self.view['ball'].scroll(dy)
        self._scene.camera_transform = self.view['ball'].pose

    def on_mouse_drag(self, x, y):
        self.view['ball'].drag([x, y])
        self._scene.camera_transform = self.view['ball'].pose

    def on_mouse_double_click(self, x, y):
        res = self._scene.camera.resolution
        fov_y = np.radians(self._scene.camera.fov[1] / 2.0)
        fov_x = fov_y * (res[0] / float(res[1]))
        half_fov = np.stack([fov_x, fov_y])

        right_top = np.tan(half_fov)
        right_top *= 1 - (1.0 / res)
        left_bottom = -right_top

        right, top = right_top
        left, bottom = left_bottom

        xy_vec = tu.grid_linspace(bounds=[[left, top], [right, bottom]], count=res).astype(np.float64)
        pixels = tu.grid_linspace(bounds=[[0, 0], [res[0] - 1, res[1] - 1]], count=res).astype(np.int64)
        assert xy_vec.shape == pixels.shape

        transform = self._scene.camera_transform
        vectors = tu.unitize(np.column_stack((xy_vec, -np.ones_like(xy_vec[:, :1]))))
        vectors = tf.transform_points(vectors, transform, translate=False)
        origins = (np.ones_like(vectors) * tf.translation_from_matrix(transform))

        indices = np.where(np.all(pixels == np.array([x, y]), axis=1))
        if len(indices) > 0 and len(indices[0]) > 0:
            pixel_id = indices[0][0]
            ray_origin = np.expand_dims(origins[pixel_id], 0)
            ray_direction = np.expand_dims(vectors[pixel_id], 0)
            # print(x, y, pixel_id, ray_origin, ray_direction)

            mesh = self._scene.geometry['geometry_0']

            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=ray_origin,
                ray_directions=ray_direction)

            if locations.size == 0:
                return

            ray_origins = np.tile(ray_origin, [locations.shape[0], 1])
            distances = np.linalg.norm(locations - ray_origins, axis=1)
            idx = np.argsort(distances)  # sort by disctances

            # color closest hit
            tri_color = mesh.visual.face_colors[index_tri[idx[0]]]
            if not np.alltrue(tri_color == [255, 0, 0, 255]):
                tri_color = [255, 0, 0, 255]
            else:
                # unselect triangle
                tri_color = [200, 200, 200, 255]

            mesh.visual.face_colors[index_tri[idx[0]]] = tri_color

            # collect clicked triangle ids
            tri_ids = np.where(np.all(mesh.visual.face_colors == [255, 0, 0, 255], axis=-1))[0]

            if len(tri_ids) >= self._settings_loader.min_triangles:
                # get center of triangles
                barycentric = mesh.triangles_center[tri_ids]
                joint_x = np.mean(barycentric[:, 0])
                joint_y = np.mean(barycentric[:, 1])
                joint_z = np.mean(barycentric[:, 2])
                joint = np.stack([joint_x, joint_y, joint_z])

                if 'joint_0' in self._scene.geometry:
                    self._scene.delete_geometry('joint_0')

                joint = np.expand_dims(joint, 0)
                joint = PointCloud(joint, process=False)
                self._scene.add_geometry(joint, geom_name='joint_0')

            if self.view['rays']:
                from trimesh import load_path
                ray_visualize = load_path(np.hstack((ray_origin, ray_origin + ray_direction)).reshape(-1, 2, 3))
                self._scene.add_geometry(ray_visualize, geom_name='cam_rays')

                # draw path where camera ray hits with mesh (only take 2 closest hits)
                path = np.hstack(locations[:2]).reshape(-1, 2, 3)
                ray_visualize = load_path(path)
                self._scene.add_geometry(ray_visualize, geom_name='cam_rays_hits')

    def on_mouse_press(self, x, y, button, modifiers):
        state = Trackball.STATE_ROTATE

        ctrl = (modifiers & QtCore.Qt.ControlModifier)
        shift = (modifiers & QtCore.Qt.ShiftModifier)

        if button == QtCore.Qt.RightButton:
            state = Trackball.STATE_ZOOM
        elif button == QtCore.Qt.MiddleButton:
            state = Trackball.STATE_PAN

        if ctrl and shift:
            state = Trackball.STATE_ZOOM
        elif shift:
            state = Trackball.STATE_ROLL
        elif ctrl:
            state = Trackball.STATE_PAN

        self.view['ball'].set_state(state)
        self.view['ball'].down([x, y])
        self._scene.camera_transform = self.view['ball'].pose

    def on_key_pressed(self, event):
        magnitude = 10
        switcher = {
            QtCore.Qt.Key_Z: self._reset_view,
            QtCore.Qt.Key_W: self._toggle_wireframe,
            QtCore.Qt.Key_C: self._toggle_culling,
            # currently smoothing can cause change of mesh topology
            # if generating the regressor does not work don't use smoothing
            QtCore.Qt.Key_V: self._unmerge_vertices,
            QtCore.Qt.Key_J: self._toggle_joints,
            QtCore.Qt.Key_R: self._toggle_camera_rays,
            QtCore.Qt.Key_Q: self.window().close,
            QtCore.Qt.Key_M: self.window().showMaximized,
            QtCore.Qt.Key_F: self.window().showFullScreen,
            # arrow keys
            QtCore.Qt.Key_Left: (lambda x=-magnitude, y=0: self._arrow_key_pressed(x, y)),
            QtCore.Qt.Key_Right: (lambda x=magnitude, y=0: self._arrow_key_pressed(x, y)),
            QtCore.Qt.Key_Down: (lambda x=0, y=-magnitude: self._arrow_key_pressed(x, y)),
            QtCore.Qt.Key_Up: (lambda x=0, y=magnitude: self._arrow_key_pressed(x, y)),
        }

        for key, func in switcher.items():
            if key == event.key():
                func()

    def _unmerge_vertices(self):
        mesh = self._scene.geometry['geometry_0']

        self.view['merged_vertices'] = not self.view['merged_vertices']
        if self.view['merged_vertices']:
            mesh.merge_vertices()
        else:
            mesh.unmerge_vertices()
        self.updateGL()

    def _toggle_joints(self):
        self.view['show_joints'] = not self.view['show_joints']
        mesh = self._scene.geometry['geometry_0']
        for geom_name, regressor in self._regressors.items():

            if self.view['show_joints']:
                colors = mesh.visual.vertex_colors
                colors = self._add_joints(colors, mesh, regressor, geom_name)
            else:
                colors = np.array([[200, 200, 200, 255]])
                colors = np.tile(colors, regressor.shape[1]).reshape(-1, 4)
                self._scene.delete_geometry(geom_name)

            mesh.visual.vertex_colors = colors
        self.updateGL()

    def _add_joints(self, colors, mesh, regressor, geom_name='reg_joints'):
        joint_color = self._joint_colors.get(geom_name)
        for reg in regressor:
            color_ids = np.where(reg > 0.)
            colors[color_ids] = joint_color

        x = np.matmul(regressor, mesh.vertices[:, 0])
        y = np.matmul(regressor, mesh.vertices[:, 1])
        z = np.matmul(regressor, mesh.vertices[:, 2])
        joints = np.vstack((x, y, z)).T
        joints = PointCloud(joints, colors=joint_color, process=False)
        self._scene.add_geometry(joints, geom_name=geom_name)

        return colors

    def _toggle_camera_rays(self):
        self.view['rays'] = not self.view['rays']
        if not self.view['rays']:
            rays = [g for g in self._scene.geometry if 'cam_rays' in g]
            self._scene.delete_geometry(rays)

    def _toggle_wireframe(self):
        self.view['wireframe'] = not self.view['wireframe']
        gl_set_polygon_mode(self.view['wireframe'])

    def _toggle_culling(self):
        self.view['cull'] = not self.view['cull']
        gl_set_culling(self.view['cull'])

    def _arrow_key_pressed(self, x, y):
        self.view['ball'].down([0, 0])
        self.view['ball'].drag([x, y])
        self._scene.camera_transform = self.view['ball'].pose

    def unmerge_if_necessary(self):
        if not self.view['merged_vertices']:
            self.view['merged_vertices'] = not self.view['merged_vertices']
            mesh = self._scene.geometry['geometry_0']
            mesh.merge_vertices()
            self.updateGL()


class QtTrackball(Trackball):

    def drag(self, point):

        if self._state == Trackball.STATE_PAN:

            # qt's coordinate system works from upper left origin [0, 0]
            # to lower right [width, height], deltas are being com
            point = np.array(point, dtype=np.float32)
            dx = point[0] - self._pdown[0]
            dy = self._pdown[1] - point[1]

            mindim = 0.3 * np.min(self._size)

            x_axis = self._pose[:3, 0].flatten()
            y_axis = self._pose[:3, 1].flatten()

            dx = -dx / (5.0 * mindim) * self._scale
            dy = -dy / (5.0 * mindim) * self._scale

            translation = dx * x_axis + dy * y_axis
            self._n_target = self._target + translation
            t_tf = np.eye(4)
            t_tf[:3, 3] = translation
            self._n_pose = t_tf.dot(self._pose)

        else:
            super(QtTrackball, self).drag(point)
