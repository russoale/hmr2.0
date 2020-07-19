import pickle
import sys

import cv2
import matplotlib.pyplot as plot
import numpy as np
import os
from matplotlib import gridspec

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from main.config import Config
from main.dataset import Dataset
from main.local import LocalConfig

colors = {
    'pink': [197, 27, 125],
    'light_pink': [233, 163, 201],
    'green': [77, 146, 33],
    'light_green': [161, 215, 106],
    'orange': [200, 90, 39],
    'light_orange': [252, 141, 89],
    'blue': [69, 117, 180],
    'light_blue': [145, 191, 219],
    'red': [215, 48, 39],
    'purple': [118, 42, 131],
    'white': [255, 255, 255],
}

joint_colors = [
    'light_pink', 'light_pink', 'light_pink', 'pink',
    'green', 'light_green', 'light_green', 'light_green',
    'light_blue', 'light_blue', 'light_blue',
    'light_orange', 'light_orange', 'light_orange',
    'purple', 'purple',
    'red',
    'blue', 'blue',
    'orange', 'orange',
]


def _convert_to_int(image, joints):
    # convert image float to int
    if image.min() < 0. and image.max() < 2.:
        image = ((image + 1) / 2 * 255).astype(np.uint8)

    # convert joints float to int
    if np.issubdtype(joints.dtype, np.floating):
        if joints.min() < 0. and joints.max() < 2.:
            joints = _convert_joints(joints, image.shape[:2])
        else:
            joints = np.round(joints).astype(np.int32)

    return image, joints


def _convert_joints(joints, img_shape):
    return ((joints + 1) / 2 * img_shape).astype(np.int32)


def _convert_point(image, joints, parent_id):
    point_pa = joints[parent_id, :]
    point_pa = np.maximum(point_pa, 0)
    point_pa = np.minimum(point_pa, np.array([image.shape[1], image.shape[0]]) - 1)
    return point_pa


def _get_parent_ids(joints, index=1):
    if joints.shape[-1] == 2 or joints.shape[-1] == 3:
        # joints are sorted [K, 2]
        index = 0

    # parent indices -1 means no parents
    if joints.shape[index] == 14:
        parents = np.array([1, 2, 8, 9, 3, 4, 7, 8, -1, -1, 9, 10, 13, -1])
    elif joints.shape[index] == 19:
        parents = np.array([1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16])
    elif joints.shape[index] == 21:
        parents = np.array([1, 2, 3, 10, 11, 4, 5, 6, 9, 10, -1, -1, 11, 12, -1, 14, -1, -1, 19, 20])
    else:
        raise ValueError('Unknown skeleton!!')

    return parents


def load_faces():
    c = Config()
    with open(c.SMPL_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model["f"].astype(np.int32)


def get_original(params, verts, cam, joints):
    # original image params
    img_size = params['img_size']
    img_scale = params['scale']
    img_start = params['start']

    cam_scale = cam[0]
    cam_trans = cam[1:]

    focal = 500.
    trans_z = focal / (0.5 * img_size * cam_scale)
    trans = np.hstack([cam_trans, trans_z])
    vert_shifted = verts + trans

    undo_scale = 1. / np.array(img_scale)

    margin = int(img_size / 2)
    kp_original = (joints + img_start - margin) * undo_scale

    return vert_shifted, kp_original


def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scale = 1.
    if np.max(img.shape[:2]) != img_size:
        print('Resizing image to {}'.format(img_size))
        scale = (float(img_size) / np.max(img.shape[:2]))

    image_scaled, actual_factor = resize_img(img, scale)
    center = np.round(np.array(image_scaled.shape[:2]) / 2).astype(int)
    center = center[::-1]  # image center in (x,y)

    margin = int(img_size / 2)
    image_pad = np.pad(image_scaled, ((margin,), (margin,), (0,)), mode='edge')
    center_pad = center + margin
    start = center_pad - margin
    end = center_pad + margin

    crop = image_pad[start[1]:end[1], start[0]:end[0], :]
    crop = 2 * ((crop / 255.) - 0.5)  # Normalize image to [-1, 1]

    params = {'img_size': img_size, 'scale': scale, 'start': start, 'end': end, }

    return img, crop, params


def resize_img(img, scale):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])]
    return new_img, actual_factor


def visualize(renderer, img, params, verts, cam, joints):
    """Renders the result in original image coordinate frame."""

    vert_shifted, joints_orig = get_original(params, verts, cam, joints)

    # Render results
    img_kp2d = draw_2d_on_image(img, joints_orig)
    img_overlay = renderer(vert_shifted, img=img, bg_color=np.array((255.0, 255.0, 255.0, 1)))
    img_mesh = renderer(vert_shifted, img_size=img.shape[:2])
    img_mesh_rot1 = renderer.rotated(vert_shifted, 60, img_size=img.shape[:2])
    img_mesh_rot2 = renderer.rotated(vert_shifted, -60, img_size=img.shape[:2])

    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.05, hspace=0.05)
    plot.axis('off')
    plot.clf()

    def put_image_on_axis(_img, i, title):
        ax = plot.subplot(gs[i])
        ax.imshow(_img)
        ax.set_title(title)
        ax.axis('off')

    put_image_on_axis(img, 0, 'input')
    put_image_on_axis(img_kp2d, 1, 'joint projection')
    put_image_on_axis(img_overlay, 2, '3D Mesh overlay')
    put_image_on_axis(img_mesh, 3, '3D mesh')
    put_image_on_axis(img_mesh_rot1, 4, 'diff vp')
    put_image_on_axis(img_mesh_rot2, 5, 'diff vp')

    plot.show()


def draw_2d_on_image(input_image, joints, draw_edges=True, vis=None):
    image = input_image.copy()

    image, joints = _convert_to_int(image, joints)
    parents = _get_parent_ids(joints)

    radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))
    for child in range(len(parents)):
        # If kp not visible skip
        if vis is not None and vis[child] == 0:
            continue

        # draw keypoints in image
        point = _convert_point(input_image, joints, child)
        cv2.circle(image, (point[0], point[1]), radius, colors['white'])
        cv2.circle(image, (point[0], point[1]), radius - 1, colors[joint_colors[child]], -1)

        parent_id = parents[child]
        if draw_edges and parent_id >= 0:
            if vis is not None and vis[parent_id] == 0:
                continue

            # draw parent keypoint and connect by edge
            point_pa = _convert_point(input_image, joints, parent_id)
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1, colors[joint_colors[parent_id]], -1)
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]), colors[joint_colors[child]], radius - 2)

    return image


def _get_child_parent_ids(num_joints):
    if num_joints == 14:
        child = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        parent = np.array([1, 2, 8, 9, 3, 4, 7, 8, 8, 9, 9, 10, 13, 13])
        left_right = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], dtype=bool)

    elif num_joints == 16:
        child = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        parent = np.array([1, 2, 3, 10, 11, 4, 5, 6, 9, 10, 10, 11, 11, 12, 15, 15])
        left_right = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], dtype=bool)

    elif num_joints == 19:
        child = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        parent = np.array([1, 2, 8, 9, 3, 4, 7, 8, 8, 9, 9, 10, 13, 13, 14, 15, 16, 17, 18])
        left_right = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1, -1, ], dtype=bool)

    elif num_joints == 21:
        child = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        parent = np.array([1, 2, 3, 10, 11, 4, 5, 6, 9, 10, 10, 11, 11, 12, 15, 15, 16, 17, 18, 19, 20])
        left_right = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1], dtype=bool)

    else:
        raise ValueError('Unknown skeleton!!')

    return child, parent, left_right


def get_color(left_right, i, lcolor='#3498db', rcolor='#e74c3c'):
    if left_right[i] == -1:
        color = '#000000'
    elif left_right[i] == 1:
        color = rcolor
    else:
        color = lcolor
    return color


def show_2d_pose(joints, vis, ax, img_shape=None):
    child, parent, left_right = _get_child_parent_ids(joints.shape[0])

    if img_shape is not None:
        joints = _convert_joints(joints, img_shape)

    # Make connection matrix
    for i in np.arange(len(child)):
        if vis[child[i]] + vis[parent[i]] != 2:
            continue

        x, y = [np.array([joints[parent[i], j], joints[child[i], j]]) for j in range(2)]
        color = get_color(left_right, i)
        ax.plot(x, y, lw=2, c=color)

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    ax.set_aspect('equal')
    ax.invert_yaxis()


def show_3d_pose(joints, ax):
    child, parent, left_right = _get_child_parent_ids(joints.shape[0])

    # Make connection matrix
    for i in np.arange(len(child)):
        x, y, z = [np.array([joints[parent[i], j], joints[child[i], j]]) for j in range(3)]
        color = get_color(left_right, i)
        ax.plot(x, z, y, lw=2, c=color)

    ax.invert_zaxis()

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white) but keep z pane
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    return html


def display_weight_stats(_model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = _model.layers
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for layer in layers:
        weight_values = layer.get_weights()  # list of Numpy arrays

        # if not set to 
        layer.trainable = True
        weight_tensors = layer.weights  # list of TF tensors

        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (layer.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    return display_table(table)


if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


    class DastasetConfig(LocalConfig):
        # DATA_DIR = join('/', 'data', 'ssd1', 'russales', 'new_records')
        # DATASETS = ['coco'] #['lsp', 'lsp_ext', 'mpii', 'coco', 'mpii_3d', 'h36m']
        # SMPL_DATASETS = ['cmu', 'joint_lim']
        TRANS_MAX = 20


    # class Config is implemented as singleton, inizialize subclass first!
    config = DastasetConfig()

    import tensorflow as tf

    # Place tensors on the CPU
    with tf.device('/CPU:0'):
        dataset = Dataset()
        ds_train = dataset.get_train()
        ds_smpl = dataset.get_smpl()
        ds_val = dataset.get_val()

    import matplotlib.pyplot as plt

    for images, kp2d, kp3d, has3d in ds_train.take(1):
        fig = plt.figure(figsize=(9.6, 5.4))
        image_orig = tf.image.decode_jpeg(images[0], channels=3)
        image_orig = image_orig.numpy()
        kp2d = kp2d[0].numpy()
        ax0 = fig.add_subplot(111)
        image_2d = draw_2d_on_image(image_orig, kp2d[:, :2], vis=kp2d[:, 2])
        ax0.imshow(image_2d)

        fig2 = plt.figure(figsize=(9.6, 5.4))
        kp3d = kp3d[0].numpy()
        ax1 = fig2.add_subplot(121, projection='3d')
        show_3d_pose(kp3d, ax1)

        plt.show()
