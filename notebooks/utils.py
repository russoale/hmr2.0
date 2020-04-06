import cv2
import numpy as np

colors = {
    'pink': [197, 27, 125],  # L lower leg
    'light_pink': [233, 163, 201],  # L upper leg
    'light_green': [161, 215, 106],  # L lower arm
    'green': [77, 146, 33],  # L upper arm
    'red': [215, 48, 39],  # head
    'light_red': [252, 146, 114],  # head
    'light_orange': [252, 141, 89],  # chest
    'orange': [200, 90, 39],
    'purple': [118, 42, 131],  # R lower leg
    'light_purple': [175, 141, 195],  # R upper
    'light_blue': [145, 191, 219],  # R lower arm
    'blue': [69, 117, 180],  # R upper arm
    'gray': [130, 130, 130],  #
    'white': [255, 255, 255],  #
}

joint_colors = [
    'light_pink', 'light_pink', 'light_pink',
    'pink', 'pink', 'pink',
    'light_blue', 'light_blue', 'light_blue',
    'blue', 'blue', 'blue',
    'purple', 'purple',
    'red',
    'green', 'green',
    'white', 'white',
    'orange', 'light_orange', 'orange', 'light_orange',
    'pink', 'light_pink'
]


def _convert_to_int(image, joints):
    # convert image float to int
    if image.min() < 0. and image.max() < 2.:
        image = ((image + 1) / 2 * 255).astype(np.uint8)

    # convert joints float to int
    if np.issubdtype(joints.dtype, np.float):
        if joints.min() < 0. and joints.max() < 2.:
            joints = ((joints + 1) / 2 * image.shape[:2]).astype(np.int32)
        else:
            joints = np.round(joints).astype(np.int32)

    return image, joints


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
    elif joints.shape[index] == 25:
        parents = np.array([24, 2, 8, 9, 3, 23, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16, 23, 24, 19, 20, 4, 1])
    else:
        raise ValueError('Unknown skeleton!!')

    return parents


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


def show_2d_pose(joints, ax):
    parents = _get_parent_ids(joints)

    for child in range(len(parents)):
        point = joints[child, :]
        parent_id = parents[child]
        if parent_id >= 0:
            point_parent = joints[parent_id, :]
            x, y = [point[0], point_parent[0]], [point[1], point_parent[1]]
            ax.plot(x, y, lw=2, c='#3498db')

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    ax.set_aspect('equal')
    ax.invert_yaxis()


def show_3d_pose(joints, ax):
    parents = _get_parent_ids(joints)

    for child in range(len(parents)):
        point = joints[child, :]
        parent_id = parents[child]
        if parent_id >= 0:
            point_parent = joints[parent_id, :]
            x, y, z = [point[0], point_parent[0]], [point[1], point_parent[1]], [point[2], point_parent[2]]
            ax.plot(x, y, zs=z, lw=2, c='#3498db')

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
