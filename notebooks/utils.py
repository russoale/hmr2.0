import cv2
import numpy as np


def draw_skeleton(input_image, joints, draw_edges=True, vis=None):
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
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
        'orange', 'light_orange', 'orange', 'light_orange', 'pink', 'light_pink'
    ]

    image = input_image.copy()
    max_val = image.max()
    if np.issubdtype(image.dtype, np.float32) or np.issubdtype(image.dtype, np.float64):
        x = 255 if max_val <= 2. else 1
        image = (image * x).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))
    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16])
        # Left is dark and right is light.
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([1, 2, 8, 9, 3, 4, 7, 8, -1, -1, 9, 10, 13, -1])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    elif joints.shape[1] == 25:
        # parent indices -1 means no parents
        parents = np.array([24, 2, 8, 9, 3, 23, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16, 23, 24, 19, 20, 4, 1])
        # Left is dark and right is light.
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',  # Right shoulder
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple',
            19: 'orange',  # Left Big Toe
            20: 'light_orange',  # Right Big Toe
            21: 'orange',  # Left Small Toe
            22: 'light_orange',  # Right Small Toe
            # Ankles!
            23: 'green',  # Left
            24: 'gray'  # Right
        }
    else:
        print('Unknown skeleton!!')
        assert False

    for child in range(len(parents)):
        if child not in ecolors.keys():
            continue

        point = joints[:, child]

        # Limit to coordinates available in the Image. Image.shape is [heigth, width, c] but point is [x, y].
        # Thus image shape needs to be reversed in the minimum op.
        point = np.maximum(point, 0)
        point = np.minimum(point, np.array([input_image.shape[1], input_image.shape[0]]) - 1)

        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'])
            cv2.circle(image, (point[0], point[1]), radius - 1, colors[joint_colors[child]], -1)
        else:
            cv2.circle(image, (point[0], point[1]), radius - 1, colors[joint_colors[child]], 1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            point_pa = np.maximum(point_pa, 0)
            point_pa = np.minimum(point_pa, np.array([input_image.shape[1], input_image.shape[0]]) - 1)

            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1, colors[joint_colors[pa_id]], -1)
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]), colors[ecolors[child]], radius - 2)

    return image
