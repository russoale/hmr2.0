import cv2
import numpy as np
import os
import trimesh
import trimesh.transformations as trans

from main.local import LocalConfig
from main.model import Model
from notebooks.vis_util import preprocess_image, visualize, load_faces


class TrimeshRenderer(object):

    def __init__(self, img_size=(224, 224), focal_length=500.):
        self.h, self.w = img_size[0], img_size[1]
        self.focal_length = focal_length
        self.faces = load_faces()

    def __call__(self, verts, img=None, img_size=None, bg_color=None):
        """Render smpl mesh
        Args:
            verts: [6890 x 3], smpl vertices
            img: [h, w, channel] (optional)
            img_size: [h, w] specify frame size of rendered mesh (optional)
        """

        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h, w = img_size[0], img_size[1]
        else:
            h, w = self.h, self.w

        mesh = self.mesh(verts)
        scene = mesh.scene()

        if bg_color is not None:
            bg_color = np.zeros(4)

        image_bytes = scene.save_image(resolution=(h, w), background=bg_color, visible=True)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            x_offset = y_offset = 0
            y1, y2 = y_offset, y_offset + image.shape[0]
            x1, x2 = x_offset, x_offset + image.shape[1]

            alpha_mesh = image[:, :, 3] / 255.0
            alpha_image = 1.0 - alpha_mesh

            for c in range(0, 3):
                img[y1:y2, x1:x2, c] = (alpha_mesh * image[:, :, c] + alpha_image * img[y1:y2, x1:x2, c])

            # image = cv2.addWeighted(img, 0., image, 1., 0)
            image = img

        return image

    def mesh(self, verts):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces)
        # this transform is necessary to get correct image
        # because z axis is other way around in trimesh
        transform = trans.rotation_matrix(np.deg2rad(-180), [1, 0, 0], mesh.centroid)
        mesh.apply_transform(transform)
        return mesh

    def rotated(self, verts, deg, axis='y', img=None, img_size=None):
        rad = np.deg2rad(deg)

        if axis == 'x':
            mat = [rad, 0, 0]
        elif axis == 'y':
            mat = [0, rad, 0]
        else:
            mat = [0, 0, rad]

        around = cv2.Rodrigues(np.array(mat))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(new_v, img=img, img_size=img_size)


if __name__ == '__main__':
    class TrimeshConfig(LocalConfig):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath('../../logs/26042020-171733')


    config = TrimeshConfig()

    # initialize model
    model = Model()
    original_img, input_img, params = preprocess_image('images/coco1.png', config.ENCODER_INPUT_SHAPE[0])

    result = model.detect(input_img)

    cam = np.squeeze(result['cam'].numpy())[:3]
    vertices = np.squeeze(result['vertices'].numpy())
    joints = np.squeeze(result['kp2d'].numpy())

    renderer = TrimeshRenderer()
    visualize(renderer, original_img, params, vertices, cam, joints)
