import argparse
import os
import sys

import numpy as np

# # to make run from console for module import
# sys.path.append(os.path.abspath('..'))

from main.config import Config
from main.model import Model
import visualise.trimesh_renderer
import visualise.vis_util


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo HMR2.0')

    parser.add_argument('--image', required=False, default='coco1.png')
    parser.add_argument('--model', required=False, default='base_model', help="model from logs folder")
    parser.add_argument('--setting', required=False, default='paired(joints)', help="setting of the model")
    parser.add_argument('--joint_type', required=False, default='cocoplus', help="<cocoplus|custom>")
    parser.add_argument(
        '--init_toes', required=False, default=False, type=str2bool,
        help="only set to True when joint_type=cocoplus")

    print("Muchos menus")

    args = parser.parse_args()
    if args.init_toes:
        assert args.joint_type, "Only init toes when joint type is cocoplus!"

    class DemoConfig(Config):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath('../logs/{}/{}'.format(args.setting, args.model))
        INITIALIZE_CUSTOM_REGRESSOR = args.init_toes
        JOINT_TYPE = args.joint_type

    print(args)

    config = DemoConfig()

    # initialize model
    model = Model()
    original_img, input_img, params = visualise.vis_util.preprocess_image(args.image, config.ENCODER_INPUT_SHAPE[0])

    import icecream
    icecream.ic(original_img.shape)
    icecream.ic(type(original_img))
    icecream.ic(input_img.shape)
    icecream.ic(params)

    result = model.detect(input_img)

    cam = np.squeeze(result['cam'].numpy())[:3]
    vertices = np.squeeze(result['vertices'].numpy())
    joints = np.squeeze(result['kp2d'].numpy())
    joints = ((joints + 1) * 0.5) * params['img_size']

    renderer = visualise.trimesh_renderer.TrimeshRenderer()
    visualise.vis_util.visualize(renderer, original_img, params, vertices, cam, joints)
