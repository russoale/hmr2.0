"""
Modul with visualization commands
"""

import invoke


@invoke.task
def predictions_on_photobridge_data(_context):
    """
    Visualize hmr2 prediction on photobridge data

    Args:
        _context (invoke.Conext): context instance
    """

    import numpy as np
    import os

    import main.config
    import main.model
    import visualise.trimesh_renderer
    import visualise.vis_util

    class DemoConfig(main.config.Config):

        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath('../logs/{}/{}'.format("paired(joints)", "base_model"))
        INITIALIZE_CUSTOM_REGRESSOR = False
        JOINT_TYPE = "cocoplus"

    config = DemoConfig()

    image_path = "../test_images_modified/105508985_original.jpg"

    # initialize model
    model = main.model.Model()

    original_img, input_img, params = visualise.vis_util.preprocess_image(
        image_path, config.ENCODER_INPUT_SHAPE[0])

    result = model.detect(input_img)

    cam = np.squeeze(result['cam'].numpy())[:3]
    vertices = np.squeeze(result['vertices'].numpy())
    joints = np.squeeze(result['kp2d'].numpy())
    joints = ((joints + 1) * 0.5) * params['img_size']

    renderer = visualise.trimesh_renderer.TrimeshRenderer()
    visualise.vis_util.visualize(renderer, original_img, params, vertices, cam, joints)
