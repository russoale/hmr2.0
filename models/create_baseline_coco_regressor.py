import pickle
from os.path import abspath, join

import numpy as np


def convert():
    file_name = abspath(join(__file__, '..', 'neutral_smpl_coco_regressor.pkl'))
    with open(file_name, "rb") as f:
        model = pickle.load(f)
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


def main():
    file_name = abspath(join(__file__, '..', 'neutral_smpl_coco_regressor.pkl'))
    with open(file_name, "rb") as f:
        model = pickle.load(f)

    coco_regressor = model['cocoplus_regressor']

    shoulder_right = np.load(abspath(join(__file__, '..', 'regressors', 'regressor_shoulder_right.npy')))
    shoulder_left = np.load(abspath(join(__file__, '..', 'regressors', 'regressor_shoulder_left.npy')))

    coco_regressor[8] = shoulder_right.T
    coco_regressor[9] = shoulder_left.T

    new_dict = {}
    for key, value in model.items():
        tmp = value
        if key == 'cocoplus_regressor':
            tmp = coco_regressor
        new_dict[key] = tmp

    output_file_name = abspath(join(__file__, '..', 'neutral_smpl_coco_regressor_tool_shoulders.pkl'))
    with open(output_file_name, 'wb') as f:
        pickle.dump(new_dict, f)


if __name__ == '__main__':
    main()
