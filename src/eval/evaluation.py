import json
import sys

print(sys.executable, sys.version)

import numpy as np
import pandas as pd

from os.path import join, abspath

# for local import
sys.path.append(abspath('..'))

from main.config import Config
from main.model import Model

mapping = {
    'mpii_3d': 'mpi',
    'h36m_filtered': 'h36m',
    'h36m': 'h36m',
    'total_cap': 'TC'
}


def eval_per_sequence(sequences, all_kp3d_mpjpe, all_kp3d_mpjpe_aligned):
    # sort sequence
    eval_dict = {}
    for i, sequence in zip(range(len(sequences)), sequences):
        sequence = sequence.decode('utf-8')
        if ' 2_' in sequence or ' 3_' in sequence:  # ignore camera 2 and 3 for mpii 3d
            continue
        mpjpe_tuple = tuple([all_kp3d_mpjpe_aligned[i], all_kp3d_mpjpe[i]])
        if sequence not in eval_dict:
            eval_dict[sequence] = [mpjpe_tuple]
        else:
            eval_dict[sequence].append(mpjpe_tuple)

    seq_names, data = eval_dict.keys(), np.ndarray(shape=(len(eval_dict), 9), dtype=np.float32)
    for i, value in zip(range(len(seq_names)), eval_dict.values()):
        values_per_seq = np.asarray(value, dtype=np.float32)
        mpjpe_aligned = values_per_seq[:, 0, :]
        mpjpe = values_per_seq[:, 1, :]

        mean_mpjpe_aligned = np.mean(mpjpe_aligned)
        mean_mpjpe = np.mean(mpjpe)

        data[i, 0] = mean_mpjpe_aligned  # mean_error_aligned
        data[i, 1] = mean_mpjpe  # mean_error
        data[i, 2] = np.median(mpjpe_aligned)  # median_error
        data[i, 3] = np.std(mpjpe_aligned)  # standard deviation
        data[i, 4] = mpjpe_aligned.min()  # min
        data[i, 5] = mpjpe_aligned.max()  # max
        data[i, 6] = np.percentile(mpjpe_aligned, 25)  # 25 percentile
        data[i, 7] = np.percentile(mpjpe_aligned, 50)  # 50 percentile
        data[i, 8] = np.percentile(mpjpe_aligned, 75)  # 75 percentile

    columns = ['Mean Aligned', 'Mean', 'Median', 'Standard Deviation', 'Min', 'Max', '25%', '50%', '75%']
    df_seq = pd.DataFrame(data, index=seq_names, columns=columns)
    df_seq = df_seq.sort_values(df_seq.columns[0])  # sort by ascending mean aligned
    return df_seq


def eval_per_joint(config, sequences, all_kp3d_mpjpe, all_kp3d_mpjpe_aligned):
    eval_joint_dict = {}
    for ds in config.DATASETS:
        eval_joint_dict[mapping[ds]] = []

    indices = range(len(sequences))
    for i, sequence in zip(indices, sequences):
        sequence = sequence.decode('utf-8')
        if ' 2_' in sequence or ' 3_' in sequence:
            continue
        mpjpe_tuple = tuple([all_kp3d_mpjpe_aligned[i], all_kp3d_mpjpe[i]])
        key = [k for k in eval_joint_dict.keys() if k in sequence][0]
        eval_joint_dict[key].append(mpjpe_tuple)

    columns = ['Mean Aligned', 'Mean', 'Median', 'Standard Deviation', 'Min', 'Max', '25%', '50%', '75%']
    lsp_joints = ['ankle_r', 'knee_r', 'hip_r', 'hip_l', 'knee_l',
                  'ankle_l', 'wrist_r', 'elbow_r', 'shoulder_r',
                  'shoulder_l', 'elbow_l', 'wrist_l', 'neck', 'brain']
    custom_joints = ['toes_r', 'ankle_r', 'knee_r', 'hip_r', 'hip_l', 'knee_l',
                     'ankle_l', 'toes_l', 'wrist_r', 'elbow_r', 'shoulder_r',
                     'shoulder_l', 'elbow_l', 'wrist_l', 'neck', 'brain']

    joints = lsp_joints if config.JOINT_TYPE == 'cocoplus' and not config.INITIALIZE_CUSTOM_REGRESSOR else custom_joints
    num_joints = config.NUM_KP3D
    df_frames = []
    for key, value in eval_joint_dict.items():
        ds_mpjpe_aligned = np.asarray(value)[:, 0, :]
        ds_mpjpe = np.asarray(value)[:, 1, :]

        data = np.ndarray(shape=(num_joints, len(columns)), dtype=float)
        for i, _ in enumerate(joints):
            data[i, 0] = np.mean(ds_mpjpe_aligned[:, i])  # mean_error_aligned
            data[i, 1] = np.mean(ds_mpjpe[:, i])  # mean_error
            data[i, 2] = np.median(ds_mpjpe_aligned[:, i])  # median_error
            data[i, 3] = np.std(ds_mpjpe_aligned[:, i])  # standard deviation
            data[i, 4] = ds_mpjpe_aligned[:, i].min()  # min
            data[i, 5] = ds_mpjpe_aligned[:, i].max()  # max
            data[i, 6] = np.percentile(ds_mpjpe_aligned[:, i], 25)  # 25 percentile
            data[i, 7] = np.percentile(ds_mpjpe_aligned[:, i], 50)  # 50 percentile
            data[i, 8] = np.percentile(ds_mpjpe_aligned[:, i], 75)  # 75 percentile

        df = pd.DataFrame(data, index=joints, columns=columns)
        df = df.sort_values(df.columns[0])
        df_frames.append(df)

    df_joint = pd.concat(df_frames, axis=1, sort=False, keys=list(eval_joint_dict.keys()))
    return df_joint


def eval_per_dataset(datasets, sequences, all_kp3d_mpjpe_aligned, all_kp3d_mpjpe):
    eval_ds_dict = {}
    for ds in datasets:
        eval_ds_dict[mapping[ds]] = []

    indices = range(len(sequences))
    for i, sequence in zip(indices, sequences):
        sequence = sequence.decode('utf-8')
        if ' 2_' in sequence or ' 3_' in sequence:
            continue
        mpjpe_tuple = tuple([all_kp3d_mpjpe_aligned[i], all_kp3d_mpjpe[i]])
        key = [k for k in eval_ds_dict.keys() if k in sequence][0]
        eval_ds_dict[key].append(mpjpe_tuple)

    frames = []
    for ds, values in eval_ds_dict.items():
        values_per_ds = np.asarray(values, dtype=np.float32)
        mpjpe_aligned = values_per_ds[:, 0, :]
        mpjpe = values_per_ds[:, 1, :]

        mean_mpjpe_aligned = np.mean(mpjpe_aligned)
        mean_mpjpe = np.mean(mpjpe)

        data = np.ndarray(shape=(6, 2), dtype=float)
        data[0, 0] = mean_mpjpe_aligned
        data[0, 1] = mean_mpjpe
        print('{} Mean Aligned: {:.3f} --- Mean: {:.3f}'.format(ds, mean_mpjpe_aligned, mean_mpjpe))

        percentiles = [10, 30, 50, 70, 90]
        for i, percentile in enumerate(percentiles, 1):
            data[i, 0] = np.percentile(mpjpe_aligned, percentile)
            data[i, 1] = np.percentile(mpjpe, percentile)

        columns = ['All', '10%', '30%', '50%', '70%', '90%']
        df_percentiles = pd.DataFrame(data.T, index=['Mean Aligned', 'Mean'], columns=columns)
        frames.append(df_percentiles)

    df_datasets = pd.concat(frames, axis=1, sort=False, keys=list(eval_ds_dict.keys()))
    return df_datasets


def eval_angles(datasets, sequences, kps3d_pred, kps3d_real):
    def calc_angle(a, b, c):
        a = np.squeeze(a)
        b = np.squeeze(b)
        c = np.squeeze(c)
        ba = a - b
        bc = c - b

        dot = np.dot(ba, bc)
        cosine_angle = dot / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        degree = np.degrees(angle)
        return degree

    eval_angle_dict = {}
    for ds in datasets:
        eval_angle_dict[mapping[ds]] = []

    indices = range(len(sequences))
    for i, sequence in zip(indices, sequences):
        sequence = sequence.decode("utf-8")
        if ' 2_' in sequence or ' 3_' in sequence:
            continue

        kps3d = tuple([kps3d_pred[i], kps3d_real[i]])
        key = [k for k in eval_angle_dict.keys() if k in sequence][0]
        eval_angle_dict[key].append(kps3d)

    angles_indices = {
        'angles_r': [0, 1, 2],
        'angles_l': [7, 6, 5]
    }

    frames = []
    index = ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max', '25%', '50%', '75%', 'Norm', 'Small Angles 15%',
             'Big Angles 85%']

    for key, kp3d in eval_angle_dict.items():
        kp3d = np.asarray(kp3d, dtype=np.float32)
        kp3d_pred = kp3d[:, 0, :]
        kp3d_real = kp3d[:, 1, :]

        result = {}
        for angle_name, angles in angles_indices.items():
            data = np.ndarray(shape=(kp3d.shape[0], 3), dtype=float)
            for i, kp_pred, kp_real in zip(range(kp3d.shape[0]), kp3d_pred, kp3d_real):
                pred = calc_angle(kp_pred[angles[0]], kp_pred[angles[1]], kp_pred[angles[2]])
                real = calc_angle(kp_real[angles[0]], kp_real[angles[1]], kp_real[angles[2]])
                data[i, 0] = pred
                data[i, 1] = real
                data[i, 2] = np.linalg.norm(real - pred)

            data = np.nan_to_num(data)
            real_data = np.nan_to_num(data[:, 1])
            pred_data = np.nan_to_num(data[:, 0])

            small_angles = data[np.where(real_data < np.percentile(real_data, 15))[0], 2]
            big_angles = data[np.where(real_data > np.percentile(real_data, 85))[0], 2]
            norm_angles = np.concatenate([small_angles, big_angles], -1)

            result[key + '_pred_' + angle_name] = [np.mean(pred_data), np.median(pred_data), np.std(pred_data),
                                                   pred_data.min(), pred_data.max(), np.percentile(pred_data, 25),
                                                   np.percentile(pred_data, 50), np.percentile(pred_data, 75),
                                                   np.mean(norm_angles), np.mean(small_angles), np.mean(big_angles)]

            result[key + '_real_' + angle_name] = [np.mean(real_data), np.median(real_data), np.std(real_data),
                                                   real_data.min(), real_data.max(), np.percentile(real_data, 25),
                                                   np.percentile(real_data, 50), np.percentile(real_data, 75), 0, 0, 0]

        frames.append(pd.DataFrame(result, index=index, columns=list(result.keys())))

    df_angles = pd.concat(frames, axis=1, sort=False, keys=list(eval_angle_dict.keys()))
    return df_angles


if __name__ == '__main__':

    with open(abspath('eval_config.json')) as f:
        eval_config = json.load(f)

    for setting, models in eval_config.items():
        for model in models:
            file_name = '{} - {}'.format(model['file_name'], setting)
            description = model['description']
            evaluate_angles = model['evaluate_angles']
            config = model['config']


            class EvalConfig(Config):
                ENCODER_ONLY = True
                LOG_DIR = join('/', 'data', 'ssd1', 'russales', 'logs', setting, config['model'])
                DATA_DIR = join('/', 'data', 'ssd1', 'russales', config['data_dir'])
                DATASETS = config['datasets']
                JOINT_TYPE = config['joint_type']
                INITIALIZE_CUSTOM_REGRESSOR = config['init_custom_regressor']


            config = EvalConfig()
            model = Model(display_config=False)
            if config.JOINT_TYPE == 'cocoplus' and config.INITIALIZE_CUSTOM_REGRESSOR:
                config.NUM_KP2D = 21
                config.NUM_KP3D = 16

            result = model.test(return_kps=evaluate_angles)
            all_kp3d_mpjpe = result['kp3d_mpjpe'].numpy()
            all_kp3d_mpjpe_aligned = result['kp3d_mpjpe_aligned'].numpy()
            sequences = result['seq'].numpy()

            df_seq = eval_per_sequence(sequences, all_kp3d_mpjpe, all_kp3d_mpjpe_aligned)
            df_joint = eval_per_joint(config, sequences, all_kp3d_mpjpe, all_kp3d_mpjpe_aligned)
            df_dataset = eval_per_dataset(config.DATASETS, sequences, all_kp3d_mpjpe_aligned, all_kp3d_mpjpe)

            if evaluate_angles:
                kps3d_real = result['kps3d_real'].numpy()
                kps3d_pred = result['kps3d_pred'].numpy()
                df_angles = eval_angles(config.DATASETS, sequences, kps3d_pred, kps3d_real)

            with pd.ExcelWriter('reports/{}/{}.xlsx'.format(setting, file_name), engine="xlsxwriter") as writer:
                print('saving Evaluation Excel to {}.xlsx'.format(file_name))
                df_seq.to_excel(writer, sheet_name='Sequences')
                df_joint.to_excel(writer, sheet_name='Joints')
                df_dataset.to_excel(writer, sheet_name='Datasets Overall')
                if evaluate_angles:
                    df_angles.to_excel(writer, sheet_name='Angles')

                config_dict = config.read_config()
                if config_dict is None:
                    config_dict = {a: getattr(config, a) for a in dir(config)
                                   if not a.startswith("_") and not callable(getattr(config, a))}
                worksheet = writer.book.add_worksheet('Model Description')
                worksheet.write(0, 0, description)

                worksheet.write(2, 0, 'Config')
                for i, (k, v) in enumerate(config_dict.items(), 3):
                    worksheet.write(i, 0, k)
                    worksheet.write(i, 1, str(v))

            config.reset()
            print('done\n')
