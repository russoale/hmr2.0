import tensorflow as tf

from main.util import batch_rodrigues, align_by_pelvis, compute_similarity_transform


############################################################
#  Losses
############################################################

def batch_kp2d_l1_loss(real_kp2d, predict_kp2d):
    mae = tf.losses.MeanAbsoluteError()

    vis = real_kp2d[:, :, 2]
    real_kp2d = tf.boolean_mask(real_kp2d[:, :, :2], vis)
    predict_kp2d = tf.boolean_mask(predict_kp2d[:, :, :2], vis)
    return mae(real_kp2d, predict_kp2d)


def batch_kp3d_l2_loss(real_kp3d, predict_kp3d, has3d):
    mse = tf.losses.MeanSquaredError()

    has3d = tf.expand_dims(has3d, -1)
    real_kp3d = align_by_pelvis(real_kp3d)
    predict_kp3d = align_by_pelvis(predict_kp3d)
    return mse(real_kp3d, predict_kp3d, sample_weight=has3d)


def batch_pose_l2_loss(real_pose, predict_pose, has_smpl):
    mse = tf.losses.MeanSquaredError()
    real_pose = batch_rodrigues(real_pose)[:, 1:, :]
    predict_pose = batch_rodrigues(predict_pose)[:, 1:, :]

    return mse(real_pose, predict_pose, sample_weight=has_smpl)


def batch_shape_l2_loss(real_shape, predict_shape, has_smpl):
    mse = tf.losses.MeanSquaredError()
    return mse(real_shape, predict_shape, sample_weight=has_smpl)


def batch_generator_disc_l2_loss(disc_output_generator):
    return tf.reduce_mean(tf.reduce_sum((disc_output_generator - 1) ** 2, axis=1))


def batch_disc_l2_loss(real_disc_output, fake_disc_output):
    d_loss_real = tf.reduce_mean(tf.reduce_sum((real_disc_output - 1) ** 2, axis=1))
    d_loss_fake = tf.reduce_mean(tf.reduce_sum(fake_disc_output ** 2, axis=1))
    d_loss = d_loss_real + d_loss_fake
    return d_loss_real, d_loss_fake, d_loss


############################################################
#  Validation Metrics
############################################################

def mean_per_joint_position_error_2d(real_kp2d, predict_kp2d):
    vis = real_kp2d[:, :, 2]
    kp2d_norm = tf.norm(predict_kp2d - real_kp2d[:, :, :2], axis=2) * vis
    return tf.reduce_sum(kp2d_norm) / tf.reduce_sum(vis)


def batch_mean_mpjpe_3d(real_kp3d, predict_kp3d):
    return tf.reduce_mean(batch_mpjpe_3d(real_kp3d, predict_kp3d))


def batch_mpjpe_3d(real_kp3d, predict_kp3d):
    real_kp3d = align_by_pelvis(real_kp3d)
    predict_kp3d = align_by_pelvis(predict_kp3d)

    return tf.norm(predict_kp3d - real_kp3d, axis=2)


def batch_mean_mpjpe_3d_aligned(real_kp3d, predict_kp3d):
    return tf.reduce_mean(batch_mpjpe_3d_aligned(real_kp3d, predict_kp3d))


def batch_mpjpe_3d_aligned(real_kp3d, predict_kp3d):
    aligned_kp3d = compute_similarity_transform(real_kp3d, predict_kp3d)
    return tf.norm(aligned_kp3d - real_kp3d, axis=2)
