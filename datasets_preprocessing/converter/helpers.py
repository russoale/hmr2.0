import cv2
import numpy as np
import tensorflow as tf


############################################################
#  TFRecordConverter Helper
############################################################

def check_type(name, value, dtype):
    if not isinstance(value, dtype):
        raise ValueError('parameter {} should be of type {} but got {}'.format(name, dtype, value.dtype))
    return value


def check_np_array(name, value, shape, dtype=None):
    is_nd = isinstance(value, np.ndarray)
    is_dtype = value.dtype == dtype if dtype is not None else True
    has_shape = all([value.shape[i] == dim for i, dim in enumerate(shape)])
    if not (is_nd and is_dtype and has_shape):
        raise ValueError('{} should be a ndarray of dtype {} with shape {},'.format(name, dtype, shape) +
                         'but got {} of type: {} with shape: {}'.format(type(value), value.dtype, value.shape))
    return value


# helper from https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample
def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    if isinstance(value, np.ndarray):
        value = np.reshape(value, -1)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    if isinstance(value, np.ndarray):
        value = np.reshape(value, -1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def resize_img(img, scale):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])]
    return new_img, actual_factor
