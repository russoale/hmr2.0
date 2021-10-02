import argparse

import abc
import numpy as np
import tensorflow as tf
from os import path, makedirs, listdir, environ
from tqdm import tqdm

# tf INFO and WARNING messages are not printed
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from converter.helpers import check_type, float_feature


class SmplTFRecordConverter(abc.ABC):

    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.add_arguments()
        self.args, _ = self.parser.parse_known_args()

        self.data_dir = path.join(self.args.data_directory, self.args.dataset_name)
        self.output_dir = path.join(self.args.output_directory, self.args.dataset_name)

        if not path.exists(self.output_dir):
            makedirs(self.output_dir)
        print('Saving results to {}'.format(self.output_dir))

        # create data
        self.__smpl_data_set_splits = []
        self.__examples = []

        self.prepare_data()  # this needs to be implemented by subclass

        print('\n-----SmplTFRecordConverter-----')
        self.convert_data()

    def add_arguments(self):
        self.parser.add_argument('--data_directory', required=False, metavar='/path/to/data',
                                 help='Directory containing the original dataset')
        self.parser.add_argument('--output_directory', required=False, metavar='/path/to/output/data',
                                 help='Directory where to store the generated TFRecord')
        self.parser.add_argument('--dataset_name', required=False, metavar='<dataset name>',
                                 help='Name of the dataset to be converted')
        self.parser.add_argument('--num_shards', required=False, default=10000, metavar='<int>',
                                 help='Number of shards in TFRecord files')

    def convert_data(self):
        print('convert data...')
        for d in self.__smpl_data_set_splits:
            len_poses = d.poses.shape[0]
            save_total = int(np.ceil(len_poses / self.args.num_shards))
            save_points = np.arange(1, save_total) * self.args.num_shards
            save_points = np.append(save_points, len_poses - 1)  # save rest when arrived at last element

            total = tqdm(zip(range(1, len_poses), d.poses, d.shapes), total=len_poses)
            total.set_description_str('processing')
            for i, pose, shape in total:
                self._create_and_add_example(pose, shape)

                if i in save_points:
                    self._save(d.config, lambda message: total.write(message))

    def _create_and_add_example(self, pose, shape=None):
        feat_dict = {'pose': float_feature(pose.astype(np.float32))}
        if shape is not None:
            feat_dict.update({'shape': float_feature(shape.astype(np.float32))})

        self.__examples.append(tf.train.Example(features=tf.train.Features(feature=feat_dict)))

    def _save(self, config, print_saving):
        num_record = 0
        if not (len(listdir(self.output_dir)) == 0):
            import re
            num_record = int(max([re.findall(r"\d+", f)[0] for f in listdir(self.output_dir)])) + 1

        record_name = path.join(self.output_dir, '%03d_{}.tfrecord'.format(config.name))
        tf_record_name = record_name % num_record
        print_saving('saving {}'.format(tf_record_name))
        with tf.io.TFRecordWriter(tf_record_name) as writer:
            for x in self.__examples[:self.args.num_shards]:
                writer.write(x.SerializeToString())
        self.__examples = self.__examples[self.args.num_shards:]

    @abc.abstractmethod
    def prepare_data(self):
        raise NotImplementedError('prepare_data method not yet implemented')

    @property
    def smpl_data_set_splits(self):
        return self.__smpl_data_set_splits

    @smpl_data_set_splits.setter
    def smpl_data_set_splits(self, value):
        if not isinstance(value, list) and value == []:
            raise ValueError('smpl data set splits should be of type List and not empty!')

        if any(not isinstance(x, SmplDataSetSplit) for x in value):
            raise ValueError('smpl data set splits must be a list of type DataSetSplits!')

        self.__smpl_data_set_splits = value


class SmplDataSetConfig:

    def __init__(self, name):
        self.name = check_type('name', name, str)


class SmplDataSetSplit:

    def __init__(self, config: SmplDataSetConfig, poses, shapes=None):
        self.config = config
        self.poses = check_type('poses', poses, np.ndarray)
        self.shapes = check_type('shapes', shapes, np.ndarray) if shapes is not None else None
