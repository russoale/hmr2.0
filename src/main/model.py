import os
import sys
import time

# to make run from console for module import
sys.path.append(os.path.abspath(".."))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

try:
    from IPython import get_ipython

    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except:  # noqa E722
    from tqdm import tqdm

from main.config import Config
from main.dataset import Dataset
from main.discriminator import Discriminator
from main.generator import Generator
from main.model_util import batch_align_by_pelvis, batch_compute_similarity_transform, batch_rodrigues

import tensorflow.compat.v1.losses as v1_loss


class ExceptionHandlingIterator:
    """This class was introduced to avoid tensorflow.python.framework.errors_impl.InvalidArgumentError
        thrown while iterating over the zipped datasets.

        One assumption is that the tf records contain one wrongly generated set due to following error message:
            Expected begin[1] in [0, 462], but got -11 [[{{node Slice}}]] [Op:IteratorGetNextSync]
    """

    def __init__(self, iterable):
        self._iter = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._iter.__next__()
        except StopIteration as e:
            raise e
        except Exception as e:
            print(e)
            return self.__next__()


class Model:

    def __init__(self, display_config=True):
        self.config = Config()
        self.config.save_config()
        if display_config:
            self.config.display()

        self._build_model()
        self._setup_summary()

    def _build_model(self):
        print('building model...\n')

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        gen_input = ((self.config.BATCH_SIZE,) + self.config.ENCODER_INPUT_SHAPE)

        self.generator = Generator()
        self.generator.build(input_shape=gen_input)
        self.generator_opt = tf.optimizers.Adam(learning_rate=self.config.GENERATOR_LEARNING_RATE)

        if not self.config.ENCODER_ONLY:
            disc_input = (self.config.BATCH_SIZE, self.config.NUM_JOINTS * 9 + self.config.NUM_SHAPE_PARAMS)

            self.discriminator = Discriminator()
            self.discriminator.build(input_shape=disc_input)
            self.discriminator_opt = tf.optimizers.Adam(learning_rate=self.config.DISCRIMINATOR_LEARNING_RATE)

        # setup checkpoint
        self.checkpoint_prefix = os.path.join(self.config.LOG_DIR, "ckpt")
        if not self.config.ENCODER_ONLY:
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                             discriminator=self.discriminator,
                                             generator_opt=self.generator_opt,
                                             discriminator_opt=self.discriminator_opt)
        else:
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                             generator_opt=self.generator_opt)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, self.config.LOG_DIR, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        self.restore_check = None
        if self.checkpoint_manager.latest_checkpoint:
            restore_path = self.config.RESTORE_PATH
            if restore_path is None:
                restore_path = self.checkpoint_manager.latest_checkpoint

            self.restore_check = checkpoint.restore(restore_path).expect_partial()
            print('Checkpoint restored from {}'.format(restore_path))

    def _setup_summary(self):
        self.summary_path = os.path.join(self.config.LOG_DIR, 'hmr2.0', '3D_{}'.format(self.config.USE_3D))
        self.summary_writer = tf.summary.create_file_writer(self.summary_path)

        self.generator_loss_log = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        self.kp2d_loss_log = tf.keras.metrics.Mean('kp2d_loss', dtype=tf.float32)
        self.gen_disc_loss_log = tf.keras.metrics.Mean('gen_disc_loss', dtype=tf.float32)

        if self.config.USE_3D:
            self.kp3d_loss_log = tf.keras.metrics.Mean('kp3d_loss', dtype=tf.float32)
            self.pose_shape_loss_log = tf.keras.metrics.Mean('pose_shape_loss', dtype=tf.float32)

        self.discriminator_loss_log = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
        self.disc_real_loss_log = tf.keras.metrics.Mean('disc_real_loss', dtype=tf.float32)
        self.disc_fake_loss_log = tf.keras.metrics.Mean('disc_fake_loss', dtype=tf.float32)

        self.kp2d_mpjpe_log = tf.keras.metrics.Mean('kp2d_mpjpe', dtype=tf.float32)
        self.kp3d_mpjpe_log = tf.keras.metrics.Mean('kp3d_mpjpe', dtype=tf.float32)
        self.kp3d_mpjpe_aligned_log = tf.keras.metrics.Mean('kp3d_mpjpe_aligned', dtype=tf.float32)

    ############################################################
    #  Train/Val
    ############################################################

    def train(self):
        # Place tensors on the CPU
        with tf.device('/CPU:0'):
            dataset = Dataset()
            ds_train = dataset.get_train()
            ds_smpl = dataset.get_smpl()
            ds_val = dataset.get_val()

        start = 1
        if self.config.RESTORE_EPOCH:
            start = self.config.RESTORE_EPOCH

        for epoch in range(start, self.config.EPOCHS + 1):

            start = time.time()
            print('Start of Epoch {}'.format(epoch))

            dataset_train = ExceptionHandlingIterator(tf.data.Dataset.zip((ds_train, ds_smpl)))
            total = int(self.config.NUM_TRAINING_SAMPLES / self.config.BATCH_SIZE)

            for image_data, theta in tqdm(dataset_train, total=total, position=0, desc='training'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._train_step(images, kp2d, kp3d, has3d, theta)

            self._log_train(epoch=epoch)

            total = int(self.config.NUM_VALIDATION_SAMPLES / self.config.BATCH_SIZE)
            for image_data in tqdm(ds_val, total=total, position=0, desc='validate'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._val_step(images, kp2d, kp3d, has3d)

            self._log_val(epoch=epoch)

            print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))

            # saving (checkpoint) the model every 5 epochs
            if epoch % 5 == 0:
                print('saving checkpoint\n')
                self.checkpoint_manager.save(epoch)

        self.summary_writer.flush()
        self.checkpoint_manager.save(self.config.EPOCHS + 1)

    @tf.function
    def _train_step(self, images, kp2d, kp3d, has3d, theta):
        tf.keras.backend.set_learning_phase(1)
        batch_size = images.shape[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_outputs = self.generator(images, training=True)
            # only use last computed theta (from iterative feedback loop)
            _, kp2d_pred, kp3d_pred, pose_pred, shape_pred, _ = generator_outputs[-1]

            vis = tf.expand_dims(kp2d[:, :, 2], -1)
            kp2d_loss = v1_loss.absolute_difference(kp2d[:, :, :2], kp2d_pred, weights=vis)
            kp2d_loss = kp2d_loss * self.config.GENERATOR_2D_LOSS_WEIGHT

            if self.config.USE_3D:
                has3d = tf.expand_dims(has3d, -1)

                kp3d_real = batch_align_by_pelvis(kp3d)
                kp3d_pred = batch_align_by_pelvis(kp3d_pred[:, :self.config.NUM_KP3D, :])

                kp3d_real = tf.reshape(kp3d_real, [batch_size, -1])
                kp3d_pred = tf.reshape(kp3d_pred, [batch_size, -1])

                kp3d_loss = v1_loss.mean_squared_error(kp3d_real, kp3d_pred, weights=has3d) * 0.5
                kp3d_loss = kp3d_loss * self.config.GENERATOR_3D_LOSS_WEIGHT

                """Calculating pose and shape loss basically makes no sense
                    due to missing paired 3d and mosh ground truth data.
                    The original implementation has paired data for Human 3.6 M dataset
                    which was not published due to licence conflict.
                    Nevertheless with SMPLify paired data can be generated
                    (see http://smplify.is.tue.mpg.de/ for more information)
                """
                pose_pred = tf.reshape(pose_pred, [batch_size, -1])
                shape_pred = tf.reshape(shape_pred, [batch_size, -1])
                pose_shape_pred = tf.concat([pose_pred, shape_pred], 1)

                # fake ground truth
                has_smpl = tf.zeros(batch_size, tf.float32)  # do not include loss
                has_smpl = tf.expand_dims(has_smpl, -1)
                pose_shape_real = tf.zeros(pose_shape_pred.shape)

                ps_loss = v1_loss.mean_squared_error(pose_shape_real, pose_shape_pred, weights=has_smpl) * 0.5
                ps_loss = ps_loss * self.config.GENERATOR_3D_LOSS_WEIGHT

            # use all poses and shapes from iterative feedback loop
            fake_disc_input = self.accumulate_fake_disc_input(generator_outputs)
            fake_disc_output = self.discriminator(fake_disc_input, training=True)

            real_disc_input = self.accumulate_real_disc_input(theta)
            real_disc_output = self.discriminator(real_disc_input, training=True)

            gen_disc_loss = tf.reduce_mean(tf.reduce_sum((fake_disc_output - 1) ** 2, axis=1))
            gen_disc_loss = gen_disc_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT

            generator_loss = tf.reduce_sum([kp2d_loss, gen_disc_loss])
            if self.config.USE_3D:
                generator_loss = tf.reduce_sum([generator_loss, kp3d_loss, ps_loss])

            disc_real_loss = tf.reduce_mean(tf.reduce_sum((real_disc_output - 1) ** 2, axis=1))
            disc_fake_loss = tf.reduce_mean(tf.reduce_sum(fake_disc_output ** 2, axis=1))

            discriminator_loss = tf.reduce_sum([disc_real_loss, disc_fake_loss])
            discriminator_loss = discriminator_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT

        generator_grads = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        discriminator_grads = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.generator_opt.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        self.discriminator_opt.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

        self.generator_loss_log.update_state(generator_loss)
        self.kp2d_loss_log.update_state(kp2d_loss)
        self.gen_disc_loss_log.update_state(gen_disc_loss)

        if self.config.USE_3D:
            self.kp3d_loss_log.update_state(kp3d_loss)
            self.pose_shape_loss_log.update_state(ps_loss)

        self.discriminator_loss_log.update_state(discriminator_loss)
        self.disc_real_loss_log.update_state(disc_real_loss)
        self.disc_fake_loss_log.update_state(disc_fake_loss)

    def accumulate_fake_disc_input(self, generator_outputs):
        fake_poses, fake_shapes = [], []
        for output in generator_outputs:
            fake_poses.append(output[3])
            fake_shapes.append(output[4])
        # ignore global rotation
        fake_poses = tf.reshape(tf.convert_to_tensor(fake_poses), [-1, self.config.NUM_JOINTS_GLOBAL, 9])[:, 1:, :]
        fake_poses = tf.reshape(fake_poses, [-1, self.config.NUM_JOINTS * 9])
        fake_shapes = tf.reshape(tf.convert_to_tensor(fake_shapes), [-1, self.config.NUM_SHAPE_PARAMS])

        fake_disc_input = tf.concat([fake_poses, fake_shapes], 1)
        return fake_disc_input

    def accumulate_real_disc_input(self, theta):
        real_poses = theta[:, :self.config.NUM_POSE_PARAMS]
        # compute rotations matrices for [batch x K x 9] - ignore global rotation
        real_poses = batch_rodrigues(real_poses)[:, 1:, :]
        real_poses = tf.reshape(real_poses, [-1, self.config.NUM_JOINTS * 9])
        real_shapes = theta[:, -self.config.NUM_SHAPE_PARAMS:]

        real_disc_input = tf.concat([real_poses, real_shapes], 1)
        return real_disc_input

    def _log_train(self, epoch):
        template = 'Generator Loss: {}, Discriminator Loss: {}'
        print(template.format(self.generator_loss_log.result(), self.discriminator_loss_log.result()))

        with self.summary_writer.as_default():
            tf.summary.scalar('generator_loss', self.generator_loss_log.result(), step=epoch)
            tf.summary.scalar('kp2d_loss', self.kp2d_loss_log.result(), step=epoch)
            tf.summary.scalar('gen_disc_loss', self.gen_disc_loss_log.result(), step=epoch)

            if self.config.USE_3D:
                tf.summary.scalar('kp3d_loss', self.kp3d_loss_log.result(), step=epoch)
                tf.summary.scalar('pose_shape_loss', self.pose_shape_loss_log.result(), step=epoch)

            tf.summary.scalar('discriminator_loss', self.discriminator_loss_log.result(), step=epoch)
            tf.summary.scalar('disc_real_loss', self.disc_real_loss_log.result(), step=epoch)
            tf.summary.scalar('disc_fake_loss', self.disc_fake_loss_log.result(), step=epoch)

        self.generator_loss_log.reset_states()
        self.kp2d_loss_log.reset_states()
        self.gen_disc_loss_log.reset_states()

        if self.config.USE_3D:
            self.kp3d_loss_log.reset_states()
            self.pose_shape_loss_log.reset_states()

        self.discriminator_loss_log.reset_states()
        self.disc_real_loss_log.reset_states()
        self.disc_fake_loss_log.reset_states()

    @tf.function
    def _val_step(self, images, kp2d, kp3d, has3d):
        tf.keras.backend.set_learning_phase(0)

        result = self.generator(images, training=False)
        # only use last computed theta (from accumulated iterative feedback loop)
        _, kp2d_pred, kp3d_pred, _, _, _ = result[-1]

        vis = kp2d[:, :, 2]
        kp2d_norm = tf.norm(kp2d_pred[:, :self.config.NUM_KP2D, :] - kp2d[:, :, :2], axis=2) * vis
        kp2d_mpjpe = tf.reduce_sum(kp2d_norm) / tf.reduce_sum(vis)
        self.kp2d_mpjpe_log(kp2d_mpjpe)

        if self.config.USE_3D:
            # check if at least one 3d sample available
            if tf.reduce_sum(has3d) > 0:
                kp3d_real = tf.boolean_mask(kp3d, has3d)
                kp3d_predict = tf.boolean_mask(kp3d_pred, has3d)
                kp3d_predict = kp3d_predict[:, :self.config.NUM_KP3D, :]

                kp3d_real = batch_align_by_pelvis(kp3d_real)
                kp3d_predict = batch_align_by_pelvis(kp3d_predict)

                kp3d_mpjpe = tf.norm(kp3d_predict - kp3d_real, axis=2)
                kp3d_mpjpe = tf.reduce_mean(kp3d_mpjpe)

                aligned_kp3d = batch_compute_similarity_transform(kp3d_real, kp3d_predict)
                kp3d_mpjpe_aligned = tf.norm(aligned_kp3d - kp3d_real, axis=2)
                kp3d_mpjpe_aligned = tf.reduce_mean(kp3d_mpjpe_aligned)

                self.kp3d_mpjpe_log.update_state(kp3d_mpjpe)
                self.kp3d_mpjpe_aligned_log.update_state(kp3d_mpjpe_aligned)

    def _log_val(self, epoch):
        print('MPJPE kp2d: {}'.format(self.kp2d_mpjpe_log.result()))
        if self.config.USE_3D:
            print('MPJPE kp3d: {}, MPJPE kp3d aligned: {}'.format(self.kp3d_mpjpe_log.result(),
                                                                  self.kp3d_mpjpe_aligned_log.result()))

        with self.summary_writer.as_default():
            tf.summary.scalar('kp2d_mpjpe', self.kp2d_mpjpe_log.result(), step=epoch)
            if self.config.USE_3D:
                tf.summary.scalar('kp3d_mpjpe', self.kp3d_mpjpe_log.result(), step=epoch)
                tf.summary.scalar('kp3d_mpjpe_aligned', self.kp3d_mpjpe_aligned_log.result(), step=epoch)

        self.kp2d_mpjpe_log.reset_states()
        if self.config.USE_3D:
            self.kp3d_mpjpe_log.reset_states()
            self.kp3d_mpjpe_aligned_log.reset_states()

    ############################################################
    #  Test
    ############################################################

    def test(self, return_kps=False):
        """Run evaluation of the model
        Specify LOG_DIR to point to the saved checkpoint directory

        Args:
            return_kps: set to return keypoints - default = False
        """

        if self.restore_check is None:
            raise RuntimeError('restore did not succeed, pleas check if you set config.LOG_DIR correctly')

        if self.config.INITIALIZE_CUSTOM_REGRESSOR:
            self.restore_check.assert_nontrivial_match()
        else:
            self.restore_check.assert_existing_objects_matched().assert_nontrivial_match()

        # Place tensors on the CPU
        with tf.device('/CPU:0'):
            dataset = Dataset()
            ds_test = dataset.get_test()

        start = time.time()
        print('Start of Testing')

        mpjpe, mpjpe_aligned, sequences, kps3d_pred, kps3d_real = [], [], [], [], []

        total = int(self.config.NUM_TEST_SAMPLES / self.config.BATCH_SIZE)
        for image_data in tqdm(ds_test, total=total, position=0, desc='testing'):
            image, kp3d, sequence = image_data[0], image_data[1], image_data[2]
            kp3d_mpjpe, kp3d_mpjpe_aligned, predict_kp3d, real_kp3d = self._test_step(image, kp3d, return_kps=return_kps)

            if return_kps:
                kps3d_pred.append(predict_kp3d)
                kps3d_real.append(real_kp3d)

            mpjpe.append(kp3d_mpjpe)
            mpjpe_aligned.append(kp3d_mpjpe_aligned)
            sequences.append(sequence)

        print('Time taken for testing {} sec\n'.format(time.time() - start))

        def convert(tensor, num=None, is_kp=False):
            if num is None:
                num = self.config.NUM_KP3D
            if is_kp:
                return tf.squeeze(tf.reshape(tf.stack(tensor), [-1, num, 3]))

            return tf.squeeze(tf.reshape(tf.stack(tensor), [-1, num]))

        mpjpe, mpjpe_aligned, sequences = convert(mpjpe), convert(mpjpe_aligned), convert(sequences, 1)
        result_dict = {"kp3d_mpjpe": mpjpe, "kp3d_mpjpe_aligned": mpjpe_aligned, "seq": sequences, }

        if return_kps:
            kps3d_pred, kps3d_real = convert(kps3d_pred, is_kp=True), convert(kps3d_real, is_kp=True)
            result_dict.update({'kps3d_pred': kps3d_pred, 'kps3d_real': kps3d_real})

        return result_dict

    @tf.function
    def _test_step(self, image, kp3d, return_kps=False):
        tf.keras.backend.set_learning_phase(0)

        if len(tf.shape(image)) != 4:
            image = tf.expand_dims(image, 0)
            kp3d = tf.expand_dims(kp3d, 0)

        result = self.generator(image, training=False)
        # only use last computed theta (from accumulated iterative feedback loop)
        _, _, kp3d_pred, _, _, _ = result[-1]

        factor = tf.constant(1000, tf.float32)
        kp3d, kp3d_predict = kp3d * factor, kp3d_pred * factor  # convert back from m -> mm
        kp3d_predict = kp3d_predict[:, :self.config.NUM_KP3D, :]

        real_kp3d = batch_align_by_pelvis(kp3d)
        predict_kp3d = batch_align_by_pelvis(kp3d_predict)

        kp3d_mpjpe = tf.norm(real_kp3d - predict_kp3d, axis=2)

        aligned_kp3d = batch_compute_similarity_transform(real_kp3d, predict_kp3d)
        kp3d_mpjpe_aligned = tf.norm(real_kp3d - aligned_kp3d, axis=2)

        if return_kps:
            return kp3d_mpjpe, kp3d_mpjpe_aligned, predict_kp3d, real_kp3d

        return kp3d_mpjpe, kp3d_mpjpe_aligned, None, None

    ############################################################
    #  Detect/Single Inference
    ############################################################

    def detect(self, image):
        tf.keras.backend.set_learning_phase(0)

        if self.restore_check is None:
            raise RuntimeError('restore did not succeed, pleas check if you set config.LOG_DIR correctly')

        if self.config.INITIALIZE_CUSTOM_REGRESSOR:
            self.restore_check.assert_nontrivial_match()
        else:
            self.restore_check.assert_existing_objects_matched().assert_nontrivial_match()

        if len(tf.shape(image)) != 4:
            image = tf.expand_dims(image, 0)

        result = self.generator(image, training=False)

        vertices_pred, kp2d_pred, kp3d_pred, pose_pred, shape_pred, cam_pred = result[-1]
        result_dict = {
            "vertices": tf.squeeze(vertices_pred),
            "kp2d": tf.squeeze(kp2d_pred),
            "kp3d": tf.squeeze(kp3d_pred),
            "pose": tf.squeeze(pose_pred),
            "shape": tf.squeeze(shape_pred),
            "cam": tf.squeeze(cam_pred)
        }
        return result_dict


if __name__ == '__main__':
    model = Model()
    model.train()
