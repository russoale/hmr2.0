import sys
import time

import os

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
except:
    from tqdm import tqdm

from main.config import Config
from main.dataset import Dataset
from main.discriminator import Discriminator
from main.generator import Generator
from main.losses import batch_kp3d_l2_loss, batch_kp2d_l1_loss, batch_pose_l2_loss, batch_shape_l2_loss, \
    batch_generator_disc_l2_loss, batch_disc_l2_loss, mean_per_joint_position_error_2d, \
    batch_mpjpe_3d, batch_mpjpe_3d_aligned, batch_mean_mpjpe_3d, batch_mean_mpjpe_3d_aligned


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

    def __init__(self):
        self.config = Config()
        self.config.save_config()
        self.config.display()

        self._build_model()
        self._setup_summary()

    def _build_model(self):
        print('building model...\n')

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.generator = Generator()
        self.generator_opt = tf.optimizers.Adam(learning_rate=self.config.ENCODER_LEARNING_RATE)

        if not self.config.ENCODER_ONLY:
            self.discriminator = Discriminator()
            self.discriminator_opt = tf.optimizers.Adam(learning_rate=self.config.DISCRIMINATOR_LEARNING_RATE)

        self.checkpoint_prefix = os.path.join(self.config.LOG_DIR, "ckpt")

        if not self.config.ENCODER_ONLY:
            self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                                  discriminator=self.discriminator,
                                                  generator_opt=self.generator_opt,
                                                  discriminator_opt=self.discriminator_opt)
        else:
            self.checkpoint = tf.train.Checkpoint(generator=self.generator, generator_opt=self.generator_opt)

        checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.config.LOG_DIR, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if checkpoint_manager.latest_checkpoint:
            restore_path = self.config.RESTORE_PATH
            if restore_path is None:
                restore_path = checkpoint_manager.latest_checkpoint

            self.checkpoint.restore(restore_path).expect_partial()
            print('Checkpoint restored from {}'.format(restore_path))

    def _setup_summary(self):
        self.summary_path = os.path.join(self.config.LOG_DIR, "hmr2.0")
        self.summary_writer = tf.summary.create_file_writer(self.summary_path)

        self.generator_loss_log = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        self.kp2d_loss_log = tf.keras.metrics.Mean('kp2d_loss', dtype=tf.float32)
        self.kp3d_loss_log = tf.keras.metrics.Mean('kp3d_loss', dtype=tf.float32)
        self.shape_loss_log = tf.keras.metrics.Mean('shape_loss', dtype=tf.float32)
        self.pose_loss_log = tf.keras.metrics.Mean('pose_loss', dtype=tf.float32)
        self.gen_disc_loss_log = tf.keras.metrics.Mean('gen_disc_loss', dtype=tf.float32)

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
            train_image_data = dataset.get_train()
            val_image_data = dataset.get_val()
            smpl_data = dataset.get_smpl()

        for epoch in range(self.config.EPOCHS):
            start = time.time()
            print('Start of Epoch {}'.format(epoch))

            ds = ExceptionHandlingIterator(tf.data.Dataset.zip((train_image_data, smpl_data)))
            total = int(self.config.NUM_SAMPLES / self.config.BATCH_SIZE)

            for image_data, theta in tqdm(ds, total=total, position=0, desc='training'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                if images.shape[0] is not self.config.BATCH_SIZE or theta.shape[0] is not self.config.BATCH_SIZE:
                    continue
                self._train_step(images, kp2d, kp3d, has3d, theta)

            self._log_train(epoch=epoch)

            total = int(self.config.NUM_VALIDATION_SAMPLES / self.config.BATCH_SIZE)
            for image_data in tqdm(val_image_data, total=total, position=0, desc='validate'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._val_step(images, kp2d, kp3d, has3d)

            self._log_val(epoch=epoch)

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

            # saving (checkpoint) the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                print('\nsaving checkpoint')
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        self.summary_writer.flush()
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    @tf.function
    def _train_step(self, images, kp2d, kp3d, has3d, theta):
        """Open a GradientTape to record the operations run
            during the forward pass, which enables auto differentiation.
            (persistent is set to True because the tape is used more than
            once to calculate the gradients)
        """
        with tf.GradientTape(persistent=True) as tape:
            generator_outputs = self.generator(images, training=True)

            # only use last computed theta (from accumulated iterative feedback loop)
            theta_predict, _, kp2d_predict, kp3d_predict, _ = generator_outputs[-1]

            kp2d_loss = batch_kp2d_l1_loss(kp2d, kp2d_predict[:, :self.config.NUM_KP2D, :])
            kp2d_loss = kp2d_loss * self.config.ENCODER_LOSS_WEIGHT
            kp3d_loss = batch_kp3d_l2_loss(kp3d, kp3d_predict[:, :self.config.NUM_KP3D, :], has3d)
            kp3d_loss = kp3d_loss * self.config.REGRESSOR_LOSS_WEIGHT

            """Calculating pose and shape loss basically makes no sense 
                due to missing paired 3d and mosh ground truth data.
                The original implementation has paired data for Human 3.6 M dataset
                which was not published due to licence conflict.
                Nevertheless with SMPLify paired data can be generated 
                (see http://smplify.is.tue.mpg.de/ for more information)
            """
            has_smpl = tf.constant(0, tf.float32)  # do not include loss

            end_index = self.config.NUM_CAMERA_PARAMS + self.config.NUM_POSE_PARAMS
            pose_predict = theta_predict[:, self.config.NUM_CAMERA_PARAMS:end_index]
            pose_real = tf.zeros(pose_predict.shape)
            pose_loss = batch_pose_l2_loss(pose_real, pose_predict, has_smpl)
            pose_loss = pose_loss * self.config.REGRESSOR_LOSS_WEIGHT

            shape_predict = theta_predict[:, -self.config.NUM_SHAPE_PARAMS:]
            shape_real = tf.zeros(shape_predict.shape)
            shape_loss = batch_shape_l2_loss(shape_real, shape_predict, has_smpl)
            shape_loss = shape_loss * self.config.REGRESSOR_LOSS_WEIGHT

            # use all thetas from iterative feedback loop
            disc_output_generator = self._accumulate_disc_output(generator_outputs)
            gen_disc_loss = batch_generator_disc_l2_loss(disc_output_generator)
            gen_disc_loss = gen_disc_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT

            generator_loss = tf.reduce_sum([kp2d_loss, kp3d_loss, shape_loss, pose_loss, gen_disc_loss])

            fake_disc_output = self._accumulate_disc_output(generator_outputs)
            real_disc_output = self.discriminator(theta, training=True)
            disc_real_loss, disc_fake_loss, disc_loss = batch_disc_l2_loss(real_disc_output, fake_disc_output)

            disc_real_loss = disc_real_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT
            disc_fake_loss = disc_fake_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT
            discriminator_loss = disc_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect
        # to the loss. Calculate the gradients for generator and discriminator.
        generator_grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        discriminator_grads = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        # Apply the gradients to the optimizer
        self.generator_opt.apply_gradients(zip(generator_grads, self.generator.trainable_weights))
        self.discriminator_opt.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_weights))

        self.generator_loss_log(generator_loss)
        self.kp2d_loss_log(kp2d_loss)
        self.kp3d_loss_log(kp3d_loss)
        self.shape_loss_log(shape_loss)
        self.pose_loss_log(pose_loss)
        self.gen_disc_loss_log(gen_disc_loss)

        self.discriminator_loss_log(discriminator_loss)
        self.disc_real_loss_log(disc_real_loss)
        self.disc_fake_loss_log(disc_fake_loss)

    def _accumulate_disc_output(self, generator_outputs):
        discriminator_output = []
        for output in generator_outputs:
            theta = output[0]
            discriminator_output.append(self.discriminator(theta, training=True))

        dim = discriminator_output[0].shape[-1]
        return tf.reshape(tf.convert_to_tensor(discriminator_output), (-1, dim))

    def _log_train(self, epoch):
        template = 'Generator Loss: {}, Discriminator Loss: {}'
        print(template.format(self.generator_loss_log.result(), self.discriminator_loss_log.result()))

        with self.summary_writer.as_default():
            tf.summary.scalar('generator_loss', self.generator_loss_log.result(), step=epoch)
            tf.summary.scalar('kp2d_loss', self.kp2d_loss_log.result(), step=epoch)
            tf.summary.scalar('kp3d_loss', self.kp3d_loss_log.result(), step=epoch)
            tf.summary.scalar('shape_loss', self.shape_loss_log.result(), step=epoch)
            tf.summary.scalar('pose_loss', self.pose_loss_log.result(), step=epoch)
            tf.summary.scalar('gen_disc_loss', self.gen_disc_loss_log.result(), step=epoch)

            tf.summary.scalar('discriminator_loss', self.discriminator_loss_log.result(), step=epoch)
            tf.summary.scalar('disc_real_loss', self.disc_real_loss_log.result(), step=epoch)
            tf.summary.scalar('disc_fake_loss', self.disc_fake_loss_log.result(), step=epoch)

        self.generator_loss_log.reset_states()
        self.kp2d_loss_log.reset_states()
        self.kp3d_loss_log.reset_states()
        self.shape_loss_log.reset_states()
        self.pose_loss_log.reset_states()
        self.gen_disc_loss_log.reset_states()
        self.discriminator_loss_log.reset_states()
        self.disc_real_loss_log.reset_states()
        self.disc_fake_loss_log.reset_states()

    @tf.function
    def _val_step(self, images, kp2d, kp3d, has3d):
        result = self.generator(images, training=False)

        # only use last computed theta (from accumulated iterative feedback loop)
        theta_predict, vertices_predict, kp2d_predict, kp3d_predict, rotation_predict = result[-1]

        kp2d_mpjpe = mean_per_joint_position_error_2d(kp2d, kp2d_predict[:, :self.config.NUM_KP2D, :])
        self.kp2d_mpjpe_log(kp2d_mpjpe)

        # check if at least one 3d sample available
        if tf.reduce_sum(has3d) > 0:
            kp3d_real = tf.boolean_mask(kp3d, has3d)
            kp3d_predict = tf.boolean_mask(kp3d_predict, has3d)
            kp3d_predict = kp3d_predict[:, :self.config.NUM_KP3D, :]

            kp3d_mpjpe = batch_mean_mpjpe_3d(kp3d_real, kp3d_predict)
            kp3d_mpjpe_aligned = batch_mean_mpjpe_3d_aligned(kp3d_real, kp3d_predict)

            self.kp3d_mpjpe_log(kp3d_mpjpe)
            self.kp3d_mpjpe_aligned_log(kp3d_mpjpe_aligned)

    def _log_val(self, epoch):
        template = 'MPJPE kp2d: {}, MPJPE kp3d: {}, MPJPE kp3d aligned: {}'
        print(template.format(self.kp2d_mpjpe_log.result(),
                              self.kp3d_mpjpe_log.result(),
                              self.kp3d_mpjpe_aligned_log.result()))

        with self.summary_writer.as_default():
            tf.summary.scalar('kp2d_mpjpe', self.kp2d_mpjpe_log.result(), step=epoch)
            tf.summary.scalar('kp3d_mpjpe', self.kp3d_mpjpe_log.result(), step=epoch)
            tf.summary.scalar('kp3d_mpjpe_aligned', self.kp3d_mpjpe_aligned_log.result(), step=epoch)

        self.kp2d_mpjpe_log.reset_states()
        self.kp3d_mpjpe_log.reset_states()
        self.kp3d_mpjpe_aligned_log.reset_states()

    ############################################################
    #  Test
    ############################################################

    def test(self, vis=False):
        """Run evaluation of the model
        Specify LOG_DIR to point to the saved checkpoint directory
        Args:
            vis: bool, if True this will return theta, vertices, joint rotations and kp2d as well
        """
        # Place tensors on the CPU
        with tf.device('/CPU:0'):
            dataset = Dataset()
            test_image_data = dataset.get_test()

        start = time.time()
        print('Start of Testing')

        all_kp3d_mpjpe, all_kp3d_mpjpe_aligned, sequences = [], [], []
        all_thetas, all_vertices, all_kp2d, all_rotation = [], [], [], []

        total = int(self.config.NUM_TEST_SAMPLES / self.config.BATCH_SIZE)
        for image_data in tqdm(test_image_data, total=total, position=0, desc='testing'):
            images, kp3d, sequence = image_data[0], image_data[1], image_data[2]
            kp3d_mpjpe, kp3d_mpjpe_aligned, theta, vertices, kp2d, rotation = self._test_step(images, kp3d, vis)
            all_kp3d_mpjpe.append(kp3d_mpjpe)
            all_kp3d_mpjpe_aligned.append(kp3d_mpjpe_aligned)
            sequences.append(sequence)

            if vis:
                all_thetas.append(theta)
                all_vertices.append(vertices)
                all_kp2d.append(kp2d)
                all_rotation.append(rotation)

        print('Time taken for testing {} sec\n'.format(time.time() - start))

        all_kp3d_mpjpe = tf.reshape(tf.stack(all_kp3d_mpjpe), (-1, self.config.NUM_KP3D))
        all_kp3d_mpjpe_aligned = tf.reshape(tf.stack(all_kp3d_mpjpe_aligned), (-1, self.config.NUM_KP3D))
        sequences = tf.reshape(tf.stack(sequences), (-1,))

        result_dict = {
            "kp3d_mpjpe": all_kp3d_mpjpe,
            "kp3d_mpjpe_aligned": all_kp3d_mpjpe_aligned,
            "sequences": sequences,
        }

        if vis:
            all_thetas = tf.reshape(tf.stack(all_thetas), (-1, self.config.NUM_SMPL_PARAMS))
            all_vertices = tf.reshape(tf.stack(all_vertices), (-1, self.config.NUM_VERTICES, 3))
            all_kp2d = tf.reshape(tf.stack(all_kp2d), (-1, self.config.NUM_KP2D, 2))
            all_rotation = tf.reshape(tf.stack(all_rotation), (-1, self.config.NUM_JOINTS_GLOBAL, 3, 3))

            result_dict.update({
                "thetas": all_thetas,
                "vertices": all_vertices,
                "kp2d": all_kp2d,
                "rotations": all_rotation
            })

        return result_dict

    @tf.function
    def _test_step(self, images, kp3d, vis=False):
        result = self.generator(images, training=False)
        # only use last computed theta (from accumulated iterative feedback loop)
        theta_predict, vertices_predict, kp2d_predict, kp3d_predict, rotation_predict = result[-1]

        factor = tf.constant(1000, tf.float32)
        kp3d, kp3d_predict = kp3d * factor, kp3d_predict * factor  # convert back from m -> mm
        kp3d_predict = kp3d_predict[:, :self.config.NUM_KP3D, :]

        kp3d_mpjpe = batch_mpjpe_3d(kp3d, kp3d_predict)
        kp3d_mpjpe_aligned = batch_mpjpe_3d_aligned(kp3d, kp3d_predict)

        if vis:
            return kp3d_mpjpe, kp3d_mpjpe_aligned, theta_predict, vertices_predict, kp2d_predict, rotation_predict
        else:
            return kp3d_mpjpe, kp3d_mpjpe_aligned, None, None, None, None

    ############################################################
    #  Detect/Single Inference
    ############################################################

    def detect(self, image):
        if len(tf.shape(image)) is not 4:
            image = tf.expand_dims(image, 0)

        result = self.generator(image, training=False)

        thetas, vertices, kp2d, kp3d, rotations = result[-1]

        result_dict = {
            "thetas": thetas,
            "vertices": vertices,
            "kp2d": kp2d,
            "kp3d": kp3d,
            "rotations": rotations
        }

        return result_dict


if __name__ == '__main__':
    model = Model()
    model.train()
