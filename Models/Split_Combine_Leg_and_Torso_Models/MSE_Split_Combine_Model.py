import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Add, Flatten, Dense, BatchNormalization, ReLU, Input, Concatenate
from keras.models import Model
from scipy.spatial.transform import Rotation as R
from numpy.random import choice
import numpy as np
import keras.metrics as M
from tensorflow.keras.optimizers import Adam
from scipy import linalg
import pickle


# Residual Block from the paper (the paper uses relu and no dropout)
def residual_block(X, d_rate=0.5):
    X_shortcut = X

    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Add()([X, X_shortcut])

    return X


def disc_residual_block(X, d_rate=0.5):
    X_shortcut = X

    X = Dense(1024)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Dense(1024)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Add()([X, X_shortcut])

    return X


def independent_torso_generator(torso_keypoints):
    X = Flatten()(torso_keypoints)
    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = residual_block(X, 0.5)
    X = residual_block(X, 0.5)
    X = Dense(640)(X)
    return X

def independent_leg_generator(torso_keypoints):
    X = Flatten()(torso_keypoints)
    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = residual_block(X, 0.5)
    X = residual_block(X, 0.5)
    X = Dense(384)(X)
    return X


def build_body_generator(input_shape):
    body_input = Input(input_shape)
    torso_output = independent_torso_generator(body_input[:, 6:, :])
    legs_output = independent_leg_generator(body_input[:, :6, :])
    X = Concatenate()([legs_output, torso_output])
    X = residual_block(X, 0.5)
    X = residual_block(X, 0.5)
    X = Dense(16, activation='tanh')(X)
    model = Model(inputs=body_input, outputs=X, name='Generator')
    return model


def build_discriminator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = ReLU()(X)
    X = disc_residual_block(X, 0.5)
    X = disc_residual_block(X, 0.5)
    X = disc_residual_block(X, 0.5)
    X = Dense(1, activation='linear')(X)  # 2 class softmax as output (like the paper)
    model = Model(inputs=X_input, outputs=X, name='Discriminator')
    return model


class TIPSy(Model):
    """Create a Binary GAN
        Args:
            discriminator: the discriminator you want to use in the GAN
            generator: the generator you want to use in the GAN
            reprojector: the reprojection layer used to reproject the predicted 3D pose back into 2D for the discriminator
        Returns:
            generator and discriminator loss
    """

    def __init__(self, discriminator, generator):
        super(TIPSy, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, disc_opt, gen_opt, loss_func, mse):
        super(TIPSy, self).compile()
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.loss_func = loss_func
        self.mse = mse
        self.disc_loss_metric = M.Mean(name='disc_loss')
        self.gen_loss_metric = M.Mean(name='gen_loss')
        self.josh_loss_metric = M.Mean(name='self_consistency_loss')
        self.comparison_loss = M.Mean(name='Two D Comparison Loss')
        self.cross_entropy_metric = M.Mean(name='Cross Entropy Loss')
        self.three_d_loss = M.Mean(name='Three D Comparison Loss')

    @property
    def metrics(self):
        return [self.disc_loss_metric, self.gen_loss_metric]

    def train_step(self, pose_batch):
        # Randomly flip the input pose batch along the y-axis
        if np.random.choice([0, 1]) == 0:
            x_pose = pose_batch[:, :, 0]
            y_pose = pose_batch[:, :, 1]
            x_pose *= -1
            pose_batch = tf.stack([x_pose, y_pose], axis=2)

        # Generate fake 3D pose data
        generated_z = self.generator(pose_batch)
        generated_z = tf.expand_dims(generated_z, -1)  # Reshape purely for concatenation reasons
        generated_3d_pose = tf.concat([pose_batch, generated_z], axis=2)

        """One Random Rotation 3D rotations"""
        azimuth_angle = np.random.uniform(low=-(8 / 9) * np.pi, high=(8 / 9) * np.pi)
        elevation_angle = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)
        azimuth_rotation = R.from_rotvec(azimuth_angle * np.array([0, 1, 0]))
        elevation_rotation = R.from_rotvec(elevation_angle * np.array([1, 0, 0]))
        azimuth_matrix = tf.constant(azimuth_rotation.as_matrix(), dtype=tf.float32)
        elevation_matrix = tf.constant(elevation_rotation.as_matrix(), dtype=tf.float32)
        rotated_pose = tf.matmul(generated_3d_pose, azimuth_matrix)
        rotated_pose = tf.matmul(rotated_pose, elevation_matrix)
        new_poses = rotated_pose
        new_poses = tf.convert_to_tensor(new_poses)
        """End one random 3D rotation"""

        # reproject the random rotation back into 2D
        reprojected_2d_pose = new_poses[:, :, :2]

        # Combine the reprojected_2d_poses with the original batch
        combined_poses = tf.concat([reprojected_2d_pose, pose_batch], axis=0)
        # Generate the labels for 'real' data
        real_labels = tf.ones(int(pose_batch.shape[0]))

        # generate the labels for the 'fake' data
        fake_labels = tf.zeros(int(pose_batch.shape[0]))

        # # Randomly Flip labels
        if np.random.randint(1, 100) <= 10:
            labels = tf.concat([real_labels, fake_labels], axis=0)
        else:
            labels = tf.concat([fake_labels, real_labels], axis=0)

        # Train the discriminator
        with tf.GradientTape(persistent=True) as disc_tape:
            # Get the loss
            predictions = self.discriminator(combined_poses)
            disc_loss = self.mse(labels, predictions)

        # Compute and apply the gradients only if discriminator isn't too strong
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(disc_gradients, self.discriminator.trainable_weights))

        with tf.GradientTape(persistent=True) as gen_tape:
            generated_z = self.generator(pose_batch)
            reshaped_generated_z = tf.reshape(generated_z,
                                              [pose_batch.shape[0], 16, 1])  # Reshape purely for concatenation reasons
            generated_3d_pose = tf.concat([pose_batch, reshaped_generated_z], axis=2)

            """CONSISTENCY LOSS GIVEN BY JOSH"""
            z_y_minus_x = tf.concat([tf.reshape(generated_3d_pose[:, :, 2], [pose_batch.shape[0], 16, 1]),
                                     tf.reshape(generated_3d_pose[:, :, 1], [pose_batch.shape[0], 16, 1])],
                                    axis=2)

            minus_z_y_x = tf.concat([tf.reshape(-generated_3d_pose[:, :, 2], [pose_batch.shape[0], 16, 1]),
                                     tf.reshape(generated_3d_pose[:, :, 1], [pose_batch.shape[0], 16, 1])],
                                    axis=2)

            minus_x_y_minus_z = tf.concat([tf.reshape(-generated_3d_pose[:, :, 0], [pose_batch.shape[0], 16, 1]),
                                           tf.reshape(generated_3d_pose[:, :, 1], [pose_batch.shape[0], 16, 1])],
                                          axis=2)

            consistency_check = self.generator(z_y_minus_x)
            consistency_loss = self.mse(pose_batch[:, :, 0], -consistency_check)

            consistency_check = self.generator(minus_z_y_x)
            consistency_loss += self.mse(pose_batch[:, :, 0], consistency_check)

            consistency_check = self.generator(minus_x_y_minus_z)
            consistency_loss += self.mse(-generated_z, consistency_check)
            """CONSISTENCY LOSS GIVEN BY JOSH"""

            """Reprojection back into the original 2D input dimension loss"""
            azimuth_angle = np.random.uniform(low=-(8 / 9) * np.pi, high=(8 / 9) * np.pi)
            elevation_angle = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)
            azimuth_rotation = R.from_rotvec(azimuth_angle * np.array([0, 1, 0]))
            elevation_rotation = R.from_rotvec(elevation_angle * np.array([1, 0, 0]))
            azimuth_matrix = tf.constant(azimuth_rotation.as_matrix(), dtype=tf.float32)
            elevation_matrix = tf.constant(elevation_rotation.as_matrix(), dtype=tf.float32)
            rotated_pose = tf.matmul(generated_3d_pose, azimuth_matrix)
            rotated_pose = tf.matmul(rotated_pose, elevation_matrix)
            rotated_pose = tf.convert_to_tensor(rotated_pose)
            reprojected_2d_pose = rotated_pose[:, :, :2]
            comparison_z = self.generator(reprojected_2d_pose)
            comparison_z = tf.reshape(comparison_z, [pose_batch.shape[0], 16, 1])
            comparison_3d_pose = tf.concat([reprojected_2d_pose, comparison_z], axis=2)
            rotated_comparison_3d_pose = tf.matmul(comparison_3d_pose, linalg.inv(elevation_matrix))
            rotated_comparison_3d_pose = tf.matmul(rotated_comparison_3d_pose, linalg.inv(azimuth_matrix))
            two_d_comparison = rotated_comparison_3d_pose[:, :, :2]
            comparison_2d_loss = self.mse(pose_batch[:, :, :2], two_d_comparison)
            """End reprojection into the original 2D input dimension loss"""

            """Discriminator Loss"""
            # Labels that say all the poses are real for generator training
            misleading_labels = tf.ones(int(pose_batch.shape[0]))
            predictions = self.discriminator(reprojected_2d_pose)
            cross_entropy_loss = self.mse(misleading_labels, predictions)
            """Discriminator Loss"""

            generator_loss = (cross_entropy_loss) + (comparison_2d_loss * 10) + (consistency_loss * 3)

        gen_gradients = gen_tape.gradient(generator_loss, self.generator.trainable_weights)
        del gen_tape

        self.gen_opt.apply_gradients(zip(gen_gradients, self.generator.trainable_weights))
        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(generator_loss)
        self.josh_loss_metric.update_state(consistency_loss)
        self.cross_entropy_metric.update_state(cross_entropy_loss)
        self.comparison_loss.update_state(comparison_2d_loss)
        return {
            "disc_loss": self.disc_loss_metric.result(),
            "gen_loss": self.gen_loss_metric.result(),
            "gen_cross_entropy_loss": self.cross_entropy_metric.result(),
            "90_consistency_loss": self.josh_loss_metric.result(),
            'two_d_comparison_loss': self.comparison_loss.result(),
        }


class GAN_Callback(keras.callbacks.Callback):
    def __init__(self, gen_filepath=None, disc_filepath=None, GAN_filepath=None):
        self.gen_filepath = gen_filepath
        self.disc_filepath = disc_filepath
        self.GAN_filepath = GAN_filepath

    def on_epoch_end(self, epoch, logs=None):
        # Variable to control how many epochs to save weights
        N = 1
        if epoch >= 0:
            if self.gen_filepath != None:
                self.model.generator.save_weights(self.gen_filepath + '%04d' % (epoch + 1) + '_gen_weights.h5')
            if self.disc_filepath != None:
                self.model.discriminator.save_weights(self.disc_filepath + 'disc_weights%08d.h5' % (epoch + 1))
            if self.GAN_filepath != None:
                self.model.save_weights(self.GAN_filepath + 'GAN_weights%08d.h5' % (epoch + 1))


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    """Begin Actual Model---------------------------------------------------------------------------"""
    # Easy Parameter Tweaking
    batch_size = 8192
    input_size = (16, 2)
    gen_opt = Adam(learning_rate=0.0002)
    disc_opt = Adam(learning_rate=0.0002)
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    mean_squared_error = tf.keras.losses.MeanSquaredError()
    epochs = 800

    with open('h36m_train.pkl', 'rb') as f:
        train = pickle.load(f)

    train_data = []
    # Normalise the data and increase the amount of training data by 2 by flipping along the x axis
    for t in train:
        keypoints = t['joints_3d']
        keypoints = keypoints - keypoints[0]
        two_d_keypoints = keypoints[:, :2]
        pose_max = np.max(abs(two_d_keypoints))
        normalised_two_d_keypoints = two_d_keypoints / pose_max
        train_data.append(normalised_two_d_keypoints[1:, :].astype('float32'))

    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)

    discriminator = build_discriminator(input_size)
    generator = build_body_generator(input_size)

    gen_filepath = 'split_combine_leg_torso_weights/MSE_1/'

    gan = TIPSy(discriminator=discriminator, generator=generator)
    gan.compile(disc_opt=disc_opt, gen_opt=gen_opt, loss_func=cross_entropy, mse=mean_squared_error)
    model = gan.fit(train_data, epochs=epochs, verbose=2, callbacks=[GAN_Callback(gen_filepath=gen_filepath)])
