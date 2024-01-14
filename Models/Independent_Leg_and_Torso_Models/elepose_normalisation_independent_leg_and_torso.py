import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import Dropout, Add, Flatten, Dense, BatchNormalization, LeakyReLU, Input, Layer, ReLU
from keras.models import Model
from scipy.spatial.transform import Rotation as R
from numpy.random import choice
import numpy as np
import keras.metrics as M
from tensorflow.keras.optimizers import Adam
from scipy import linalg
import pickle
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

"""Bone Length Prior Taken From elepose!"""
def get_bone_lengths_all(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]]

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = tf.norm(bones, ord=2, axis=1)

    return bone_lengths

"""Normalisation taken from elepose"""
def normalize_head(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [root_joint]]

    scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 10], axis=1, keepdims=True)
    p2ds = poses_2d / scale.mean()

    p2ds = p2ds * (1 / 10)

    return p2ds



"""Perspective Projection taken from elepose"""
def perspective_projection(pose_3d):
    p2d = pose_3d[:, 0:34].reshape(-1, 2, 17)
    p2d = p2d / pose_3d[:, 34:51].reshape(-1, 1, 17)

    return p2d.reshape(-1, 34)



def manual_mse_loss(x, y, weight_matrix):
    z = (x - y) ** 2
    z = np.sum(z, axis=0)
    if len(z.shape) > 1:
        z = np.sum(z, axis=0)
    if np.max(z) != 0:  # in the case that x and y are identical
        z /= np.max(z)
    z *= weight_matrix
    z = np.sum(z) / np.product(x.shape)
    return z


class Mask(Layer):
    def __init__(self):
        super(Mask, self).__init__()
        total_key_point_num = 16
        w_init = (tf.random.uniform([total_key_point_num]) - 0.5) * 0.1
        self.w = tf.Variable(initial_value=w_init, trainable=True)

    def call(self, inputs):
        mask = tf.reshape(tf.sigmoid(self.w), [16, 1])
        mask = tf.concat([mask, mask], axis=1)
        output = inputs * (tf.stop_gradient(tf.cast((mask > 0.5), dtype=tf.float32) - mask) + mask)
        return output


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


def build_legs_generator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = residual_block(X, 0.5)
    X = residual_block(X, 0.5)
    X = Dense(7, activation='linear')(X)
    model = Model(inputs=X_input, outputs=X, name='legs_generator')
    return model


def build_torso_generator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = residual_block(X, 0.5)
    X = residual_block(X, 0.5)
    X = Dense(10, activation='linear')(X)
    model = Model(inputs=X_input, outputs=X, name='torso_generator')
    return model


def build_discriminator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = ReLU()(X)
    X = disc_residual_block(X, 0.5)
    X = Dense(1, activation='linear')(X)
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

    def __init__(self, discriminator, torso_generator, legs_generator, depth_offset):
        super(TIPSy, self).__init__()
        self.discriminator = discriminator
        self.legs_generator = legs_generator
        self.torso_generator = torso_generator
        self.depth_offset = depth_offset
        self.bone_relations_mean = tf.Variable([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
                                                0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652],
                                               trainable=False)

    def compile(self, disc_opt, gen_opt, mse):
        super(TIPSy, self).compile()
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.mse = mse
        self.disc_loss_metric = M.Mean(name='disc_loss')
        self.gen_loss_metric = M.Mean(name='gen_loss')
        self.josh_loss_metric = M.Mean(name='self_consistency_loss')
        self.comparison_loss = M.Mean(name='Two D Comparison Loss')
        self.cross_entropy_metric = M.Mean(name='Cross Entropy Loss')
        self.three_d_loss = M.Mean(name='Three D Comparison Loss')
        self.bone_length_loss = M.Mean(name='bone_length_loss')
        self.pairwise_loss = M.Mean(name='elepose_loss')
    @property
    def metrics(self):
        return [self.disc_loss_metric, self.gen_loss_metric]

    def train_step(self, pose_batch):
        # Randomly flip the input pose batch along the y-axis
        if np.random.choice([0, 1]) == 0:
            x_pose = pose_batch[:, 0, :]
            y_pose = pose_batch[:, 1, :]
            x_pose *= -1
            pose_batch = tf.stack([x_pose, y_pose], axis=1)

        # Figure out euclidean distance matrix for 2D and 3D reprojection loss weights
        average_euclidean_distance_matrix = np.sum(pose_batch, axis=0) / len(pose_batch)
        average_euclidean_distance_matrix = np.hypot(average_euclidean_distance_matrix[0, :],
                                                     average_euclidean_distance_matrix[1, :])
        average_euclidean_distance_matrix /= np.max(average_euclidean_distance_matrix)
        average_euclidean_distance_matrix += np.ones(17)


        # Generate fake 3D pose data
        generated_upper_z = self.torso_generator(pose_batch[:, :, 7:])
        generated_lower_z = self.legs_generator(pose_batch[:, :, :7])
        generated_z = tf.concat([generated_lower_z, generated_upper_z], axis=1)
        root_masks = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        generated_z *= root_masks
        generated_z += self.depth_offset
        generated_z = tf.clip_by_value(generated_z, 1, np.max(generated_z))
        generated_z = generated_z.astype('float64')
        ## lift from depth offset of 10 and normalize
        generated_3d_pose = tf.concat([(pose_batch.reshape(-1, 2, 17) * np.tile(generated_z.reshape(-1, 1, 17), (1,2,1))).reshape(-1, 34), generated_z], axis=1).reshape(-1, 3, 17)
        generated_3d_pose = generated_3d_pose.reshape(-1, 3, 17) - generated_3d_pose.reshape(-1, 3, 17)[:, :, [0]]
        generated_3d_pose = tf.transpose(generated_3d_pose, perm=[0, 2, 1])

        """One Random Rotation 3D rotation and reprojection"""
        azimuth_angle = np.random.uniform(low=-(8 / 9) * np.pi, high=(8 / 9) * np.pi)
        elevation_angle = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)
        azimuth_rotation = R.from_rotvec(azimuth_angle * np.array([0, 1, 0]))
        elevation_rotation = R.from_rotvec(elevation_angle * np.array([1, 0, 0]))
        azimuth_matrix = tf.constant(azimuth_rotation.as_matrix(), dtype=tf.float64)
        elevation_matrix = tf.constant(elevation_rotation.as_matrix(), dtype=tf.float64)
        rotated_pose = tf.matmul(generated_3d_pose, azimuth_matrix)
        rotated_pose = tf.matmul(rotated_pose, elevation_matrix)
        rotated_pose = tf.transpose(rotated_pose, perm=[0, 2, 1]).reshape(-1, 51)
        rotated_pose = tf.concat([rotated_pose[:, 0:34], rotated_pose[:, 34:51] + self.depth_offset], axis=1)
        reprojected_2d_pose = perspective_projection(rotated_pose).reshape(-1, 2, 17)
        """End one random 3D rotation and reprojection"""

        # Combine the reprojected_2d_poses with the original batch
        combined_poses = tf.concat([reprojected_2d_pose, pose_batch], axis=0)
        # Generate the labels for 'real' data
        real_labels = tf.zeros(int(pose_batch.shape[0]))

        # generate the labels for the 'fake' data
        fake_labels = tf.ones(int(pose_batch.shape[0]))

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

        # Train the generator
        # e^l/50

        with tf.GradientTape(persistent=True) as gen_tape:
            generated_upper_z = self.torso_generator(pose_batch[:, :, 7:])
            generated_lower_z = self.legs_generator(pose_batch[:, :, :7])
            generated_z = tf.concat([generated_lower_z, generated_upper_z], axis=1)
            generated_z *= root_masks
            generated_z += self.depth_offset
            generated_z = tf.clip_by_value(generated_z, 1, np.max(generated_z))
            generated_z = generated_z.astype('float64')
            ## lift from depth offset of 10 and normalize
            generated_3d_pose = tf.concat(
                [(pose_batch.reshape(-1, 2, 17) * np.tile(generated_z.reshape(-1, 1, 17), (1, 2, 1))).reshape(-1, 34),
                 generated_z], axis=1).reshape(-1, 3, 17)
            generated_3d_pose = generated_3d_pose.reshape(-1, 3, 17) - generated_3d_pose.reshape(-1, 3, 17)[:, :, [0]]

            """Bone Length Loss"""
            bl = get_bone_lengths_all(tf.reshape(generated_3d_pose, [-1, 51]))
            relative_bl = bl / bl.mean(axis=1, keepdims=True)
            torso_bone_length_loss = np.mean(np.sum((self.bone_relations_mean[6:] - relative_bl[:, 6:]) ** 2, axis=1))
            legs_bone_length_loss = np.mean(np.sum((self.bone_relations_mean[:7] - relative_bl[:, :7]) ** 2, axis=1))
            """End Bone Length Loss"""

            """90 degree CONSISTENCY LOSS"""
            z_y_minus_x = tf.concat([tf.reshape(generated_3d_pose[:, 2, :], [pose_batch.shape[0], 1, 17]),
                                     tf.reshape(generated_3d_pose[:, 1, :], [pose_batch.shape[0], 1, 17])],
                                    axis=1)

            minus_z_y_x = tf.concat([tf.reshape(-generated_3d_pose[:, 2, :], [pose_batch.shape[0], 1, 17]),
                                     tf.reshape(generated_3d_pose[:, 1, :], [pose_batch.shape[0], 1, 17])],
                                    axis=1)

            minus_x_y_minus_z = tf.concat([tf.reshape(-generated_3d_pose[:, 0, :], [pose_batch.shape[0], 1, 17]),
                                           tf.reshape(generated_3d_pose[:, 1, :], [pose_batch.shape[0], 1, 17])],
                                          axis=1)

            upper_consistency_check = self.torso_generator(z_y_minus_x[:, :, 7:])
            lower_consistency_check = self.legs_generator(z_y_minus_x[:, :, :7])
            upper_consistency_loss = manual_mse_loss(pose_batch[:, 0, 7:], -upper_consistency_check,
                                                     weight_matrix=average_euclidean_distance_matrix[7:])
            lower_consistency_loss = manual_mse_loss(pose_batch[:, 0, :7], -lower_consistency_check,
                                                     weight_matrix=average_euclidean_distance_matrix[:7])

            upper_consistency_check = self.torso_generator(minus_z_y_x[:, :, 7:])
            lower_consistency_check = self.legs_generator(minus_z_y_x[:, :, :7])
            upper_consistency_loss += manual_mse_loss(pose_batch[:, 0, 7:], upper_consistency_check,
                                                      weight_matrix=average_euclidean_distance_matrix[7:])
            lower_consistency_loss += manual_mse_loss(pose_batch[:, 0, :7], lower_consistency_check,
                                                      weight_matrix=average_euclidean_distance_matrix[:7])

            upper_consistency_check = self.torso_generator(minus_x_y_minus_z[:, :, 7:])
            lower_consistency_check = self.legs_generator(minus_x_y_minus_z[:, :, :7])
            upper_consistency_loss += manual_mse_loss(-generated_z[:, 7:], upper_consistency_check,
                                                      weight_matrix=average_euclidean_distance_matrix[7:])
            lower_consistency_loss += manual_mse_loss(-generated_z[:, :7], lower_consistency_check,
                                                      weight_matrix=average_euclidean_distance_matrix[:7])
            """90 degree CONSISTENCY LOSS"""

            generated_3d_pose = tf.transpose(generated_3d_pose, perm=[0, 2, 1])

            """Reprojection back into the original 2D input dimension loss"""
            azimuth_angle = np.random.uniform(low=-(8 / 9) * np.pi, high=(8 / 9) * np.pi)
            elevation_angle = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)
            azimuth_rotation = R.from_rotvec(azimuth_angle * np.array([0, 1, 0]))
            elevation_rotation = R.from_rotvec(elevation_angle * np.array([1, 0, 0]))
            azimuth_matrix = tf.constant(azimuth_rotation.as_matrix(), dtype=tf.float64)
            elevation_matrix = tf.constant(elevation_rotation.as_matrix(), dtype=tf.float64)
            rotated_pose = tf.matmul(generated_3d_pose, azimuth_matrix)
            rotated_pose = tf.matmul(rotated_pose, elevation_matrix)
            rotated_pose = tf.transpose(rotated_pose, perm=[0, 2, 1]).reshape(-1, 51)
            rotated_pose = tf.concat([rotated_pose[:, 0:34], rotated_pose[:, 34:51] + self.depth_offset], axis=1)
            reprojected_2d_pose = perspective_projection(rotated_pose).reshape(-1, 2, 17)
            upper_comparison_z = self.torso_generator(reprojected_2d_pose[:, :, 7:])
            lower_comparison_z = self.legs_generator(reprojected_2d_pose[:, :, :7])
            comparison_z = tf.concat([lower_comparison_z, upper_comparison_z], axis=1)
            comparison_z *= root_masks
            comparison_z += self.depth_offset
            comparison_z = tf.clip_by_value(comparison_z, 1, np.max(comparison_z))
            comparison_z = comparison_z.astype('float64')
            ## lift from depth offset of 10 and normalize
            comparison_3d_pose = tf.concat(
                [(pose_batch.reshape(-1, 2, 17) * np.tile(comparison_z.reshape(-1, 1, 17), (1, 2, 1))).reshape(-1, 34),
                 comparison_z], axis=1).reshape(-1, 3, 17)
            comparison_3d_pose = comparison_3d_pose.reshape(-1, 3, 17) - comparison_3d_pose.reshape(-1, 3, 17)[:, :, [0]]

            """3D LOSS"""
            torso_3d_loss = manual_mse_loss(rotated_pose.reshape(-1, 3, 17)[:,:,7:], comparison_3d_pose[:,:,7:], weight_matrix=average_euclidean_distance_matrix[7:])
            legs_3d_loss = manual_mse_loss(rotated_pose.reshape(-1, 3, 17)[:,:,:7], comparison_3d_pose[:,:,:7], weight_matrix=average_euclidean_distance_matrix[:7])
            """3D LOSS"""

            comparison_3d_pose = tf.transpose(comparison_3d_pose, perm=[0, 2, 1])
            rotated_comparison_3d_pose = tf.matmul(comparison_3d_pose, linalg.inv(elevation_matrix))
            rotated_comparison_3d_pose = tf.matmul(rotated_comparison_3d_pose, linalg.inv(azimuth_matrix))
            rotated_comparison_3d_pose = tf.transpose(rotated_comparison_3d_pose, perm=[0, 2, 1]).reshape(-1, 51)
            rotated_comparison_3d_pose = tf.concat([rotated_comparison_3d_pose[:, 0:34], rotated_comparison_3d_pose[:, 34:51] + self.depth_offset], axis=1)
            two_d_comparison = perspective_projection(rotated_comparison_3d_pose).reshape(-1, 2, 17)
            upper_comparison_2d_loss = manual_mse_loss(pose_batch[:, :, 7:], two_d_comparison[:, :, 7:],
                                                       weight_matrix=average_euclidean_distance_matrix[7:])
            lower_comparison_2d_loss = manual_mse_loss(pose_batch[:, :, :7], two_d_comparison[:, :, :7],
                                                       weight_matrix=average_euclidean_distance_matrix[:7])
            """End reprojection into the original 2D input dimension loss"""

            """pairwise deformation loss"""
            generated_3d_pose = tf.transpose(generated_3d_pose, perm=[0,2,1])
            num_pairs = int(np.floor(generated_3d_pose.shape[0] / 2))
            pose_pairs = generated_3d_pose[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 3, 17)
            comparison_3d_pose = tf.transpose(comparison_3d_pose, perm=[0,2,1])
            pose_pairs_re_rot_3d = comparison_3d_pose[0:(2 * num_pairs)].reshape(-1, 2, 3, 17)
            torso_pairwise_deformation = self.mse(pose_pairs[:,:,:,7:], pose_pairs_re_rot_3d[:,:,:,7:])
            legs_pairwise_deformation = self.mse(pose_pairs[:,:,:,:7], pose_pairs_re_rot_3d[:,:,:,:7])
            """pairwise deformation loss"""


            """Discriminator Loss"""
            # Labels that say all the poses are real for generator training
            misleading_labels = tf.zeros(int(pose_batch.shape[0]))
            predictions = self.discriminator(reprojected_2d_pose)
            cross_entropy_loss = self.mse(misleading_labels, predictions)
            """Discriminator Loss"""

            upper_loss = (cross_entropy_loss * 0.625) + (upper_comparison_2d_loss * 10) + (
                    upper_consistency_loss * 3) + torso_bone_length_loss + torso_pairwise_deformation + torso_3d_loss
            lower_loss = (cross_entropy_loss * 0.375) + (lower_comparison_2d_loss * 10) + (
                    lower_consistency_loss * 3) + legs_bone_length_loss + legs_pairwise_deformation + legs_3d_loss

        gen_lower_gradients = gen_tape.gradient(lower_loss, self.legs_generator.trainable_weights)
        gen_upper_gradients = gen_tape.gradient(upper_loss, self.torso_generator.trainable_weights)
        del gen_tape

        self.gen_opt.apply_gradients(zip(gen_lower_gradients, self.legs_generator.trainable_weights))
        self.gen_opt.apply_gradients(zip(gen_upper_gradients, self.torso_generator.trainable_weights))
        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(upper_loss + lower_loss)
        self.pairwise_loss.update_state(torso_pairwise_deformation + legs_pairwise_deformation)
        self.josh_loss_metric.update_state(upper_consistency_loss + lower_consistency_loss)
        self.cross_entropy_metric.update_state(cross_entropy_loss)
        self.three_d_loss.update_state(torso_3d_loss + legs_3d_loss)
        self.comparison_loss.update_state(upper_comparison_2d_loss + lower_comparison_2d_loss)
        self.bone_length_loss.update_state(torso_bone_length_loss + legs_bone_length_loss)
        return {
            "total_disc_loss": self.disc_loss_metric.result(),
            "total_gen_loss": self.gen_loss_metric.result(),
            "gen_disc_loss": self.cross_entropy_metric.result(),
            "90_consistency_loss": self.josh_loss_metric.result(),
            '2d_comparison_loss': self.comparison_loss.result(),
            'bone_length_loss': self.bone_length_loss.result(),
            'elepose_loss': self.pairwise_loss.result(),
            '3d_comparison_loss': self.three_d_loss.result()

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
                self.model.legs_generator.save_weights(self.gen_filepath + '%04d' % (epoch + 1) + '_lower_gen.h5')
                self.model.torso_generator.save_weights(self.gen_filepath + '%04d' % (epoch + 1) + '_upper_gen.h5')
            if self.disc_filepath != None:
                self.model.discriminator.save_weights(self.disc_filepath + 'disc_weights%08d.h5' % (epoch + 1))
            if self.GAN_filepath != None:
                self.model.save_weights(self.GAN_filepath + 'GAN_weights%08d.h5' % (epoch + 1))


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    """Begin Actual Model---------------------------------------------------------------------------"""
    # Easy Parameter Tweaking
    batch_size = 8192
    num_joints = 17
    depth_offset = 10
    discriminator_input_size = (2, 17)
    legs_generator_input_size = (2, 7)
    torso_generator_input_size = (2, 10)  # The input to the generator and discriminator [16 (x,y) keypoints]'
    gen_opt = Adam(learning_rate=0.0002)
    disc_opt = Adam(learning_rate=0.0002)
    mean_squared_error = tf.keras.losses.MeanSquaredError()
    epochs = 800

    with open('h36m_train.pkl', 'rb') as f:
        train = pickle.load(f)

    two_d_keypoints = []
    # Normalise the data and increase the amount of training data by 2 by flipping along the x axis
    for t in train:
        keypoints = t['joints_3d']
        two_d_keypoints.append(keypoints[:, :2])
    two_d_keypoints = np.array(two_d_keypoints).transpose(0, 2, 1).reshape(-1, 2 * num_joints)

    train_data = normalize_head(two_d_keypoints)

    train_data = train_data.reshape(-1, 2, 17)

    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)

    torso_generator = build_torso_generator(torso_generator_input_size)
    legs_generator = build_legs_generator(legs_generator_input_size)

    discriminator = build_discriminator(discriminator_input_size)

    gen_filepath = 'no_occlusion_weights/MSE_5'

    gan = TIPSy(discriminator=discriminator, torso_generator=torso_generator, legs_generator=legs_generator,
                depth_offset=depth_offset)
    gan.compile(disc_opt=disc_opt, gen_opt=gen_opt, mse=mean_squared_error)
    model = gan.fit(train_data, epochs=epochs, verbose=2, callbacks=[GAN_Callback(gen_filepath=gen_filepath)])
