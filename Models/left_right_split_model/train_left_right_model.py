import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Add, Flatten, Dense, BatchNormalization, ReLU, Input
from keras.models import Model
from scipy.spatial.transform import Rotation as R
from numpy.random import choice
import numpy as np
import keras.metrics as M
from tensorflow.keras.optimizers import Adam
from scipy import linalg
import pickle
from tensorflow.python.ops.numpy_ops import np_config
import functools
np_config.enable_numpy_behavior()

def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = tf.math.cos(angle)
    sin = tf.math.sin(angle)
    one = tf.ones_like(angle)
    zero = tf.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return tf.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.size == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, tf.unstack(euler_angles, axis=-1))
    return functools.reduce(tf.matmul, matrices)


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


def build_split_generator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = ReLU()(X)
    # Pose Path
    X_pose = residual_block(X, 0.5)
    X_pose = residual_block(X_pose, 0.5)
    X_pose = Dense(10, activation='tanh')(X_pose)  # 16 outputs (z coordinate for every keypoint)

    # Angle Path
    X_angle = residual_block(X, 0.5)
    X_angle = residual_block(X_angle, 0.5)
    X_angle = Dense(1)(X_angle)
    model = Model(inputs=X_input, outputs=[X_pose, X_angle], name='torso_generator')
    return model


def build_discriminator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = ReLU()(X)
    X = disc_residual_block(X, 0.5)
    X = Dense(1, activation='linear')(X)  # 2 class softmax as output (like the paper)
    model = Model(inputs=X_input, outputs=X, name='Discriminator')
    return model

def get_bone_lengths_all(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]]

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = tf.norm(bones, ord=2, axis=1)

    return bone_lengths


def combine_lr_split(left_split, right_split):
    split_average_mask = np.array((1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1))
    scale_left_split = left_split * split_average_mask
    scale_right_split = right_split * split_average_mask
    full_pose = tf.concat([tf.expand_dims(scale_right_split[:, 0], axis=-1),
                           tf.expand_dims(scale_right_split[:, 1], axis=-1),
                           tf.expand_dims(scale_right_split[:, 2], axis=-1),
                           tf.expand_dims(scale_left_split[:, 0], axis=-1),
                           tf.expand_dims(scale_left_split[:, 1], axis=-1),
                           tf.expand_dims(scale_left_split[:, 2], axis=-1),
                           tf.expand_dims((scale_right_split[:, 3] + scale_left_split[:, 3]), axis=-1),
                           tf.expand_dims((scale_right_split[:, 4] + scale_left_split[:, 4]), axis=-1),
                           tf.expand_dims((scale_right_split[:, 5] + scale_left_split[:, 5]), axis=-1),
                           tf.expand_dims((scale_right_split[:, 6] + scale_left_split[:, 6]), axis=-1),
                           tf.expand_dims(scale_left_split[:, 7], axis=-1),
                           tf.expand_dims(scale_left_split[:, 8], axis=-1),
                           tf.expand_dims(scale_left_split[:, 9], axis=-1),
                           tf.expand_dims(scale_right_split[:, 7], axis=-1),
                           tf.expand_dims(scale_right_split[:, 8], axis=-1),
                           tf.expand_dims(scale_right_split[:, 9], axis=-1)], axis=-1)
    full_pose = tf.expand_dims(full_pose, -1)
    full_pose = tf.cast(full_pose, tf.float32)
    return full_pose


def split_data(data):
    if len(data.shape) > 2:
        right = tf.concat([tf.expand_dims(data[:, 0, :], axis=1), tf.expand_dims(data[:, 1, :], axis=1),
                           tf.expand_dims(data[:, 2, :], axis=1), tf.expand_dims(data[:, 6, :], axis=1),
                           tf.expand_dims(data[:, 7, :], axis=1), tf.expand_dims(data[:, 8, :], axis=1),
                           tf.expand_dims(data[:, 9, :], axis=1), tf.expand_dims(data[:, 13, :], axis=1),
                           tf.expand_dims(data[:, 14, :], axis=1), tf.expand_dims(data[:, 15, :], axis=1)], axis=1)
        left = tf.concat([tf.expand_dims(data[:, 3, :], axis=1), tf.expand_dims(data[:, 4, :], axis=1),
                          tf.expand_dims(data[:, 5, :], axis=1), tf.expand_dims(data[:, 6, :], axis=1),
                          tf.expand_dims(data[:, 7, :], axis=1), tf.expand_dims(data[:, 8, :], axis=1),
                          tf.expand_dims(data[:, 9, :], axis=1), tf.expand_dims(data[:, 10, :], axis=1),
                          tf.expand_dims(data[:, 11, :], axis=1), tf.expand_dims(data[:, 12, :], axis=1)], axis=1)
    elif len(data.shape) > 1:
        right = tf.concat([tf.expand_dims(data[:, 0], axis=1), tf.expand_dims(data[:, 1], axis=1),
                           tf.expand_dims(data[:, 2], axis=1), tf.expand_dims(data[:, 6], axis=1),
                           tf.expand_dims(data[:, 7], axis=1), tf.expand_dims(data[:, 8], axis=1),
                           tf.expand_dims(data[:, 9], axis=1), tf.expand_dims(data[:, 13], axis=1),
                           tf.expand_dims(data[:, 14], axis=1), tf.expand_dims(data[:, 15], axis=1)], axis=1)
        left = tf.concat([tf.expand_dims(data[:, 3], axis=1), tf.expand_dims(data[:, 4], axis=1),
                          tf.expand_dims(data[:, 5], axis=1), tf.expand_dims(data[:, 6], axis=1),
                          tf.expand_dims(data[:, 7], axis=1), tf.expand_dims(data[:, 8], axis=1),
                          tf.expand_dims(data[:, 9], axis=1), tf.expand_dims(data[:, 10], axis=1),
                          tf.expand_dims(data[:, 11], axis=1), tf.expand_dims(data[:, 12], axis=1)], axis=1)
    else:
        right = tf.concat([data[0], data[1], data[2], data[6], data[7], data[8],
                           data[9], data[13], data[14], data[15]], axis=0)
        left = tf.concat([data[3], data[4], data[5], data[6],
                          data[7], data[8], data[9], data[10], data[11], data[12]], axis=0)
    return left, right


class LR_Split_Lifter(Model):
    """Create a Binary GAN
        Args:
            discriminator: the discriminator you want to use in the GAN
            generator: the generator you want to use in the GAN
            reprojector: the reprojection layer used to reproject the predicted 3D pose back into 2D for the discriminator
        Returns:
            generator and discriminator loss
    """

    def __init__(self):
        super(LR_Split_Lifter, self).__init__()
        lifter_input = (10, 2)
        disc_input = (16, 2)
        self.discriminator = build_discriminator(disc_input)
        self.left_lifter = build_split_generator(lifter_input)
        self.right_lifter = build_split_generator(lifter_input)
        self.bone_lengths_mean = np.array([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
                                           0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652])

    def compile(self, disc_opt, gen_opt, mse):
        super(LR_Split_Lifter, self).compile()
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.mse = mse
        self.disc_loss_metric = M.Mean(name='disc_loss')
        self.gen_loss_metric = M.Mean(name='gen_loss')
        self.josh_loss_metric = M.Mean(name='self_consistency_loss')
        self.comparison_loss = M.Mean(name='Two D Comparison Loss')
        self.three_d_loss = M.Mean(name='Three D Comparison Loss')
        self.gen_disc_loss = M.Mean(name='gen_disc_loss')
        self.bone_lengths_loss = M.Mean(name='bone loss')
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

        # split data into left and right hand side
        left_pose_input, right_pose_input = split_data(pose_batch)

        # Generate fake 3D pose data
        generated_left_z, left_angle = self.left_lifter(left_pose_input)
        generated_right_z, right_angle = self.right_lifter(right_pose_input)
        angle = (left_angle + right_angle) / 2
        generated_z = combine_lr_split(generated_left_z, generated_right_z)
        generated_3d_pose = tf.concat([pose_batch, generated_z], axis=2)

        """One Random Rotation 3D rotations"""
        x_ang_comp = tf.ones([pose_batch.shape[0], 1]) * angle
        y_ang_comp = tf.zeros([pose_batch.shape[0], 1])
        z_ang_comp = tf.zeros([pose_batch.shape[0], 1])
        euler_angles_comp = tf.concat([x_ang_comp, y_ang_comp, z_ang_comp], axis=1)
        R_comp = euler_angles_to_matrix(euler_angles_comp, 'XYZ')
        elevation = tf.concat([angle.mean().reshape(1), tf.math.reduce_std(angle).reshape(1)], axis=0)
        x_ang = (-elevation[0]) + elevation[1] * tf.random.normal([pose_batch.shape[0], 1], mean=tf.zeros([pose_batch.shape[0], 1]), stddev=tf.ones([pose_batch.shape[0], 1]))
        y_ang = (tf.random.uniform([pose_batch.shape[0], 1]) -0.5) * 1.99 * np.pi
        z_ang = tf.zeros([pose_batch.shape[0], 1])
        Rx = euler_angles_to_matrix(tf.concat([x_ang, z_ang, z_ang], axis=1), 'XYZ')
        Ry = euler_angles_to_matrix(tf.concat([z_ang, y_ang, z_ang], axis=1), 'XYZ')
        R = Rx @ (Ry @ R_comp)
        new_poses = tf.matmul(generated_3d_pose, R)
        new_poses = tf.convert_to_tensor(new_poses)
        """End one random 3D rotation"""


        # reproject the random rotation back into 2D
        reprojected_2d_pose = new_poses[:, :, :2]

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
            generated_left_z, left_angle = self.left_lifter(left_pose_input)
            generated_right_z, right_angle = self.right_lifter(right_pose_input)
            angle = (left_angle + right_angle) / 2
            generated_z = combine_lr_split(generated_left_z, generated_right_z)
            generated_3d_pose = tf.concat([pose_batch, generated_z], axis=2)

            """90 degree CONSISTENCY LOSS"""
            z_y_minus_x = tf.concat([tf.reshape(generated_3d_pose[:, :, 2], [pose_batch.shape[0], 16, 1]),
                                     tf.reshape(generated_3d_pose[:, :, 1], [pose_batch.shape[0], 16, 1])],
                                    axis=2)

            minus_z_y_x = tf.concat([tf.reshape(-generated_3d_pose[:, :, 2], [pose_batch.shape[0], 16, 1]),
                                     tf.reshape(generated_3d_pose[:, :, 1], [pose_batch.shape[0], 16, 1])],
                                    axis=2)

            ls_inp, rs_inp = split_data(z_y_minus_x)
            left_check, _ = self.left_lifter(ls_inp)
            right_check, _ = self.right_lifter(rs_inp)
            left_90_loss = self.mse(left_pose_input[:, :, 0], -left_check)
            right_90_loss = self.mse(right_pose_input[:, :, 0], -right_check)
            ls_inp, rs_inp = split_data(minus_z_y_x)
            left_check, _ = self.left_lifter(ls_inp)
            right_check, _ = self.right_lifter(rs_inp)
            left_90_loss += self.mse(left_pose_input[:, :, 0], left_check)
            right_90_loss += self.mse(right_pose_input[:, :, 0], right_check)
            """90 degree CONSISTENCY LOSS"""

            """Reprojection back into the original 2D input dimension loss"""
            """Reprojection back into the original 2D input dimension loss"""
            x_ang_comp = tf.ones([pose_batch.shape[0], 1]) * angle
            y_ang_comp = tf.zeros([pose_batch.shape[0], 1])
            z_ang_comp = tf.zeros([pose_batch.shape[0], 1])
            euler_angles_comp = tf.concat([x_ang_comp, y_ang_comp, z_ang_comp], axis=1)
            R_comp = euler_angles_to_matrix(euler_angles_comp, 'XYZ')
            elevation = tf.concat([angle.mean().reshape(1), tf.math.reduce_std(angle).reshape(1)], axis=0)
            x_ang = (-elevation[0]) + elevation[1] * tf.random.normal([pose_batch.shape[0], 1],
                                                                      mean=tf.zeros([pose_batch.shape[0], 1]),
                                                                      stddev=tf.ones([pose_batch.shape[0], 1]))
            y_ang = (tf.random.uniform([pose_batch.shape[0], 1]) - 0.5) * 1.99 * np.pi
            z_ang = tf.zeros([pose_batch.shape[0], 1])
            Rx = euler_angles_to_matrix(tf.concat([x_ang, z_ang, z_ang], axis=1), 'XYZ')
            Ry = euler_angles_to_matrix(tf.concat([z_ang, y_ang, z_ang], axis=1), 'XYZ')
            R = Rx @ (Ry @ R_comp)
            rotated_pose = tf.matmul(generated_3d_pose, R)
            rotated_pose = tf.convert_to_tensor(rotated_pose)
            reprojected_2d_pose = rotated_pose[:, :, :2]
            left_comp_inp, right_comp_inp = split_data(reprojected_2d_pose)
            left_comp_z, _ = self.left_lifter(left_comp_inp)
            right_comp_z, _ = self.right_lifter(right_comp_inp)
            comparison_z = combine_lr_split(left_comp_z, right_comp_z)
            comparison_3d_pose = tf.concat([reprojected_2d_pose, comparison_z], axis=2)
            R_inv = tf.transpose(R, perm=[0, 2, 1])
            rotated_comparison_3d_pose = tf.matmul(comparison_3d_pose, R_inv)
            two_d_comparison = rotated_comparison_3d_pose[:, :, :2]
            left_2d_comp, right_2d_comp = split_data(two_d_comparison)
            left_comp_2d_loss = self.mse(left_pose_input, left_2d_comp)
            right_comp_2d_loss = self.mse(right_pose_input, right_2d_comp)
            """End reprojection into the original 2D input dimension loss"""

            """Discriminator Loss"""
            # Labels that say all the poses are real for generator training
            misleading_labels = tf.zeros(int(pose_batch.shape[0]))
            predictions = self.discriminator(reprojected_2d_pose)
            gen_disc_loss = self.mse(misleading_labels, predictions)
            """Discriminator Loss"""


            """Bone relations loss"""
            roots = tf.zeros([pose_batch.shape[0], 1, 3])
            full_3d_pose = tf.concat([roots, generated_3d_pose], axis=1)
            full_3d_comp_pose = tf.concat([roots, comparison_3d_pose], axis=1)
            og_bl = get_bone_lengths_all(tf.transpose(full_3d_pose, perm=[0,2,1]).reshape(-1, 51))
            comp_bl = get_bone_lengths_all(tf.transpose(full_3d_comp_pose, perm=[0,2,1]).reshape(-1, 51))
            rel_og_bl = og_bl / og_bl.mean(axis=1, keepdims=True)
            rel_comp_bl = comp_bl / comp_bl.mean(axis=1, keepdims=True)
            left_bl, right_bl = split_data(rel_og_bl)
            left_comp_bl, right_comp_bl = split_data(rel_comp_bl)
            left_bl_gt, right_bl_gt = split_data(self.bone_lengths_mean)
            left_bone_length_loss = np.mean(np.sum((left_bl - left_bl_gt) ** 2, axis=1))
            left_bone_length_loss += np.mean(np.sum((left_comp_bl - left_bl_gt) ** 2, axis=1))
            right_bone_length_loss = np.mean(np.sum((right_bl - right_bl_gt) ** 2, axis=1))
            right_bone_length_loss += np.mean(np.sum((right_comp_bl - right_bl_gt) ** 2, axis=1))
            """Bone relations loss"""

            left_loss = (gen_disc_loss * 0.5) + (left_comp_2d_loss * 10) + (left_90_loss * 3) + (left_bone_length_loss * 50)
            right_loss = (gen_disc_loss * 0.5) + (right_comp_2d_loss * 10) + (right_90_loss * 3) + (right_bone_length_loss * 50)

        left_gen_gradients = gen_tape.gradient(left_loss, self.left_lifter.trainable_weights)
        right_gen_gradients = gen_tape.gradient(right_loss, self.right_lifter.trainable_weights)
        del gen_tape

        self.gen_opt.apply_gradients(zip(left_gen_gradients, self.left_lifter.trainable_weights))
        self.gen_opt.apply_gradients(zip(right_gen_gradients, self.right_lifter.trainable_weights))
        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(left_loss + right_loss)
        self.josh_loss_metric.update_state(left_90_loss + right_90_loss)
        self.gen_disc_loss.update_state(gen_disc_loss)
        self.comparison_loss.update_state(left_comp_2d_loss + right_comp_2d_loss)
        self.bone_lengths_loss.update_state(left_bone_length_loss + right_bone_length_loss)
        return {
            "disc_loss": self.disc_loss_metric.result(),
            "gen_loss": self.gen_loss_metric.result(),
            "gen_disc_loss": self.gen_disc_loss.result(),
            "90_consistency_loss": self.josh_loss_metric.result(),
            'two_d_comparison_loss': self.comparison_loss.result(),
            'bone length loss': self.bone_lengths_loss.result()
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
                self.model.left_lifter.save_weights(self.gen_filepath + '%04d' % (epoch + 1) + '_left_lifter.h5')
                self.model.right_lifter.save_weights(self.gen_filepath + '%04d' % (epoch + 1) + '_right_lifter.h5')
            if self.disc_filepath != None:
                self.model.discriminator.save_weights(self.disc_filepath + 'disc_weights%08d.h5' % (epoch + 1))
            if self.GAN_filepath != None:
                self.model.save_weights(self.GAN_filepath + 'GAN_weights%08d.h5' % (epoch + 1))


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    """Begin Actual Model---------------------------------------------------------------------------"""
    # Easy Parameter Tweaking
    batch_size = 8192
    gen_opt = Adam(learning_rate=0.0002)
    disc_opt = Adam(learning_rate=0.0002)
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

    gen_filepath = 'left_right_split_weights/MSE_1/'

    gan = LR_Split_Lifter()
    gan.compile(disc_opt=disc_opt, gen_opt=gen_opt, mse=mean_squared_error)
    model = gan.fit(train_data, epochs=epochs, verbose=2, callbacks=[GAN_Callback(gen_filepath=gen_filepath)])
