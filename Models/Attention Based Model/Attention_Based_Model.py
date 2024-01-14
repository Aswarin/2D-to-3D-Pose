import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Add, Flatten, Dense, BatchNormalization, ReLU, Input, Layer
from keras.models import Model
from scipy.spatial.transform import Rotation as R
from numpy.random import choice
import numpy as np
import keras.metrics as M
from tensorflow.keras.optimizers import Adam
from scipy import linalg
import pickle

class mask_layer(Layer):
    def __init__(self, units=16, input_dim=1):
        super(mask_layer, self).__init__
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs, training=None):
        x = tf.matmul(inputs, self.w) + self.b
        x = 1 / (1 + tf.exp(-x))  # sigmoid activation
        if training:
            return x



# Residual Block from the paper (the paper uses relu and no dropout)
def residual_block(X, neuron_count, d_rate=0.5, lifter=False):
    X_shortcut = X

    X = Dense(neuron_count)(X)
    if lifter == True:
        X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Dense(neuron_count)(X)
    if lifter == True:
        X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Add()([X, X_shortcut])

    return X


def split_lifter(input_shape, neuron_count, residual_block_count, mask):
    X_input = Input(input_shape)

    X = tf.math.multiply(X_input, mask)

    X = Flatten()(X)
    X = Dense(neuron_count)(X)
    for _ in range(residual_block_count):
        X = residual_block(X, neuron_count, lifter=True)
    return X, mask




def build_attention_lifter(input_shape, split_number, neuron_count, residual_block_count):
    features = []
    mask = mask_layer(16)([42,3])
    for i in range(split_number):
        f, m = split_lifter(input_shape=input_shape, neuron_count=neuron_count, residual_block_count=residual_block_count, mask=masks[i])
        features.append(f)
        masks.append(m)




    X = Dense(16, activation='tanh')(X)  # 16 outputs (z coordinate for every keypoint)
    model = Model(inputs=X_input, outputs=[X, mask], name='Student')
    return model









def build_discriminator(input_shape, neuron_count, residual_block_count):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(neuron_count)(X)
    for _ in range(residual_block_count):
        X = residual_block(X, neuron_count)
    X = Dense(1, activation='linear')(X)  # 2 class softmax as output (like the paper)
    model = Model(inputs=X_input, outputs=X, name='Discriminator')
    return model


class Attention_Based_GAN(Model):
    def __init__(self, split_number, residual_block_count, dense_layer_neurons):
        super(Attention_Based_GAN, self).__init__()
        input_shape = (16, 2)
        self.attention_lifter = build_attention_lifter(input_shape=input_shape, split_number=split_number, residual_block_count=residual_block_count, neuron_count=dense_layer_neurons)
        self.discriminator = build_discriminator(input_shape, dense_layer_neurons, int((residual_block_count * split_number)/2))

    def compile(self, student_opt, disc_opt, cross_entropy_loss_func,
                mse):
        super(Attention_Based_GAN, self).compile()
        self.disc_opt = disc_opt
        self.student_opt = student_opt
        self.cross_entropy_loss_func = cross_entropy_loss_func
        self.mse = mse
        self.disc_loss_metric = M.Mean(name='disc_loss')
        self.solo_gen_loss_metric = M.Mean(name='lifting_network_loss')
        self.comparison_loss = M.Mean(name='Two D Comparison Loss')
        self.cross_entropy_metric = M.Mean(name='Cross Entropy Loss')
        self.ninety_degree_metric = M.Mean(name='90 degree loss')
        self.three_d_metric = M.Mean(name='3D Metric')

    def train_step(self, pose_batch):
        # Randomly flip the input pose batch along the y-axis
        if np.random.choice([0, 1]) == 0:
            x_pose = pose_batch[:, :, 0]
            y_pose = pose_batch[:, :, 1]
            x_pose *= -1
            pose_batch = tf.stack([x_pose, y_pose], axis=2)

        # Begin discriminator training which will fight the student
        student_predicted_z = self.student(pose_batch)
        student_predicted_z = tf.reshape(student_predicted_z, [pose_batch.shape[0], 16, 1])
        student_predicted_3d_pose = tf.concat([pose_batch, student_predicted_z], axis=2)

        """randomly rotate the student predicted 3d pose"""
        azimuth_angle = np.random.uniform(low=-(8 / 9) * np.pi, high=(8 / 9) * np.pi)
        elevation_angle = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)
        azimuth_rotation = R.from_rotvec(azimuth_angle * np.array([0, 1, 0]))
        elevation_rotation = R.from_rotvec(elevation_angle * np.array([1, 0, 0]))
        azimuth_matrix = tf.constant(azimuth_rotation.as_matrix(), dtype=tf.float32)
        elevation_matrix = tf.constant(elevation_rotation.as_matrix(), dtype=tf.float32)
        rotated_pose = tf.matmul(tf.cast(student_predicted_3d_pose, dtype=tf.float32), azimuth_matrix)
        rotated_pose = tf.matmul(rotated_pose, elevation_matrix)
        rotated_pose = tf.convert_to_tensor(rotated_pose)
        """End random 3D rotation"""

        # reproject the random rotation back into 2D
        student_reprojected_2d_pose = rotated_pose[:, :, :2]

        # Generate the labels for 'real' data
        real_labels = tf.ones(int(pose_batch.shape[0]))

        # generate the labels for the 'fake' data
        fake_labels = tf.zeros(int(pose_batch.shape[0]))

        combined_poses = tf.concat([student_reprojected_2d_pose, pose_batch], axis=0)

        # Randomly Flip labels
        if np.random.randint(1, 100) <= 10:
            labels = tf.concat([real_labels, fake_labels], axis=0)
        else:
            labels = tf.concat([fake_labels, real_labels], axis=0)

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            # Get the loss
            disc_predictions = self.discriminator(combined_poses)
            disc_loss = self.mse(labels, disc_predictions)

        # Compute and apply the gradients only if discriminator isn't too strong
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(disc_gradients, self.discriminator.trainable_weights))

        """Begin lifter training"""
        return {
            "disc_loss": self.disc_loss_metric.result(),
            "lifting_network_loss": self.solo_gen_loss_metric.result(),
            "cross_entropy_loss": self.cross_entropy_metric.result(),
            'two_d_comparison_loss': self.comparison_loss.result(),
            '90 degree loss': self.ninety_degree_metric.result(),
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
                self.model.student.save_weights(self.gen_filepath + '%04d' % (epoch + 1) + '_solo_weights.h5')
            if self.disc_filepath != None:
                self.model.discriminator.save_weights(self.disc_filepath + 'disc_weights%08d.h5' % (epoch + 1))
            if self.GAN_filepath != None:
                self.model.save_weights(self.GAN_filepath + 'GAN_weights%08d.h5' % (epoch + 1))

import json
if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    """Begin Actual Model---------------------------------------------------------------------------"""

    # Easy Parameter Tweaking
    batch_size = 500
    student_opt = Adam(learning_rate=0.0002)
    disc_opt = Adam(learning_rate=0.0002)
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    mean_squared_error = tf.keras.losses.MeanSquaredError()
    epochs = 800

    with open('correct_normalised_2d_train_poses.json') as f:
        train_data = json.load(f)


    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)

    gen_filepath = 'attention_based_model/'

    gan = Attention_Based_GAN(split_number=1, residual_block_count=4, dense_layer_neurons=256)
    gan.compile(student_opt=student_opt, disc_opt=disc_opt, cross_entropy_loss_func=cross_entropy,
                mse=mean_squared_error)
    model = gan.fit(train_data, epochs=epochs, verbose=2, callbacks=[GAN_Callback(gen_filepath=gen_filepath)])
