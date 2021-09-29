import keras.callbacks
from keras.layers import Input, Dense, Dropout, Add, BatchNormalization, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.metrics as M
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from tensorflow.keras.utils import to_categorical
import numpy as np
import keras.backend as K
import keras.layers
from keras import optimizers
from tensorflow.keras.layers import Layer
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
from typing import List, Tuple


# Residual Block from the paper (the paper uses relu and no dropout)
def residual_block(X):
    X_shortcut = X

    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.8)(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(0.3)(X)

    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.8)(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(0.3)(X)

    X = Add()([X, X_shortcut])
    return X


def tcn_residual_block(X, s, i, nb_filters, kernel_size, padding, dropout_rate=0, name=''):
    """Defines the residual block for the WaveNet TCN
    Args:
        X: The previous layer in the model
        s: The stack index i.e. which stack in the overall TCN
        i: The dilation power of 2 we are using for this residual block
        activation: The name of the type of activation to use
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        name: Name of the model. Useful when having multiple TCN.
    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    X_shortcut = X
    X = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding=padding,
               name=name + '_dilated_conv_%d_tanh_s%d' % (i, s))(X)

    X = BatchNormalization(momentum=0.8)(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_%d_s%d_%f' % (i, s, dropout_rate))(X)
    res_X = Add()([X, X_shortcut])
    return res_X, X



def build_fc_generator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = residual_block(X)
    X = residual_block(X)
    X = residual_block(X)
    X = residual_block(X)
    X = Dense(16, activation='tanh')(X)  # 16 outputs (z coordinate for every keypoint)
    model = Model(inputs=X_input, outputs=X, name='FC_Generator')
    return model


def build_tcn_generator(input_shape):
    X_input = Input(input_shape)
    X = TCN()(X_input)
    X = Dense(48, activation='tanh')(X)
    model = Model(inputs=X_input, outputs=X, name='TCN_Generator')
    return model


def build_discriminator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = residual_block(X)
    X = residual_block(X)
    X = residual_block(X)
    X = Dense(2, activation='softmax')(X)  # 2 class softmax as output (like the paper)
    model = Model(inputs=X_input, outputs=X, name='Discriminator')
    return model


def reprojection_layer(generated_3d_pose):
    new_poses = [_ for _ in range(len(generated_3d_pose))]
    for i in range(len(generated_3d_pose)):
        azimuth_angle = np.random.uniform(low=-np.pi, high=np.pi)
        elevation_angle = np.random.uniform(low=-np.pi / 9, high=np.pi / 9)
        azimuth_rotation = R.from_rotvec(azimuth_angle * np.array([0, 1, 0]))
        elevation_rotation = R.from_rotvec(elevation_angle * np.array([1, 0, 0]))
        azimuth_matrix = tf.constant(azimuth_rotation.as_matrix(), dtype=tf.float32)
        elevation_matrix = tf.constant(elevation_rotation.as_matrix(), dtype=tf.float32)
        rotated_pose = tf.matmul(generated_3d_pose[i], azimuth_matrix)
        rotated_pose = tf.matmul(rotated_pose, elevation_matrix)
        new_poses[i] = rotated_pose
    new_poses = tf.convert_to_tensor(new_poses)
    return new_poses[:, :, :2]


class TCN():
    """Creates a TCN layer.
        Args:
            input_layer: A tensor of shape (batch_size, timesteps, input_dim).
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self, nb_filters=64, kernel_size=3, nb_stacks=1, dilations=None, padding='causal',
                 use_skip_connections=True, dropout_rate=0.0, return_sequences=True, name='tcn'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' paddings are compatible for this layer.")

    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]
        X = inputs
        X = Convolution1D(self.nb_filters, 1, padding=self.padding, name=self.name + '_initial_conv')(X)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                X, skip_out = tcn_residual_block(X, s, i, self.nb_filters,
                                                    self.kernel_size, self.padding, self.dropout_rate, name=self.name)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            X = keras.layers.add(skip_connections)
        X = LeakyReLU(alpha=0.2)(X)

        if not self.return_sequences:
            output_slice_index = -1
            X = Lambda(lambda tt: tt[:, output_slice_index, :])(X)
        return X


class TCN_GAN(Model):
    """Create a TCN GAN
        Args:
            discriminator: the discriminator you want to use in the GAN
            generator: the generator you want to use in the GAN
            reprojector: the reprojection layer used to reproject the predicted 3D pose back into 2D for the discriminator
        Returns:
            generator and discriminator loss
    """

    def __init__(self, discriminator, generator, reprojector):
        super(TCN_GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.reprojector = reprojector

    def compile(self, disc_opt, gen_opt, loss_func):
        super(TCN_GAN, self).compile()
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.loss_func = loss_func
        self.disc_loss_metric = M.Mean(name='disc_loss')
        self.gen_loss_metric = M.Mean(name='gen_loss')

    @property
    def metrics(self):
        return [self.disc_loss_metric, self.gen_loss_metric]

    def train_step(self, pose_batch):
        # Split the batch in half, one half will train the discriminator, the other will train the generator
        batch_one, batch_two = tf.split(pose_batch, 2)
        if np.random.choice([0, 1]) == 0:
            gen_batch = batch_one
            disc_batch = batch_two
        else:
            gen_batch = batch_two
            disc_batch = batch_one

        # Generate fake 3D pose data
        generated_z = self.generator(disc_batch)
        generated_z = tf.reshape(generated_z, [disc_batch.shape[0], 16,
                                               1])  # Reshape purely for concatenation reasons as it is currently (256, 16)
        generated_3d_pose = tf.concat([disc_batch, generated_z], axis=2)

        reprojected_2d_pose = self.reprojector(generated_3d_pose)

        # Combine the reprojected_2d_poses with the original batch
        reprojected_2d_pose = tf.cast(reprojected_2d_pose, tf.float32)
        combined_poses = tf.concat([reprojected_2d_pose, disc_batch], axis=0)

        # Assemble labels discriminating real from reprojected poses as one hot
        labels = to_categorical(
            tf.concat([tf.ones((int(disc_batch.shape[0]), 1)), tf.zeros((int(disc_batch.shape[0]), 1))], axis=0),
            num_classes=2)

        # Adds random noise to the labels (comment out if not wanted)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            # Get the loss
            predictions = self.discriminator(combined_poses)
            disc_loss = self.loss_func(labels, predictions)
        # Compute and apply the gradients
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(disc_gradients, self.discriminator.trainable_weights))

        # Labels that say all the poses are real
        misleading_labels = to_categorical(tf.zeros((int(gen_batch.shape[0]), 1)), num_classes=2)

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generated_z = self.generator(gen_batch)
            generated_z = tf.reshape(generated_z, [gen_batch.shape[0], 16,
                                                   1])  # Reshape purely for concatenation reasons as it is currently (256, 16)
            generated_3d_pose = tf.concat([gen_batch, generated_z], axis=2)
            reprojected_2d_pose = self.reprojector(generated_3d_pose)
            predictions = self.discriminator(reprojected_2d_pose)
            gen_loss = self.loss_func(misleading_labels, predictions)
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_opt.apply_gradients(zip(gen_gradients, self.generator.trainable_weights))

        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(gen_loss)
        return {
            "disc_loss": self.disc_loss_metric.result(),
            "gen_loss": self.gen_loss_metric.result()
        }


class Binary_GAN(Model):
    """Create a Binary GAN
        Args:
            discriminator: the discriminator you want to use in the GAN
            generator: the generator you want to use in the GAN
            reprojector: the reprojection layer used to reproject the predicted 3D pose back into 2D for the discriminator
        Returns:
            generator and discriminator loss
    """

    def __init__(self, discriminator, generator, reprojector):
        super(Binary_GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.reprojector = reprojector

    def compile(self, disc_opt, gen_opt, loss_func):
        super(Binary_GAN, self).compile()
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.loss_func = loss_func
        self.disc_loss_metric = M.Mean(name='disc_loss')
        self.gen_loss_metric = M.Mean(name='gen_loss')

    @property
    def metrics(self):
        return [self.disc_loss_metric, self.gen_loss_metric]

    def train_step(self, pose_batch):
        # Split the batch in half, one half will train the discriminator, the other will train the generator
        batch_one, batch_two = tf.split(pose_batch, 2)
        if np.random.choice([0, 1]) == 0:
            gen_batch = batch_one
            disc_batch = batch_two
        else:
            gen_batch = batch_two
            disc_batch = batch_one

        # Generate fake 3D pose data
        generated_z = self.generator(disc_batch)
        generated_z = tf.reshape(generated_z, [disc_batch.shape[0], 16,
                                               1])  # Reshape purely for concatenation reasons as it is currently (256, 16)
        generated_3d_pose = tf.concat([disc_batch, generated_z], axis=2)

        reprojected_2d_pose = self.reprojector(generated_3d_pose)

        # Combine the reprojected_2d_poses with the original batch
        reprojected_2d_pose = tf.cast(reprojected_2d_pose, tf.float32)
        combined_poses = tf.concat([reprojected_2d_pose, disc_batch], axis=0)

        # Assemble labels discriminating real from reprojected poses as one hot
        labels = to_categorical(
            tf.concat([tf.ones((int(disc_batch.shape[0]), 1)), tf.zeros((int(disc_batch.shape[0]), 1))], axis=0),
            num_classes=2)

        # Adds random noise to the labels (comment out if not wanted)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            # Get the loss
            predictions = self.discriminator(combined_poses)
            disc_loss = self.loss_func(labels, predictions)
        # Compute and apply the gradients
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(disc_gradients, self.discriminator.trainable_weights))

        # Labels that say all the poses are real
        misleading_labels = to_categorical(tf.zeros((int(gen_batch.shape[0]), 1)), num_classes=2)

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generated_z = self.generator(gen_batch)
            generated_z = tf.reshape(generated_z, [gen_batch.shape[0], 16,
                                                   1])  # Reshape purely for concatenation reasons as it is currently (256, 16)
            generated_3d_pose = tf.concat([gen_batch, generated_z], axis=2)
            reprojected_2d_pose = self.reprojector(generated_3d_pose)
            predictions = self.discriminator(reprojected_2d_pose)
            gen_loss = self.loss_func(misleading_labels, predictions)
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_opt.apply_gradients(zip(gen_gradients, self.generator.trainable_weights))

        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(gen_loss)
        return {
            "disc_loss": self.disc_loss_metric.result(),
            "gen_loss": self.gen_loss_metric.result()
        }


class GAN_Callback(keras.callbacks.Callback):
    def __init__(self, gen_filepath=None, disc_filepath=None, GAN_filepath=None):
        self.gen_filepath = gen_filepath
        self.disc_filepath = disc_filepath
        self.GAN_filepath = GAN_filepath

    def on_epoch_end(self, epoch, logs=None):
        # Variable to control how many epochs to save weights
        N = 1
        if (epoch + 1) % N == 0:
            if self.gen_filepath != None:
                self.model.generator.save_weights(self.gen_filepath + 'gen_weights%08d.h5' % (epoch + 1))
            if self.disc_filepath != None:
                self.model.discriminator.save_weights(self.disc_filepath + 'disc_weights%08d.h5' % (epoch + 1))
            if self.GAN_filepath != None:
                self.model.save_weights(self.GAN_filepath + 'GAN_weights%08d.h5' % (epoch + 1))
