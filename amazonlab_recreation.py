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
import json


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


def build_generator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = residual_block(X)
    X = residual_block(X)
    X = residual_block(X)
    X = residual_block(X)
    X = Dense(16, activation='tanh')(X)  # 16 outputs (z coordinate for every keypoint)
    model = Model(inputs=X_input, outputs=X, name='Generator')
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


class GAN(Model):
    def __init__(self, discriminator, generator, reprojector):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.reprojector = reprojector

    def compile(self, disc_opt, gen_opt, loss_func):
        super(GAN, self).compile()
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
                                               1]) # Reshape purely for concatenation reasons as it is currently (256, 16)
        generated_3d_pose = tf.concat([disc_batch, generated_z], axis=2)


        reprojected_2d_pose = self.reprojector(generated_3d_pose)

        # Combine the reprojected_2d_poses with the original batch
        reprojected_2d_pose = tf.cast(reprojected_2d_pose, tf.float32)
        combined_poses = tf.concat([reprojected_2d_pose, disc_batch], axis=0)

        # Assemble labels discriminating real from reprojected poses as one hot
        labels = to_categorical(tf.concat([tf.ones((int(disc_batch.shape[0]), 1)), tf.zeros((int(disc_batch.shape[0]), 1))], axis=0), num_classes=2)

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
    def on_epoch_end(self, epoch, logs=None):
        # Variable to control how many epochs to save weights
        N = 1
        if (epoch + 1) % N == 0:
            self.model.generator.save_weights(gen_filepath + 'gen_weights%08d.h5' % (epoch + 1))
            #self.model.discriminator.save_weights(disc_filepath + 'disc_weights%08d.h5' % (epoch + 1))
            #self.model.save_weights(GAN_filepath + 'GAN_weights%08d.h5' % (epoch + 1))


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    """Begin Actual Model---------------------------------------------------------------------------"""

    # Easy Parameter Tweaking
    batch_size = 8192
    input_size = (16, 2)  # The input to the generator and discriminator [16 (x,y) keypoints]
    gen_opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    disc_opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    epochs = 30

    # Import Data
    with open('datasets/ECCV18_Challenge/Train/NORM_POSE/normalised_2d_train_poses.json') as f:
        train_data = json.load(f)

    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)

    generator = build_generator(input_size)

    discriminator = build_discriminator(input_size)

    gen_filepath = 'training_checkpoints/amazon_lab_paper1/gen_weights/'
    disc_filepath = 'training_checkpoints/amazon_lab_paper1/disc_weights/'
    GAN_filepath = 'training_checkpoints/amazon_lab_paper1/GAN_weights/'

    gan = GAN(discriminator=discriminator, generator=generator, reprojector=reprojection_layer)
    gan.compile(disc_opt=disc_opt, gen_opt=gen_opt, loss_func=cross_entropy)
    gan.fit(train_data, epochs=epochs, verbose=2, callbacks=[GAN_Callback()])