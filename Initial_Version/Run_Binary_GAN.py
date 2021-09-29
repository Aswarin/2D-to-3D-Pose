from common import utils
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json


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

    generator = utils.build_fc_generator(input_size)

    discriminator = utils.build_discriminator(input_size)

    gen_filepath = 'training_checkpoints/amazon_lab_paper1/gen_weights/'
    disc_filepath = 'training_checkpoints/amazon_lab_paper1/disc_weights/'
    GAN_filepath = 'training_checkpoints/amazon_lab_paper1/GAN_weights/'

    gan = utils.Binary_GAN(discriminator=discriminator, generator=generator, reprojector=utils.reprojection_layer)
    gan.compile(disc_opt=disc_opt, gen_opt=gen_opt, loss_func=cross_entropy)
    gan.fit(train_data, epochs=epochs, verbose=2, callbacks=[utils.GAN_Callback(gen_filepath, disc_filepath, GAN_filepath)])