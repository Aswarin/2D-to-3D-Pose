from common import utils
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':


    tf.config.run_functions_eagerly(True)

    """Begin Actual Model---------------------------------------------------------------------------"""

    # Easy Parameter Tweaking
    gen_input_shape = (12, 32)  # generator input of 12 frames processed with 1 dimension
    disc_input_shape = (16, 2)  # discriminator input with 16 keypoints with x,y coordinates
    batch_size = 128 #batch size is actually half because gen will see half and disc will see half
    gen_opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    disc_opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    epochs = 30

    # Import Data
    #with open('datasets/h36m_train_video_anno.json') as f:
    #    train_data = json.load(f)

    #train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)

    generator = utils.build_tcn_generator(gen_input_shape)

    discriminator = utils.build_discriminator(disc_input_shape)

    gen_filepath = 'training_checkpoints/amazon_lab_paper1/gen_weights/'
    disc_filepath = 'training_checkpoints/amazon_lab_paper1/disc_weights/'
    GAN_filepath = 'training_checkpoints/amazon_lab_paper1/GAN_weights/'

    gan = utils.TCN_GAN(discriminator=discriminator, generator=generator, reprojector=utils.reprojection_layer)
    gan.compile(disc_opt=disc_opt, gen_opt=gen_opt, loss_func=cross_entropy)
    gan.fit(train_data, epochs=epochs, verbose=2, callbacks=[utils.GAN_Callback()])