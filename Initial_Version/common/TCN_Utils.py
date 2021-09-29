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