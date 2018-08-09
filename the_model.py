# coding: utf-8
"""
    One of our best models
    This model achieves 97.2% top-1 accuracy on 2013 CASIA competition data,
    better than any previously published results.
"""

from keras.models import Model
from keras.layers import (
        Input,
        Flatten,
        Dense,
        ZeroPadding2D,
        Conv2D,
        Activation,
        MaxPooling2D,
        BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU


def relu():
    return LeakyReLU(alpha=0.01)


def conv_unit(input_tensor, nb_filters, mp=False, dropout=None):
    """
    one conv-relu-bn unit
    """
    x = ZeroPadding2D()(input_tensor)
    x = Conv2D(nb_filters, (3, 3))(x)
    x = relu()(x)
    x = BatchNormalization(axis=3, momentum=0.66)(x)

    if mp:
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    return x


def out_block(input_tensor, nb_classes):
    """
    FC output
    """
    x = Flatten()(input_tensor)
    x = Dense(1024)(x)
    x = relu()(x)
    x = BatchNormalization(momentum=0.66)(x)
    x = Dense(256)(x)
    x = relu()(x)
    x = BatchNormalization(momentum=0.66)(x)
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)
    return x


def model_8(img_size, num_classes):
    """
    This is actually model N2B
    5 blocks, 14 weight layers (1-2-2-3-3--3)
    """
    inputs = Input(shape=(img_size, img_size, 1))
    x = ZeroPadding2D()(inputs)
    x = Conv2D(64, (3, 3), strides=(2, 2))(x)
    x = relu()(x)
    x = BatchNormalization(momentum=0.66)(x)
    x = conv_unit(x, 128)
    x = conv_unit(x, 128, mp=True)
    x = conv_unit(x, 256)
    x = conv_unit(x, 256, mp=True)
    x = conv_unit(x, 384)
    x = conv_unit(x, 384)
    x = conv_unit(x, 384, mp=True)
    x = conv_unit(x, 512)
    x = conv_unit(x, 512)
    x = conv_unit(x, 512, mp=True)
    x = out_block(x, num_classes)

    model = Model(inputs=inputs, outputs=x)

    return model
