from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf


def generator():
    input1 = Input((100, ))
    x = Dense(4 * 4 * 256)(input1)
    x = BatchNormalization()(x)
    x = Reshape((4, 4, 256))(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='tanh')(x)
    return Model(inputs=input1, outputs=x)


def discriminator():
    input1 = Input((64, 64, 3))

    x = Conv2D(32, 5, strides=(2, 2), padding='same') (input1)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 5, strides=(2, 2), padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, 5, strides=(2, 2), padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, 5, strides=(2, 2), padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)    

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input1, outputs=x) 


def discriminator_loss(real_output, fake_output):
    real_loss = tf.math.negative(tf.math.reduce_mean(tf.math.log(real_output)))
    fake_loss = tf.math.negative(tf.math.reduce_mean(tf.math.log(1. - fake_output)))
    return real_loss, fake_loss
    

def generator_loss(fake_output):
    return tf.math.negative(tf.math.reduce_mean(tf.math.log(fake_output)))