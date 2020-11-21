import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from hparams import hparams

def ImageTransformNet(input_shape=(256, 256, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    hparams['residual_filters']

    x = layers.ZeroPadding2D(padding=2)(inputs)
    x = layers.Conv2D(32, 9, strides=1, padding='same',
                        kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer="he_normal",
                                         gamma_initializer="he_normal")(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(64, 3, strides=2,
                      kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer="he_normal",
                                         gamma_initializer="he_normal")(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(hparams['residual_filters'], 3, strides=2,
                      kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer="he_normal",
                                         gamma_initializer="he_normal")(x)
    x = layers.Activation("relu")(x)

    for size in [hparams['residual_filters']]*hparams['residual_layers']:
        residual = x
        x = layers.Conv2D(size, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                             scale=True,
                                             beta_initializer="he_normal",
                                             gamma_initializer="he_normal")(x)

        x = layers.Activation("relu")(x)
        x = x = layers.Conv2D(size, 3, padding='same',
                              kernel_initializer='he_normal')(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                             scale=True,
                                             beta_initializer="he_normal",
                                             gamma_initializer="he_normal")(x)
        x = layers.add([x, residual])  # Add back residual

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, strides=1, padding='same',
                             kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer="he_normal",
                                         gamma_initializer="he_normal")(x)
    x = layers.Activation("relu")(x)   

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, strides=1, padding='same',
                      kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer="he_normal",
                                         gamma_initializer="he_normal")(x)
    x = layers.Activation("relu")(x)
    
    outputs = layers.Conv2D(3, 9, strides=1, padding='same', dtype='float32',
                            kernel_initializer='he_normal')(x)

    return tf.keras.Model(inputs, outputs)


def LossNetwork():
    content_layers = ['block1_conv2',
                      'block2_conv2',
                      'block3_conv3', 
                      'block4_conv3'
    ]
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = content_outputs
    return tf.keras.models.Model(vgg.input, model_outputs)
