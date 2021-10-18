'''Image Transform Net and Loss Network models for Tensorflow.

Reference:
  - Justin Johnson, Alexandre Alahi and Li Fei-Fei. 
    [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](
      https://arxiv.org/abs/1603.08155) (ECCV 2016)

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Oct 2020
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16
import tensorflow_addons as tfa


class ConvReflect(tf.keras.layers.Layer):
    # 2D Convolution layer with reflection padding
    def __init__(self, filters, kernel_size, strides=(1, 1), 
                 activation=None,
                 kernel_initializer='glorot_normal'):
        super(ConvReflect, self).__init__()
        self.size_pad = kernel_size // 2
        self.padding = tf.constant([[0, 0], 
                                    [self.size_pad, self.size_pad], 
                                    [self.size_pad, self.size_pad], 
                                    [0, 0]])
        self.conv2d = layers.Conv2D(filters, kernel_size, strides,
                                    activation=activation,
                                    kernel_initializer=kernel_initializer)

    def call(self, x):
        x = tf.pad(x, self.padding, 'REFLECT') 
        return self.conv2d(x)


def ImageTransformNet(input_shape=(256, 256, 3), residual_layers=5, 
                      residual_filters=128, initializer='glorot_normal'):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(inputs)

    x = ConvReflect(32, 9, kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer=initializer,
                                         gamma_initializer=initializer)(x)
    x = layers.Activation('relu')(x)
    
    x = ConvReflect(64, 3, strides=2, kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer=initializer,
                                         gamma_initializer=initializer)(x)
    x = layers.Activation('relu')(x)
    
    x = ConvReflect(residual_filters, 3, strides=2, 
                    kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer=initializer,
                                         gamma_initializer=initializer)(x)
    x = layers.Activation('relu')(x)

    for size in [residual_filters]*residual_layers:
        residual = x
        x = ConvReflect(size, 3, kernel_initializer=initializer)(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                             scale=True,
                                             beta_initializer=initializer,
                                             gamma_initializer=initializer)(x)

        x = layers.Activation('relu')(x)

        x = ConvReflect(size, 3, kernel_initializer=initializer)(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                             scale=True,
                                             beta_initializer=initializer,
                                             gamma_initializer=initializer)(x)
        x = layers.add([x, residual])  # Add back residual

    x = layers.UpSampling2D(2)(x)
    x = ConvReflect(64, 3, kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer=initializer,
                                         gamma_initializer=initializer)(x)
    x = layers.Activation('relu')(x)   

    x = layers.UpSampling2D(2)(x)
    x = ConvReflect(32, 3, kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                         scale=True,
                                         beta_initializer=initializer,
                                         gamma_initializer=initializer)(x)
    x = layers.Activation('relu')(x)
    
    x = ConvReflect(3, 9, kernel_initializer=initializer)(x)
    outputs = layers.Activation('tanh', dtype='float32')(x)

    return tf.keras.Model(inputs, outputs)


class LossNetwork(tf.keras.models.Model):
    def __init__(self, style_layers = ['block1_conv2',
                                       'block2_conv2',
                                       'block3_conv3', 
                                       'block4_conv3']):
        super(LossNetwork, self).__init__()
        vgg = vgg16.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        model_outputs = [vgg.get_layer(name).output for name in style_layers]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
        # mixed precision float32 output
        self.linear = layers.Activation('linear', dtype='float32') 

    def call(self, x):
        x = vgg16.preprocess_input(x)
        x = self.model(x)
        return self.linear(x)    
