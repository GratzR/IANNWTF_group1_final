import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class ResidualBlock(Model):

  def __init__(self, filters):
    super(ResidualBlock, self).__init__()
    
    self.conv1 = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation=None)
    self.relu = tf.keras.activations.relu
    self.conv2 = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation=None)

  @tf.function
  def call(self, x):
    y = x

    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)

    return x + y

class UpsamplingBlock(Model):

  def __init__(self, filters, scale=2):
    super(UpsamplingBlock, self).__init__()

    self.scale = scale

    # the upsampling layer(s)
    # for scale x2
    self.upsample_scaleX2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', activation=None)
    # for scale x3
    self.upsample_scaleX3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=3, padding='same', activation=None)
    # for scale x4
    self.upsample_scaleX4_1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', activation=None)
    self.upsample_scaleX4_2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', activation=None)

    # the last convolutional layer
    self.conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', activation=None)
    self.sigmoid = tf.keras.activations.sigmoid

  @tf.function
  def call(self, x):
    if self.scale == 2:
      x = self.upsample_scaleX2(x)
    elif self.scale == 3:
      x = self.upsample_scaleX3(x)
    elif self.scale == 4:
      x = self.upsample_scaleX4_1(x)
      x = self.upsample_scaleX4_2(x)

    x = self.conv(x)
    x = self.sigmoid(x)

    return x