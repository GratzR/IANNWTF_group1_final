import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from blocks import ResidualBlock, UpsamplingBlock

class EDSR(Model):

  def __init__(self, numResBlocks, scale, patch_size, filters=64):
    super(EDSR, self).__init__()

    self.scale = scale

    # create first convolutional layer
    self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation=None, input_shape=(patch_size,patch_size,3))

    self.resBlocks = []
    # create numResBlocks residual blocks
    for _ in range(numResBlocks):
      self.resBlocks.append(ResidualBlock(filters=filters))

    # the convolutional layer after the residual blocks
    self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation=None)

    self.upsample = UpsamplingBlock(filters, scale)

  @tf.function
  def call(self, x):
    # goes through the first conv layer
    x = self.conv1(x)
    # we need the output of that conv layer to add it to the later self.conv2 layer
    y = x

    # goes through the residual blocks
    for block in self.resBlocks:
      x = block(x)

    # goes through the conv layer after the residual blocks
    x = self.conv2(x)
    # skip connection to add details preserve details from before the residual blocks
    x = x + y

    # upsample the image
    x = self.upsample(x)
    
    return x