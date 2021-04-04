import tensorflow as tf
import numpy as np
from helpers import get_imagePatch, create_random_patch

def geom_self_ensamble(model, dataset, img, scale, patch_size, init_patch=False , norm_image=False):
  '''Creates 8 augmented versions of the input, feeds them to the model, 
  recreates the original orientations from the results and computes the average of these.'''
  
  reshape_shape=(-1, patch_size, patch_size, 3)

  if init_patch:
    patch_x, patch_y = init_patch
  else:
    patch_x, patch_y = create_random_patch(img, patch_size)

  img = get_imagePatch(img, patch_x, patch_y, patch_size=patch_size)

  if norm_image:
    # normalize the image
    img = tf.cast(image, tf.float32)
    img = tf.image.per_image_standardization(img)
  
  # construct the 8 images
  # original
  prediction = model(tf.reshape(img, reshape_shape))

  # flip image horizontally
  flip_hor = tf.image.flip_left_right(img)
  prediction_hor = model(tf.reshape(flip_hor, reshape_shape))

  # flip image vertically 
  flip_vert = tf.image.flip_up_down(img) 
  prediction_vert = model(tf.reshape(flip_vert, reshape_shape))

  # rotate image 90 deg
  rot_90 = tf.image.rot90(img, k=3)
  prediction_rot_90 = model(tf.reshape(rot_90, reshape_shape))

  # rotate image -90 deg
  rot_minus_90 = tf.image.rot90(img, k=1)
  prediction_rot_minus_90 = model(tf.reshape(rot_minus_90, reshape_shape))

  # rotate image 180 deg
  rot_180 = tf.image.rot90(img, k=2)
  prediction_rot_180 = model(tf.reshape(rot_180, reshape_shape))

  # flip horizontally and rotate -90 deg
  flip_hor_minus_90 = tf.image.flip_left_right(rot_minus_90)
  prediction_hor_minus_90 = model(tf.reshape(flip_hor_minus_90, reshape_shape))

  # flip vertically and rotate -90 deg
  flip_vert_minus_90 = tf.image.flip_up_down(rot_minus_90)
  prediction_vert_minus_90 = model(tf.reshape(flip_vert_minus_90, reshape_shape))


  # inverse transformations
  one = tf.image.flip_left_right(prediction_hor)
  two = tf.image.flip_up_down(prediction_vert)
  three = tf.image.rot90(prediction_rot_90, k=1)
  four = tf.image.rot90(prediction_rot_minus_90, k=3)
  five = tf.image.rot90(prediction_rot_180, k=2)
  six = tf.image.flip_left_right(tf.image.rot90(prediction_hor_minus_90, k=1))
  seven = tf.image.flip_up_down(tf.image.rot90(prediction_vert_minus_90, k=1))

  # store the elements in a tensor
  elements = [prediction, one, two, three, four, five, six, seven]

  # calculate the mean
  prediction_mean = tf.math.reduce_mean(elements, axis=0)

  return prediction_mean