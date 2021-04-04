import tensorflow as tf

def batchAugment(train_batch, label_batch):
  '''Augments a batch of trainingdata'''
  input_aug = []
  label_aug = []

  for i, single_input in enumerate(train_batch):
    augmented_images = singleAugment(single_input,label_batch[i])
    input_aug.append(augmented_images[0])
    label_aug.append(augmented_images[1])

  return tf.stack(input_aug), tf.stack(label_aug)

def singleAugment(train_img, label_img):
  '''Augments a single pair of image and label.
  Image can be flippen and/or rotated right or left by 90 degrees.'''
  flip_prob=tf.random.uniform(shape=[1])

  if flip_prob > 0.5:
    train_img = tf.image.flip_left_right(train_img)
    label_img = tf.image.flip_left_right(label_img)

  rota_prob=tf.random.uniform(shape=[1])
  if rota_prob > 0.5 and rota_prob < 0.75:
    train_img = tf.image.rot90(train_img)
    label_img = tf.image.rot90(label_img)

  elif rota_prob > 0.5:
    train_img = tf.image.rot90(train_img, k=3)
    label_img = tf.image.rot90(label_img, k=3)

  return train_img,label_img