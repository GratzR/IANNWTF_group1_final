import tensorflow as tf

def dataset_to_array(dataset):
  '''Transforms dataset to array for improved accesability'''
  x = []
  y = []
  z = []
  for img,label in dataset:
      z.append(img)
      x.append(tf.image.per_image_standardization(tf.cast(img, dtype=tf.float32)))
      y.append(label/255)
  return x, y, z

# create input_pipeline function to preprocess the different datasets 
def input_pipeline(train, test, scale):
  '''Input pipeline to preproccess the dataset'''
  # train dataset
  # we just want to preprocess the interpolated images since we want to train only on them
  # cast to float
  train_dataset_images = train.map(lambda x, y: (tf.cast(x, dtype=tf.float32),y))
  # normalize the images
  train_dataset_images = train_dataset_images.map(lambda x, y: (tf.image.per_image_standardization(x), y/255))
  # resize the images
  train_dataset_images = train_dataset_images.map(lambda x, y: (tf.image.resize(x, (600//scale, 600//scale)), tf.image.resize(y, (600, 600))))
  # batch and prefetch
  train_dataset_images = train_dataset_images.batch(4)
  train_dataset_images = train_dataset_images.prefetch(2)

  # test dataset 
  # cast to float
  test_dataset_images = test.map(lambda x, y: (tf.cast(x, dtype=tf.float32),y))
  # normalize the images
  test_dataset_images = test_dataset_images.map(lambda x, y: (tf.image.per_image_standardization(x), y/255))
  # resize the images
  test_dataset_images = test_dataset_images.map(lambda x, y: (tf.image.resize(x, (600//scale, 600//scale)), tf.image.resize(y, (600, 600))))
  # batch and prefetch
  test_dataset_images = test_dataset_images.batch(4)
  test_dataset_images = test_dataset_images.prefetch(2)

  return train_dataset_images, test_dataset_images