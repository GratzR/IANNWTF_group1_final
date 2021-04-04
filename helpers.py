import numpy as np
import time

def get_example_images(images, number_shown, patch_size):
  '''Returns the specified number of images'''
  example_images = []
  for i,img in enumerate(images):
    if i < number_shown:
      patch_x, patch_y = create_random_patch(img, patch_size)

      img = img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
      example_images.append(img)

  return example_images

def timing(start):
  '''Computes the time gone by since the given start time.'''
  now = time.time()
  time_gone_by = now - start

  return round(time_gone_by, 2)

def create_random_patch(image, patch_size):
  '''creates the coordinates for a random patch in the range of the image.'''
  patch_x = np.random.randint(0, image.shape[0]-patch_size)
  patch_y = np.random.randint(0, image.shape[1]-patch_size)

  return patch_x,patch_y

def get_imagePatch(image, patch_x, patch_y, patch_size, scale=1):
  '''Get the image patch at the given coordinates.'''
  return image[patch_x*scale:patch_x*scale+patch_size*scale, patch_y*scale:patch_y*scale+patch_size*scale]