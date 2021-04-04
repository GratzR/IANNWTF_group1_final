import imageio
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_handling import dataset_to_array
from geometric_self_ensamble import geom_self_ensamble
from helpers import create_random_patch, get_imagePatch

def print_images(dataset, num_images,patch_size,scale=2):
  '''print x random images from the given dataset.
  First the whole image followed by the low- and high resolution patch'''
  pictures = [(image, label) for (image,label) in dataset.take(num_images)]
  patch_size_scaled = patch_size * scale

  fig, ax = plt.subplots(num_images, 3, figsize=(15, 10))
  for i, (image, label) in enumerate(pictures):
    patch_x, patch_y = create_random_patch(image, patch_size)
    patch_x_scaled = patch_x*scale
    patch_y_scaled = patch_y*scale

    # initialize a rectangle to indicate where the patch is taken from the original image.
    rect = patches.Rectangle((patch_y_scaled, patch_x_scaled), patch_size_scaled, patch_size_scaled, linewidth=2,edgecolor='yellow',facecolor='none')
      
    # get the images
    image_patch = get_imagePatch(image, patch_x, patch_y, patch_size)
    label_patch = get_imagePatch(label, patch_x, patch_y, patch_size,scale)

    # plot the images
    ax[i,0].imshow(label)
    ax[i,0].set_title(f"Original image \n Shape = {label.shape}")
    ax[i,0].axis('off')
    ax[i,0].add_patch(rect)
    ax[i,1].imshow(image_patch)
    ax[i,1].set_title(f"Low resolution image \n Shape = {image_patch.shape}")
    ax[i,1].axis('off')
    ax[i,2].imshow(label_patch)
    ax[i,2].set_title(f"High resolution image \n Shape = {label_patch.shape}")
    ax[i,2].axis('off')

    plt.tight_layout()

def plot_edsr_images(model, dataset, patch_size, num_images=6, save_images=False, geom=False):
  '''Plots x images from the given dataset upscaled by the passed model.
  geom=True can be used to add Geomtric Self-ensamble.
  The image is first shown in low resolution followed by the model prediction and the original.'''
  test_images,test_labels,raw_images = dataset_to_array(dataset)

  patch_size_scaled = patch_size * model.scale

  fig, ax = plt.subplots(num_images,3, figsize=(12,25))
  for i in range(num_images):
    patch_x, patch_y = create_random_patch(test_images[i], patch_size)

    # get the images
    if geom:
      sr_image = geom_self_ensamble(model, dataset, test_images[i], scale=3,patch_size=patch_size, init_patch=(patch_x, patch_y))
    else:
      sr_image = model(tf.reshape(get_imagePatch(test_images[i], patch_x, patch_y, patch_size=patch_size), (-1, patch_size, patch_size, 3)))
    label = get_imagePatch(test_labels[i], patch_x, patch_y, patch_size=patch_size, scale=model.scale)
    low_image = get_imagePatch(raw_images[i], patch_x, patch_y, patch_size=patch_size)

    # save the produced images (and/or corresponding labels)to the hard drive 
    # (and replace the directory path ofc) / colab files
    if save_images:
      imageio.imwrite(f'LowRes{i}.png', (np.asarray(low_image).astype(np.uint8)))
      imageio.imwrite(f'SRImage{i}.png', (np.reshape(sr_image, (patch_size_scaled,patch_size_scaled,3))*255).astype(np.uint8))
      imageio.imwrite(f'Label{i}.png', (np.asarray(label*255).astype(np.uint8)))

    # plot the images
    ax[i,0].imshow(low_image)
    ax[i,0].axis("off")
    ax[i,0].set_title("Low Resolution Image")
    ax[i,1].imshow(tf.reshape(sr_image, (patch_size_scaled, patch_size_scaled,3)))
    ax[i,1].axis("off")
    ax[i,1].set_title("Super Resolution Image "+ ("+ " if geom else "") + f"\n PSNR = {tf.image.psnr(sr_image, label, max_val=1)}")
    ax[i,2].imshow(label)
    ax[i,2].axis("off")
    ax[i,2].set_title("Label")
  plt.tight_layout()
  plt.show()

def print_pictureComparison(model, dataset, num_images, patch_size):
  '''Compares the results of the model to the original picture.
  Prints the whole picture and the patch location followed by the high- and low 
  resolution version and the model predictions with- and withoug Geometric Self-ensamble.'''
  test_images,test_labels,raw_images = dataset_to_array(dataset)

  patch_size_scaled = patch_size * model.scale
  
  for i in range(num_images):
    patch_x, _ = create_random_patch(test_images[i], patch_size)
    patch_y = np.random.randint(10, 100)
    patch_x_scaled = patch_x*model.scale
    patch_y_scaled = patch_y*model.scale

    # get the images
    edsr_output = model(tf.reshape(get_imagePatch(test_images[i], patch_x, patch_y, patch_size=patch_size), (-1, patch_size, patch_size, 3)))
    ensamble_img = geom_self_ensamble(model,dataset, test_images[i], scale=model.scale,patch_size=patch_size, init_patch=(patch_x, patch_y))
    label = get_imagePatch(test_labels[i], patch_x, patch_y,patch_size=patch_size, scale=model.scale)
    low_image = get_imagePatch(raw_images[i], patch_x, patch_y, patch_size=patch_size)

    # initialize a rectangle 
    rect = patches.Rectangle((patch_y_scaled, patch_x_scaled), patch_size_scaled, patch_size_scaled, linewidth=2,edgecolor='yellow',facecolor='none')

    fig = plt.figure(tight_layout=True, figsize=(9,6))
    gs = gridspec.GridSpec(2, 3)

    # plot the images
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(test_labels[i][:, 0:600])
    ax.add_patch(rect)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(label)
    ax.set_title("HR")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(low_image)
    ax.set_title("Bicubic")
    ax.axis("off")      

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(tf.reshape(edsr_output, (patch_size_scaled, patch_size_scaled,3)))
    ax.set_title(f"EDSR, PSNR = {tf.image.psnr(edsr_output, label, max_val=1)}")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(tf.reshape(ensamble_img, (patch_size_scaled,patch_size_scaled,3)))
    ax.set_title(f"EDSR +, PSNR = {tf.image.psnr(ensamble_img, label, max_val=1)}")
    ax.axis("off")

    fig.align_labels()
    plt.show()