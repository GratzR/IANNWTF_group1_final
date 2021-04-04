import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import time
from helpers import timing, get_example_images, create_random_patch
from data_augmentation import batchAugment

def train_step(model, input, target, loss_function, optimizer,patch_size):
  '''single training step for the given batch'''

  # initialize random patch coordinates
  patch_x, patch_y = create_random_patch(input[0], patch_size)

  input, target = batchAugment(input, target)

  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
      # get the prediction from the input of the selected image patch (I hope that all batches are included)
      prediction = model(input[:, patch_x:patch_x+patch_size, patch_y:patch_y+patch_size])
      loss = loss_function(target[:, (patch_x*model.scale):patch_x*model.scale+(patch_size*model.scale), (patch_y*model.scale):patch_y*model.scale+(patch_size*model.scale)], prediction)
      gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss

def test(model, test_data, loss_function,patch_size):
  '''single testing step'''

  # test over complete test data
  test_loss_aggregator = []
  psnr_aggregator = []
  ssim_aggregator = []

  for (input, target) in test_data:      
    # initialize random patch coordinates
    patch_x, patch_y = create_random_patch(input[0], patch_size)

    # calculate the loss between input and target
    prediction = model(input[:, patch_x:patch_x+patch_size, patch_y:patch_y+patch_size, :])
    target = target[:, (patch_x*model.scale):patch_x*model.scale+(patch_size*model.scale), (patch_y*model.scale):patch_y*model.scale+(patch_size*model.scale)]
    sample_test_loss = loss_function(target, prediction)

    # calculate psnr between input and target
    psnr_sample = tf.image.psnr(target, prediction, max_val=1)
    # calculate ssim between input and target
    ssim_sample = tf.image.ssim(target, prediction, max_val=1)

    # append sample values to the respective lists
    test_loss_aggregator.append(sample_test_loss.numpy())
    psnr_aggregator.append(psnr_sample.numpy())
    ssim_aggregator.append(ssim_sample.numpy())

    # calculate the mean over the respective lists

    test_loss = np.mean(test_loss_aggregator)
    psnr = np.mean(psnr_aggregator)
    ssim = np.mean(ssim_aggregator)

    return test_loss, psnr, ssim

def train(model, train_dataset, test_dataset, epochs ,patch_size ,tensorboard_images, number_test_images=3):
  '''Training the model with the given datasets'''

  # Define filewriters for Tensorboard tracking.
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_loss_log_dir = 'logs/gradient_tape/' + current_time + '/train_loss'
  test_loss_log_dir = 'logs/gradient_tape/' + current_time + '/test_loss'
  psnr_log_dir = 'logs/gradient_tape/' + current_time + '/psnr'
  ssim_log_dir = 'logs/gradient_tape/' + current_time + '/ssim'
  img_log_dir = 'logs/gradient_tape/' + current_time + '/img'

  train_loss_summary_writer = tf.summary.create_file_writer(train_loss_log_dir)
  test_loss_summary_writer = tf.summary.create_file_writer(test_loss_log_dir)
  psnr_summary_writer = tf.summary.create_file_writer(psnr_log_dir)
  ssim_summary_writer = tf.summary.create_file_writer(ssim_log_dir)
  img_summary_writer = tf.summary.create_file_writer(img_log_dir)
  
  tf.keras.backend.clear_session()

  #get example images from dataset for tensorboard process tracking
  example_images = get_example_images(tensorboard_images,number_test_images,patch_size)

  # hyperparameters
  num_epochs = epochs
  learning_rate = 1e-4
  running_average_factor = 0.95

  # initialize L1 loss -> mean absolute error (we take the mean of the absolute difference between label and prediction)
  loss = tf.keras.losses.MeanAbsoluteError()

  # learning rate will be halved 4 times in the process
  optimizer = tf.keras.optimizers.Adam(learning_rate=PiecewiseConstantDecay(boundaries=[num_epochs//5, (num_epochs//5)*2, (num_epochs//5)*3, (num_epochs//5)*4], 
                                                                            values=[learning_rate, (learning_rate)/2, (learning_rate)/(2*2), (learning_rate)/(2*4), (learning_rate)/(2*8)]),
                                       beta_1=0.9, 
                                       beta_2=0.999)
  
  train_losses = []
  test_losses = []
  psnr_list = []
  ssim_list = []

  # test once before we start 
  train_loss,_,_ = test(model, train_dataset, loss, patch_size)
  train_losses.append(train_loss)

  test_loss, psnr, ssim = test(model, test_dataset, loss, patch_size)
  test_losses.append(test_loss)
  psnr_list.append(psnr)
  ssim_list.append(ssim)

  # initialize time
  training_time = time.time()

  # train for num_epochs
  for epoch in range(num_epochs):
    epoch_time = time.time()

    running_average = 0
    for i,(input, target) in enumerate(train_dataset):
      train_loss = train_step(model, input, target, loss, optimizer, patch_size)
      running_average = running_average_factor * running_average  + (1 - running_average_factor) * train_loss     
    
    # append train losses
    train_losses.append(running_average)

    # testing
    test_loss, psnr, ssim = test(model, test_dataset, loss, patch_size)
    # append test losses
    test_losses.append(test_loss)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

    with train_loss_summary_writer.as_default():
      tf.summary.scalar('loss', train_losses[-1], step=epoch)

    with test_loss_summary_writer.as_default():
      tf.summary.scalar('loss', test_losses[-1], step=epoch)

    with test_loss_summary_writer.as_default():
      tf.summary.scalar('psnr', psnr_list[-1], step=epoch)

    with test_loss_summary_writer.as_default():
      tf.summary.scalar('ssim', ssim_list[-1], step=epoch)

    # load produced images into Tensorboard
    pred_images = []

    for image in example_images:
      pred_images.append(model(tf.expand_dims(image, axis=0)))

    with img_summary_writer.as_default():
      tf.summary.image("upscaled images", tf.squeeze(pred_images), step=epoch, max_outputs=number_test_images)

    print(f'Epoch {str(epoch)}: training loss = {running_average}, test loss = {test_loss}, psnr = {psnr}, ssim = {ssim}, time {timing(epoch_time)} seconds')