import numpy as np
import tensorflow as tf 
import math 

import os 

from mysetting import * 
from utils_cm import list_dir
from functools import partial 



def get_lr_schedule(epoch, dataset=FLAGS.dataset): 
    if dataset == 'cifar10': 
        return lr_schedule_cifar10(epoch)
    elif dataset == 'mnist': 
        return lr_schedule_mnist(epoch)
    elif dataset == 'cifar100': 
        return lr_schedule_cifar100(epoch)

lr_schedule = partial(get_lr_schedule, dataset=FLAGS.dataset)

def lr_schedule_cifar10(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_schedule_cifar100(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_schedule_mnist(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 15, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 30:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def get_final_checkpoint(save_dir, selected_epoch=0): 
    """
    Args: 
        save_dir 
        selected_epoch
    Returns: 
        selected_epoch if selected_epoch <= 0 else return final_epoch in folder save_dir 
    """
    if selected_epoch > 0: 
        return os.path.join(save_dir, 'model.{:03d}.h5'.format(selected_epoch))
    else: 
        print('search checkpoints in folder: ', save_dir+'/')
        all_dir = list_dir(save_dir+'/', filetype='.h5')
        return all_dir[-1]

def get_statistic(pred): 
  def _entropy(pred):
    log_y = np.log(pred+1e-12)
    temp = -np.multiply(pred, log_y)
    return np.sum(temp, axis=1)

  en = _entropy(pred)
  # h = _hist(en)
  m = np.mean(en)
  s = np.std(en)
  return en, m, s

def get_model_eval(sess, x, y, predictions, X_test=None, Y_test=None, batch_size=None):
  """
  Compute the accuracy of a TF model on some data
  :param sess: TF session to use
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param predictions: model output predictions
  :param X_test: numpy array with training inputs
  :param Y_test: numpy array with training outputs
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :return: a float with the accuracy value and predictions 
  """
  if X_test is None or Y_test is None:
    raise ValueError("X_test argument and Y_test argument "
                     "must be supplied.")

  # Define accuracy symbolically
  correct_preds = tf.equal(tf.argmax(y, axis=-1),
                             tf.argmax(predictions, axis=-1))


  # Init result var
  accuracy = 0.0
  preds = []

  with sess.as_default():
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
    assert nb_batches * batch_size >= len(X_test)

    X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((batch_size,) + Y_test.shape[1:],
                     dtype=Y_test.dtype)
    for batch in range(nb_batches):

      # Must not use the `batch_indices` function here, because it
      # repeats some examples.
      # It's acceptable to repeat during training, but not eval.
      start = batch * batch_size
      end = min(len(X_test), start + batch_size)

      # The last batch may be smaller than all others. This should not
      # affect the accuarcy disproportionately.
      cur_batch_size = end - start
      X_cur[:cur_batch_size] = X_test[start:end]
      Y_cur[:cur_batch_size] = Y_test[start:end]
      feed_dict = {x: X_cur, y: Y_cur}

      cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)
      cur_preds = predictions.eval(feed_dict=feed_dict)

      accuracy += cur_corr_preds[:cur_batch_size].sum()
      preds.append(cur_preds)

    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)
    preds = np.concatenate(preds, axis=0)

  return accuracy, preds