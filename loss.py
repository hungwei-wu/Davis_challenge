"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        logits = logits
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
def loss_proj2(logits, labels):
  """
  From Wei
  Args:
    logits: Logits from inference and upsampled(), with settings.BATCH_SIZE, settings.NUM_CLASSES 
      , and same size to the labelled image. ([batch, height, width, classes])
    labels: Batched labels from image labels, same size with the images. 
  Returns:
    Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  comparison = tf.equal( labels, tf.constant(255,dtype=tf.int64) )
  labels_new = labels
  #labels_new = labels.assign( tf.where(comparison, tf.zeros_like(labels), labels) )
  #labels_new = tf.assign( labels,tf.where(comparison, tf.zeros_like(labels), labels) )
  labels_new = tf.where(comparison, tf.zeros_like(labels), labels)
  #Calculate the average cross entropy loss across the batch.
  #labels_new = tf.cast(labels_new, tf.int64)
  #labels_new = tf.to_int64(labels_new)
  #labels_new = tf.cast(, tf.int64)
  print(logits)
  print(labels)
  print(labels_new)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_new, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  
  #The total loss is defined as the cross entropy loss plus all of the weight
  #decay terms (L2 loss).
  #return tf.add_n([cross_entropy_mean], name='total_loss') 
  return cross_entropy_mean
