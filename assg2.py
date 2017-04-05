from input_assg2 import *
from PIL import Image
from settings import settings
#import train_pal
import time
from datetime import datetime
#import loss_pal
from loss import *
import numpy
from input_by_numpy import *

import fcn32_vgg
import utils
def conv_transpose(tensor, out_channel, shape, strides, name=None):
  out_shape = tensor.get_shape().as_list()
  in_channel = out_shape[-1]
  kernel = weight_variable([shape, shape, out_channel, in_channel], name=name)
  shape[-1] = out_channel
  return tf.nn.conv2d_transpose(x, kernel, output_shape=out_shape, strides=[1, strides, strides, 1],
                                padding='SAME', name='conv_transpose')
def actual_train(images_batch,labels_batch):
	
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    
    print("===> building graphs")
    # Build a Graph that computes the logits predictions from the
    # inference model.
    vgg_fcn = fcn32_vgg.FCN32VGG(vgg16_npy_path='./tmp/vgg16.npy')
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(images_batch, debug=True)
    #logits = train_pal.inference(images_batch,isTrain = True)
    

    ############### output of VGG net
    logits = vgg_fcn.upscore
    #logits = vgg_fcn.conv5
    ###############
    #loss = loss_pal.loss(logits,labels_batch)
    logits = tf.cast(logits,tf.float32)
    loss_scalar = loss_proj2(logits, labels_batch)
    #train_op = train_pal.train(loss, global_step)
    opt = tf.train.AdamOptimizer(settings.INITIAL_LEARNING_RATE)
    grads = opt.compute_gradients(loss_scalar)

    # Apply gradients.
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    merged = tf.summary.merge_all()



    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss_scalar)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 1 == 0:
          num_examples_per_step = settings.BATCH_SIZE
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir='checkpoints/',
        hooks=[tf.train.StopAtStepHook(last_step=settings.MAX_STEPS),
               tf.train.NanTensorHook(loss_scalar),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=settings.log_device_placement),
	save_checkpoint_secs=60) as sess:
      
      train_writer = tf.summary.FileWriter(settings.summaries_dir + '/train',
                                      sess.graph)
      while not sess.should_stop():
        print("===>training")


        (res_image,res_label) = readimg.read_next_natch()
        feed_dict={images_batch: res_image,label_batch:res_label}
        summary,_,global_step_out = sess.run([merged,train_op,global_step],feed_dict=feed_dict)
        train_writer.add_summary(summary, global_step_out)
        print("This is %d step" %global_step_out)
        #steps += 1
    '''
    with tf.Session() as sess:
      
      sess.run(tf.global_variables_initializer())

      my_label = tf.get_collection('label')
      #print(sess.run(my_label))
      while steps < settings.MAX_STEPS:
        (res_image,res_label) = readimg.read_next_natch()
        feed_dict={images_batch: res_image,label_batch:res_label}
        print(sess.run([train_op],feed_dict=feed_dict))
        print("This is %d step" %steps)
        steps += 1
    '''
BATCH_SIZE = settings.BATCH_SIZE

#DATA_DIR = './data/TrainVal/VOCdevkit/VOC2011/JPEGImages'
#LABEL_DIR = './data/TrainVal/VOCdevkit/VOC2011/SegmentationClass'
#filenames = read_filenames_from_txt('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')
numpy.set_printoptions(threshold=numpy.nan)
#readimg = readIMage('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt',
readimg = readIMage(settings.TRAIN_TXT,
  settings.IMAGE_DIR,
  settings.LABEL_DIR)


images_batch = tf.placeholder(tf.float32, shape=(settings.BATCH_SIZE,None,None,3),name='imageHolder')
label_batch = tf.placeholder(tf.uint8,shape=(settings.BATCH_SIZE,None,None),name='labelHolder')
actual_train(images_batch,label_batch)

