import tensorflow as tf
from PIL import Image
from settings import settings
import time
from datetime import datetime
from loss import *
import numpy
from pascal import *
import lstm_net
import nets.fcn32_vgg as fcn32_vgg
import utils
import os


vgg_saver = tf.train.import_meta_graph(os.path.join('checkpoints/','model.ckpt-6630.meta'))
vgg_graph = tf.get_default_graph()
vgg_names = [n.name for n in vgg_graph.as_graph_def().node]
#print(vgg_names)
#print(tf.all_variables())
readimg = input_by_numpy.readIMage(settings.TRAIN_TXT,
  settings.IMAGE_DIR,
  settings.LABEL_DIR)
images_batch = tf.placeholder(tf.float32, shape=(settings.BATCH_SIZE,480,640,3),name='imageHolder')
labels_batch = tf.placeholder(tf.uint8,shape=(settings.BATCH_SIZE,480,640),name='labelHolder')
vgg_out = vgg_graph.get_tensor_by_name('content_vgg/pool5:0')
temp = set(tf.all_variables())
vgg_out = tf.reshape(vgg_out,[settings.BATCH_SIZE,480/32,640/32,512])
#vgg_out = vgg_graph.get_tensor_by_name('content_vgg/fc6/Relu:0')
logits = []
with tf.variable_scope("lstm") as scope:
    logits = lstm_net.lstm_deconv_layers(vgg_out)
grads = []
global_step = []
train_op = []
with tf.variable_scope("fine_tune_lstm") as scope:
    logits = tf.cast(logits,tf.float32)
    loss_scalar = loss_proj2(logits, labels_batch)
    #train_op = train_pal.train(loss, global_step)
    opt = tf.train.AdamOptimizer(settings.INITIAL_LEARNING_RATE)
    grads = opt.compute_gradients(loss_scalar)
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = opt.apply_gradients(grads, global_step=global_step)
merged = tf.summary.merge_all()
temp2 = set(tf.all_variables())
lstm_init_op = tf.variables_initializer(list(temp2))
#saver = tf.train.Saver(list(temp))

with tf.Session() as sess:
  #lstm_init_op = tf.variables_initializer(list(temp2))
  sess.run(lstm_init_op)
  vgg_saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
  #saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
  while True:
    print("===>training")


    (res_image,res_label) = readimg.read_next_natch()
    res_image.astype(numpy.float32)
    feed_dict={images_batch: res_image,labels_batch:res_label}


    summary,_,global_step_out = sess.run([merged,train_op,global_step],feed_dict=feed_dict)
    if global_step_out%50 == 0:
        save_path = saver.save(sess, "/checkpoints/new_model.ckpt")
        print("Model saved in file: %s" % save_path)
    
    print("This is %d step" %global_step_out)
