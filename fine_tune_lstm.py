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

from input_shuffle.input_new import *



#abs_checkpoint_dir = '/home/hungwei/my_scratch/checkpoints'
abs_checkpoint_dir = settings.vgg_checkpoint_dir
abs_lstm_checkpoint_dir = settings.full_lstm_dir

from tensorflow.python.tools import inspect_checkpoint
inspect_checkpoint.print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(abs_checkpoint_dir),[],[])
vgg_saver = tf.train.import_meta_graph(os.path.join(abs_checkpoint_dir,'model.ckpt-6630.meta'))
vgg_graph = tf.get_default_graph()
vgg_names = [n.name for n in vgg_graph.as_graph_def().node]
#print(vgg_names)
#print(tf.all_variables())

#readimg = input_by_numpy.readIMage(settings.TRAIN_TXT,settings.IMAGE_DIR,settings.LABEL_DIR)
#readimg=readIMage('/scratch/eecs542w17_fluxg/ytchang/data/DAVIS/ImageSets/480p/train.txt','/scratch/eecs542w17_fluxg/ytchang/data/DAVIS')

readimg=readIMage(os.path.join(settings.my_scratch,'data/DAVIS/ImageSets/480p/train.txt'),
                  os.path.join(settings.my_scratch,'data/DAVIS'))

#images_batch = tf.placeholder(tf.float32, shape=(settings.BATCH_SIZE,480,640,3),name='imageHolder')
images_batch = vgg_graph.get_tensor_by_name('imageHolder:0')
print("Get images_batch from meta file: ",images_batch)



#images_batch = tf.reshape(images_batch, shape=(settings.BATCH_SIZE,480,640,3))
#images_batch = tf.Print(images_batch,[images_batch])
labels_batch = tf.placeholder(tf.uint8,shape=(settings.BATCH_SIZE,480,640),name='labelHolder')

state_holder=tf.placeholder(tf.uint8,shape=(settings.BATCH_SIZE,480,640,settings.NUM_CLASSES*2),name='stateHolder')

vgg_out = vgg_graph.get_tensor_by_name('content_vgg/pool5:0')
print("Get content_vgg/pool5 from meta file: ",vgg_out)
temp = set(tf.all_variables())
vgg_out = tf.reshape(vgg_out,[settings.BATCH_SIZE,480/32,640/32,512])
vgg_out = tf.Print(vgg_out,[vgg_out])
#vgg_out = vgg_graph.get_tensor_by_name('content_vgg/fc6/Relu:0')
logits = []
with tf.variable_scope("lstm") as scope:
    state_holder = tf.to_float(state_holder)
    logits, state = lstm_net.lstm_deconv_layers(vgg_out,state_holder)
    logits = tf.identity(logits, name="lstm_logits") #for renaming
    tf.add_to_collection('lstm_state',state)
    tf.add_to_collection('lstm_logits',logits)
    print("successfully open scope: lstm")

grads = []
global_step = []
train_op = []
with tf.variable_scope("fine_tune_lstm") as scope:
  logits = tf.cast(logits,tf.float32)
  loss_scalar = loss_fore_background(logits, labels_batch)
  tf.add_to_collection('loss_scalar',loss_scalar)
  #train_op = train_pal.train(loss, global_step)
  opt = tf.train.AdamOptimizer(settings.INITIAL_LEARNING_RATE)
  grads = opt.compute_gradients(loss_scalar)
  global_step = tf.contrib.framework.get_or_create_global_step()
  train_op = opt.apply_gradients(grads, global_step=global_step)
  tf.add_to_collection('lstm_train_op', train_op)
  print("successfully open scope: fine_tune_lstm")
print("train op is",train_op)
merged = tf.summary.merge_all()
temp2 = set(tf.all_variables())
lstm_init_op = tf.variables_initializer(list(temp2-temp))
#saver = tf.train.Saver(list(temp))
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
  #lstm_init_op = tf.variables_initializer(list(temp2))
  sess.run(lstm_init_op)
  vgg_saver.restore(sess, tf.train.latest_checkpoint(abs_checkpoint_dir))
  #saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
  print("===>training")


  #(res_image,res_label) = readimg.read_next_natch()
  
  
  sequence=readimg.sample_and_shuffle(video_number=settings.BATCH_SIZE,shuffle=True,batch=15)
  res_image=sequence[0][0]
  res_label=sequence[0][1]
  
  res_image = numpy.asarray(res_image,dtype=numpy.float32)
  res_label = numpy.asarray(res_label,dtype=numpy.float32)
  curstate=numpy.zeros((settings.BATCH_SIZE,480,640,settings.NUM_CLASSES*2))
  feed_dict={images_batch: res_image,labels_batch:res_label, state_holder:curstate}
  
  
  
  #res_image = [a.astype(numpy.float32) for a in res_image]
  #res_image.astype(numpy.float32)
  #res_image = numpy.asarray(res_image,dtype=numpy.float32)
  print(res_image.shape)
  print(res_image.dtype)
  #feed_dict={images_batch: res_image,labels_batch:res_label}


  state,_,global_step_out = sess.run([state,train_op,global_step],feed_dict=feed_dict)
  print("state: ")
  print(state)
  #if global_step_out%10 == 0:
  meta_graph_def = tf.train.export_meta_graph(filename=os.path.join(abs_lstm_checkpoint_dir,'lstm-model.meta'))
  save_path = saver.save(sess, os.path.join(abs_lstm_checkpoint_dir,'new_model.ckpt'),global_step=global_step_out)

  print("Model saved in file: %s" % save_path)
  
  print("This is %d step" %global_step_out)
