import sys,os
sys.path.append(os.path.abspath("../"))
import tensorflow as tf
from settings import settings
from loss import *
import numpy
from pascal import *
import utils

from input_shuffle.input_new import *

def train_lstm_build(meta_graph_path=settings.lstm_meta_graph_path):



  #import and print all node in lstm graph
  lstm_saver = tf.train.import_meta_graph(meta_graph_path)
  lstm_graph = tf.get_default_graph()

  images_batch = lstm_graph.get_tensor_by_name('imageHolder:0')
  labels_batch = lstm_graph.get_tensor_by_name('labelHolder_1:0')

  lstm_finetune_train_ops = tf.get_collection('lstm_train_op')
  return (images_batch,labels_batch,lstm_finetune_train_ops,lstm_saver)


def main(argv=None):
  

  #read pascal as example
  #readimg = input_by_numpy.readIMage(settings.TRAIN_TXT,settings.IMAGE_DIR,settings.LABEL_DIR)
  readimg=readIMage('/scratch/eecs542w17_fluxg/ytchang/data/DAVIS/ImageSets/480p/train.txt','/scratch/eecs542w17_fluxg/ytchang/data/DAVIS')

  
  
  #build full_lstn graph
  (images_batch,labels_batch,lstm_finetune_train_ops,lstm_saver) = train_lstm_build()

  global_step = tf.contrib.framework.get_or_create_global_step()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  config.log_device_placement=True
  saver = tf.train.Saver()
  with tf.Session(config=config) as sess:
    
    lstm_saver.restore(sess, tf.train.latest_checkpoint(settings.full_lstm_dir))
    print("restore lstm from ",settings.full_lstm_dir)
   
    while True:
      print("===>train full lstm training")


      #(res_image,res_label) = readimg.read_next_natch()
      #res_image = numpy.asarray(res_image,dtype=numpy.float32)
      #feed_dict={images_batch: res_image,labels_batch:res_label}
      sequence=readimg.sample_and_shuffle()
      res_image=sequence[0][0]
      res_label=sequence[0][1]
      
      res_image = numpy.asarray(res_image,dtype=numpy.float32)
      res_label = numpy.asarray(res_label,dtype=numpy.float32)
      feed_dict={images_batch: res_image,labels_batch:res_label}
      
      _,global_step_out = sess.run(lstm_finetune_train_ops+[global_step],feed_dict=feed_dict)
      # save checkpoint every few steps
      if global_step_out%settings.full_lstm_step_per_save == 0:
        save_path = saver.save(sess, settings.full_lstm_checkpoint_path,global_step=global_step_out)
        print("Model saved in file: %s" % save_path)
      
      print("This is %d step" %global_step_out)
# run main function
if __name__  == "__main__":
  tf.app.run()
  
  