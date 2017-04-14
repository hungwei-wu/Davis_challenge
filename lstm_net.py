import tensorflow as tf
#from lstm_ops import basic_conv_lstm_cell
from deconv_lstm import deconv_lstm_cell
from conv_lstm import conv_lstm_cell
from settings import settings
import numpy as np
def lstm_deconv_layers(images_batch,deconvlstm_state2):
  """
  input: batch*h*w*channels
  """
  #lstm conv and deconv
  #deconvlstm_state1,deconvlstm_state2=None,None
  #convlstm_state1,convlstm_state2=None,None
  #lstm_deconv1,deconvlstm_state1=basic_conv_lstm_cell(images_batch, deconvlstm_state1, filter_size=5, num_channels=3, scope='state1')
  #lstm_conv1,convlstm_state1=conv_lstm_cell(images_batch,convlstm_state1,filter_size=3,num_channels=24, scope='convstate1')
  #lstm_conv2,convlstm_state2=conv_lstm_cell(lstm_conv1,convlstm_state2,filter_size=3,num_channels=48, scope='convstate2')
  #lstm_deconv1,deconvlstm_state1=deconv_lstm_cell(lstm_conv2, deconvlstm_state1, upscale=16, num_channels=48, scope='deconvstate1')
  lstm_deconv2,deconvlstm_state2=deconv_lstm_cell(images_batch, deconvlstm_state2, upscale=32, num_channels=settings.NUM_CLASSES, scope='deconvstate2')
  
  '''
  with tf.Session() as sess:
    res_image=np.random.rand(10,320,640,3)
    feed_dict={images_batch: res_image}
    sess.run(tf.global_variables_initializer())
    result=sess.run(lstm_deconv2,feed_dict=feed_dict)
    print(result.shape)
  '''
  return lstm_deconv2,deconvlstm_state2

#images_batch = tf.placeholder(tf.float32, shape=(10,320,640,3),name='imageHolder')

#lstm_deconv_layers(images_batch)