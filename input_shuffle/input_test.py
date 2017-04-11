from input_new import *
import numpy as np
from PIL import Image
import scipy.misc as misc
readimg = readIMage('/scratch/eecs542w17_fluxg/ytchang/data/DAVIS/ImageSets/480p/train.txt',
	'/scratch/eecs542w17_fluxg/ytchang/data/DAVIS')
'''readimg = readIMage('./test.txt',
	'./data/DAVIS')'''

#np.set_printoptions(threshold=np.nan)

for j in range(1):
  print('\n\n     (cycle) j = %d' %j+'\n==================\n')
  #(res_image,res_label) = readimg.sample_and_shuffle(video_number=6,shuffle=True,batch=10)
  #print(len(res_image))
  #sequence=readimg.sample_and_shuffle(video_number=6,shuffle=True,batch=5)
  sequence=readimg.read_whole_video(0)
  print("sequence length:{}".format(len(sequence)))
  print("batch size: {}".format(len(sequence[0][0])))
  for batch in sequence:
    img_batch=batch[0]
    label_batch=batch[1]
    sampleImg=Image.fromarray(np.array(img_batch[0]), 'RGB')
    samplelabel=Image.fromarray(np.array(label_batch[0]), 'P')
    #sampleImg.show()
    #samplelabel.show()
    
    #print("======frame begin, batch size={}======".format(len(batch[0])))
    for i in range(len(img_batch)):
      img=img_batch[i]
      label=label_batch[i]
      #print("image shape: {}".format(img.shape))
      #print("label shape: {}".format(label.shape))
    #print("========frame end=========")
  
  #for i in range(len(res_image)):
		#print(np.array(out[0][i]))
    #img = Image.fromarray(np.array(res_image[i]), 'RGB')
    #print(res_image[i].shape)
    
		#misc.imshow(res_image[i])
		#misc.imshow(res_label[i])
		#img.show()
		#print(out[1][0])
		#print(np.array(out[1][i]))
		#res_label[i] = np.array(res_label[i])
		#res_label[i] = res_label[i].astype('uint8')
		#print(res_label[i])
		#print(out[1][i].shape())
		#print(res_label[i])
		#print(i)
		#print(res_label[i].shape())
		#img = Image.fromarray(res_label[i]*100,'P')
		#.convert('RGB')
		#img.show()
	
