from input_n import *
import numpy as np
#from PIL import Image
import scipy.misc as misc
#readimg = readIMage('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt',
'''readimg = readIMage('./train_mod.txt',
	'./DAVIS-data/DAVIS/JPEGImages/480p',
	'./DAVIS-data/DAVIS/Annotations/480p')'''
readimg = readIMage('./data/DAVIS/ImageSets/480p/train.txt',
	'./data/DAVIS')
#(res_image,res_label) = readimg.read_next_natch()

#print(out[0])
#print(out[1])
#print(out[0][0])
#print(out[1][0])
np.set_printoptions(threshold=np.nan)

for j in range(100):
	(res_image,res_label) = readimg.read_next_natch(batch=1)
	for i in range(len(res_image)):
		#print(np.array(out[0][i]))
		img = Image.fromarray(np.array(res_image[i]), 'RGB')
		#print(res_image[i][0])
		#misc.imshow(res_image[i])
		#misc.imshow(res_label[i])
		img.show()
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
	print(j)
