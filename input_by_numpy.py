import numpy as np
from settings import settings
import random
from PIL import Image
import os
import scipy.misc as misc
class readIMage():
	def __init__(self,filename_file,data_dir_path,label_dir_path):
		filenames = self._read_filenames_from_txt(filename_file)
		print('read in %d (data, labels) files' %len(filenames))
		data_filenames = [os.path.join(data_dir_path,f+'.jpg') for f in filenames]
		label_filenames = [os.path.join(label_dir_path,f+'.png') for f in filenames]

		self.image_label_queue = list(zip(data_filenames,label_filenames))

		
		self.cur_queue = self.image_label_queue[:]
		self.cur_num = 0
		if settings.shuffle:
			random.shuffle(self.cur_queue)
	  
	def read_next_natch(self,shuffle=settings.shuffle,batch=settings.BATCH_SIZE):
		# return (image_batch, label_batch)
		prev_num = self.cur_num
		self.cur_num += batch
		if self.cur_num > len(self.cur_queue):
			prev_num = 0
			self.cur_num = batch
			self.cur_queue = self.image_label_queue[:]
			if shuffle:
				random.shuffle(self.cur_queue)
		print(self.cur_queue[prev_num:self.cur_num])
		return self.input_image_label(self.cur_queue[prev_num:self.cur_num])
	def _read_filenames_from_txt(self,filename_file):
		with open(filename_file) as f:
			lines = f.read().splitlines()
		return lines

	
	def input_image_label(self,queue_part):
		# return (image_batch, label_batch)
		res_image = []
		res_label = []
		for i in range(len(queue_part)):
			image = misc.imread(queue_part[i][0])
			#mean = np.mean(image)
			#variance = np.var(image)
			#print("image type = " + str(image.dtype))
			#image.astype(float)
			#image = np.array([(x-mean)/variance for x in image])
			#image.astype(np.uint8)
			res_image.append(image)
			label = misc.imread(queue_part[i][1],mode='P')
			res_label.append(label)
			
		return (res_image,res_label)