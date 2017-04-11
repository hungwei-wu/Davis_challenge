import numpy as np
import sys
sys.path.append('../')
from settings import settings
import random
from PIL import Image
import os
import scipy.misc as misc
class readIMage():
	def __init__(self,filename_file,dir_path):
		filenames = self._read_filenames_from_txt(filename_file)
		print('read in %d (data, labels) files\n\n' %len(filenames))
		data_filenames = [os.path.join(dir_path,'.'+f.split()[0]) for f in filenames]
		label_filenames = [os.path.join(dir_path,'.'+f.split()[1]) for f in filenames]
		
		#print(data_filenames)
		#print(label_filenames)
		
		video_category=[g.split('/')[7] for g in data_filenames]
    
		#print('video_category:\n'+' '.join(video_category)+'\n==================\n')
		
		self.video_category_number=[]
		self.video_category_number.append(0)		
		for j in range(len(data_filenames)-1):	
			if video_category[j+1]!=video_category[j]:
				self.video_category_number.append(j)
		self.video_category_number.append(len(data_filenames)-1)
		#print('video_category_number:\n'+' '.join(map(str, self.video_category_number))+'\n==================\n')
		print('Total number of videos is: %d' %(len(self.video_category_number)-1)+'\n==================\n')
		
		#self.image_label_queue = list(zip(data_filenames,label_filenames))
		self.image_label_queue = zip(data_filenames,label_filenames)		
		#self.cur_queue = self.image_label_queue[:]
		
	def sample_and_shuffle(self,video_number,shuffle=settings.shuffle,batch=settings.BATCH_SIZE):	
		assert (video_number <= (len(self.video_category_number)-1)),'\n[warning]''video_number exceeds Total number of videos\n'
		
		#self.sample=np.zeros(len(self.video_category_number)-1)
		self.sample=[0]*(len(self.video_category_number)-1)
		self.sample_video=[]*(batch*(len(self.video_category_number)-1))
		
		for k in range(len(self.video_category_number)-1):	
			self.sample[k]=random.randint(self.video_category_number[k],(self.video_category_number[k+1]-batch))	
		#print('sample:\n'+' '.join(map(str, self.sample))+'\n==================\n')
		
		if shuffle:
			random.shuffle(self.sample)			
			#print('sample_shuffle:\n'+' '.join(map(str, self.sample))+'\n==================\n')
		
		#for k in range(len(self.video_category_number)-1):
		for k in range(video_number):
			print('(video_number) k = %d' %k)
			self.sample_video[k*batch:k*batch+batch]=self.image_label_queue[self.sample[k]:self.sample[k]+batch]
			print('sample_video:\n'+'\n'.join(map(str, self.sample_video[k*batch:k*batch+batch]))+'\n==================\n')
	
		return self.input_image_label(self.sample_video)
	
	
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