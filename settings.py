class settings():
	"""docstring for ClassName"""
	BATCH_SIZE  = 1
	LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
	#INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
	INITIAL_LEARNING_RATE = 0.001
	NUM_EPOCHS_PER_DECAY = 350
	MOVING_AVERAGE_DECAY = 0.9999
	NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2223 # number of images under segmentationClass file
	NUM_CLASSES = 21
	DEBUG = True
	#depth = [3,96,256,384,384,256,4096,4096,21]
	#depth = [3,96,256,256,384,21]
	#layer_depth = {'conv1': (3,96), 'conv2':(96,256),'conv3': (256,256), 'conv4':(256,384),'conv5':(384,21) }
	#layer_depth = {'conv1': (3,64), 'conv2':(64,96),'conv3': (96,96), 'conv4':(96,64),'conv5':(64,64)}
	#layer_depth = {'conv1': (3,64),'conv2':(64,128), 
	#	'conv3':(128,256), 'conv4':(256,512),'conv5':(512,512),
	#	'conv_6_nopool':(512,4096), 'conv_7_nopool':(4096,21)}
	MAX_STEPS = 50
	log_device_placement = False
	#shuffle = False
	shuffle = True
	summaries_dir = 'summary/'
	TRAIN_TXT = './input_person.txt'
	IMAGE_DIR = '/home/hungwei/cv542/CV_semanticSegmentation/data/TrainVal/VOCdevkit/VOC2011/JPEGImages'
	LABEL_DIR = '/home/hungwei/cv542/CV_semanticSegmentation/data/TrainVal/VOCdevkit/VOC2011/SegmentationClass'
	def __init__(self, arg):
		super(ClassName, self).__init__()
		self.arg = arg
		
