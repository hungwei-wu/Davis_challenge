import numpy as np

from PIL import Image
from PIL import ImageColor

def create_overlay(jpegimage_filename,annotation_filename):	
##===============================================================##
## open image
##===============================================================##
	annotation = Image.open(annotation_filename)
	background = Image.open(jpegimage_filename)
	
	background = background.convert("RGBA")
	background_copy = background.copy()
	background_copy.show()

##===============================================================##
## create red overlay
##===============================================================##
	background.paste(ImageColor.getrgb('red'), mask=annotation)
	#background.paste(ImageColor.getrgb('yellow'), mask=annotation)
	
##===============================================================##
## create red transparent overlay
##===============================================================##	
	img_overlay = Image.blend(background_copy, background,0.5) 
	return img_overlay

	
	


	'''
	data = np.dstack((annotation, annotation, annotation))
	#print(data.shape)
	
	r1, g1, b1 = 255, 255, 255 # white
	r2, g2, b2 = 255, 0, 0 # red
	
	red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
	mask_object = (red == r1) & (green == g1) & (blue == b1)
	data[:,:,:3][mask_object] = [r2, g2, b2]
	
	overlay = Image.fromarray(np.array(data), 'RGB')
	#overlay.show()
	
	
	background = Image.open(jpegimage_filename)
	#overlay = Image.open("00001.png")
	
	background = background.convert("RGBA")
	#overlay = overlay.convert("RGBA")
	
	new_img = Image.blend(background, overlay, 0.5)
	#new_img.save("new.png","PNG")
	#new_img.show()
	return new_img'''
	
