import output
from PIL import Image
import sys
sys.path.append('../')
from settings import settings
import os                                                                                                             
def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(subdir + "/" + file)                                                                         
    return r     

    
files=list_files(settings.result_dir)
target_dir=os.path.join(settings.my_scratch,'data/DAVIS/Results/Segmentations/480p/overlay_smoothed_542new_test_online_400/')
print(target_dir)
for file in files:
  curNames=file.split('/')
  curName=curNames[-1]
  curVid=curNames[len(curNames)-2]
  this_directory=os.path.join(target_dir,curVid)
  target_name=this_directory+'/'+curName
  jpeg_file=settings.IMAGE_DIR+'/'+curVid+'/'+curName.split('.')[0]+'.jpg'
  print(target_name)
  if not os.path.exists(this_directory):
    os.makedirs(this_directory)
  #img=cv2.imread(file,0)
  #print(img.shape)
  #img = cv2.medianBlur(img,5)
  #cv2.imwrite(target_name,img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
  img_overlay = output.create_overlay(jpeg_file,file)
  #img_overlay.show()  
  img_overlay.save(target_name,"PNG")  
    
    
    
    
    
    
    
    


#
