from settings import settings
import cv2
import os                                                                                                             
import scipy.misc as misc
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
target_dir=os.path.join(settings.my_scratch,'data/DAVIS/Results/Segmentations/480p/smoothed_542new_test_online_400/')
print(target_dir)
for file in files:
  curNames=file.split('/')
  curName=curNames[-1]
  curDir=curNames[len(curNames)-2]
  this_directory=os.path.join(target_dir,curDir)
  target_name=this_directory+'/'+curName
  
  print(target_name)
  if not os.path.exists(this_directory):
    os.makedirs(this_directory)
  img=cv2.imread(file,0)
  print(img.shape)
  img = cv2.medianBlur(img,5)
  cv2.imwrite(target_name,img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
  #label = misc.imread(target_name,mode='P')
  #print(label.shape)
  #print(label[240])