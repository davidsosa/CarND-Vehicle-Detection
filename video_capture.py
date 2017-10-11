import cv2
import numpy as np


vidcap = cv2.VideoCapture('./testvideos/test1.mp4')

#vidcap.set(cv2.CAP_PROP_POS_MSEC,37000) 
success,image = vidcap.read()
# image is an array of array of [R,G,B] values
#cv2.imwrite("output_images/project_video_frames/project_video_frame6.jpg", image)
#cv2.imshow("37.000sec",image)

count = 0; 
print("success?",success)

width = vidcap.get(3)  # float
height = vidcap.get(4) # float

print(height," ",width)
while success:
    success,image = vidcap.read()
    cv2.imwrite("./testvideos/output_images/frame%d.jpg" % count, image)     # save frame as JPEG file
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
    	break
    count += 1

