#!/usr/bin/env python
# coding: utf-8

# # This script obtaining frames from camera and save as jpg.

# In[ ]:



import cv2
video_capture = cv2.VideoCapture(0)
c=0
while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()

    
    timeF = 10
    
    if(c%timeF == 0): #save as jpg every 10 frame  
         cv2.imwrite('~/train_dir/me'+str(c) + '.jpg',frame) #save as jpg

    c+=1
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()

