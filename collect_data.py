import cv2
import time
import numpy as np
import os

cap=cv2.VideoCapture(0)

#Change file path to desired gesture folder.Images are saved to this folder.
path=r'C:\Personal\Projects\Gesture Recognition\One'

st=time.time()
i=0

while True:
    _,frame=cap.read()
    
    #Using only right side of frame
    frame=frame[100:340,0:320]
    
    #Image pre-processing(Have to filter out hand part properly.On random backgrounds it picks other colours)
    blur = cv2.GaussianBlur(frame,(5,5),0)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Masking to extract hand region
    lower=np.array([0,0,0])
    upper=np.array([100,200,125])

    mask=cv2.inRange(hsv,lower,upper)
    
    cv2.imshow('frame',mask)
    
    #Capture 500 images 5 seconds after video feed starts(To give time for user to get hand into position)
    if ((time.time()-st)>5) and i<1500:
        time.sleep(0.001)
        cv2.imwrite(os.path.join(path,str(i)+'.jpg'),cv2.resize(mask,(50,50)))
        i+=1
    
    if i==1500:
        print('Done')
        exit()

    #Press escape to end feed
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()