import cv2
import os
import numpy as np
import tensorflow as tf
import pyautogui
import time

model = tf.keras.models.load_model('model.h5')

#Testing it out on live video feed.
cap=cv2.VideoCapture(0)
st=time.time()
temp=0
temp1=0
categories=['Fist','Five','Shaka','One','Two','None']

while True:
    _,frame=cap.read()
    
    #Same preprocessing as in collection of data.
    frame=frame[100:340,0:320]
    blur = cv2.GaussianBlur(frame,(5,5),0)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    lower=np.array([0,0,0])
    upper=np.array([100,200,125])

    mask=cv2.inRange(hsv,lower,upper)
    
    #Resizing the mask and stacking horizontally to get matching dimensions for predictions.
    mask1=cv2.resize(mask,(50,50))
    mask1 = np.stack((mask1,)*3, axis=-1)

    #Model Predictions
    prediction=np.argmax(model.predict(np.array([mask1])))
    
    try:
        #Getting my prediction in the form of gesture name.
        out=categories[prediction]  
                       
        #Automation Part
        if out=='One' and temp==0:
            temp=1
            path=r'D:\Knives.Out.2019.WEBRip.x264-ION10\Knives.Out.2019.WEBRip.x264-ION10.mp4'
            os.system('start "" "' + path+ '"')
                
        if out=='Five' and temp1==0:
            pyautogui.hotkey('space')
            temp1=1
            time.sleep(1)
            
        if out=='Shaka'and temp1==0:
            pyautogui.hotkey('m')
            temp1=1
            time.sleep(1)
            

        #Printing my prediction on the output screen.
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(mask,out,(50,50),font,2,(255,255,255),2,cv2.LINE_4)
        cv2.imshow('frame',mask)
        
        if(temp1==1):
            if(time.time()-st>2):
                temp1=0
                st=time.time()
        
    except:
        pass
    
    #Press escape to end 
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()