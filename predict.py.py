# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:54:10 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:26:39 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:29:39 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon   Mar  8 13:23:01 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:27:26 2021

@author: Admin
"""

import traceback
import cv2
import numpy as np
import math
from keras.models import load_model
from keras.preprocessing import image
import imutils
import time
import operator
import pyautogui




model=load_model('leapgesture.h5')
print('model loaded!')
categories = {0: 'palm', 1: 'l', 2: 'fist', 3: 'fist_moved', 4: 'thumb', 5: 'index',6:'ok',7:'palm_moved',9:'c',10:'dowm'}


cap = cv2.VideoCapture(0)
     
while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
   
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
     
    #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        print(hierarchy)
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
    # l = no. of defects
        l=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)
            
            
        l+=1
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        
        else:
        
     

    
            img=mask
        
            IMG_SIZE=50
        
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            test_data =img
    
            orig = img
            data = img.reshape(-1,IMG_SIZE,IMG_SIZE,1)
       
            data = data / 255
            result = model.predict([data])
            # print(model_out)
            prediction = {'palm': result[0][0], 
                      'l': result[0][1], 
                      'fist': result[0][2],
                      'fist_moved': result[0][3],
                      'thumb': result[0][4],
                      'index': result[0][5],
                  'ok':result[0][6],
                  'palm_moved':result[0][7],
                  'c':result[0][8],
                  'down':result[0][9]}
        # Sorting based on top prediction
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, prediction[0][0], (10, 120),  font, 2, (0,0,255), 3, cv2.LINE_AA)    
            
            
            # video contorl
            if prediction[0][0]=='palm':
                pyautogui.typewrite(['space'], 0.2)
                pass
            
            
            elif prediction[0][0]=='l':
                pyautogui.hotkey('ctrl', 'left')
                pass
                
                
            elif prediction[0][0]=='thumb':
                pyautogui.hotkey('ctrl', 'right')
                pass
                
            elif prediction[0][0]=='fist':
                pyautogui.hotkey('ctrl', 'down')
                pass
        
            elif prediction[0][0]=='c':
                pyautogui.hotkey('ctrl', 'up')
                pass
            
            if prediction[0][0]=='palm_moved':
                #pyautogui.hotkey('ctrl', 'up')
                pass
            
            if prediction[0][0]=='fist_moved':
                #pyautogui.hotkey('ctrl', 'up')
                pass
            
            if prediction[0][0]=='index':
                #pyautogui.hotkey('ctrl', 'up')
                pass
            
            if prediction[0][0]=='ok':
                #pyautogui.hotkey('ctrl', 'up')
                pass
            
            if prediction[0][0]=='down':
                #pyautogui.hotkey('ctrl', 'up')
                pass
        
        
        
            
            
            """if(prediction[0][0]=='c'):
               
                
                window_name = "Live Video Feed"
                cv2.namedWindow(window_name)
                camera = cv2.VideoCapture(0)
                while True:
                    return_value,image1 = camera.read()
                    gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
                    cv2.imshow(window_name,gray)
                    if cv2.waitKey(1)& 0xFF == ord('s'):
                        cv2.imwrite('test.jpg',image1)
                        break
                cv2.destroyWindow(window_name)
                #$#camera.release() """


            
            #cv2.imshow("Frame", frame)
            
            print(contours)
            print(hierarchy)

        

                
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
        
        
        

    except Exception    :
        traceback.print_exc()
        pass
       # break
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()     