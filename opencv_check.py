#LIVE DETECTION

import cv2
import os
import time
start=time.time()
# Load the cascade
face_cascade = cv2.CascadeClassifier('/home/khush/Desktop/Robot/opencv/data/haarcascades/haarcascade_frontalface_default.xml')



# To capture video from webcam. 
cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture("file")
count=0
while True:
    # Read the frame
    _, img = cap.read()
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,"Human",(x,y-10), font,1,(255,255,255),2,cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    if cv2.waitKey(2) & 0xFF==ord('q'):  
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
