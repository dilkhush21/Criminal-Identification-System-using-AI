#TO SAVE PHOTOS

import cv2
import os
import time
# Load the cascade
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id=input()
# To capture video from webcam. 
cap = cv2.VideoCapture(1)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
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
        cv2.putText(img,"user."+id+"."+str(count),(x,y), font,1,(255,0,0),2,cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        img_item="user."+id+"."+str(count)+".png"
 
    #press c to capture photo
        if cv2.waitKey(2) & 0xFF==ord('c'):
            if(count<25 ):

                path = r"C:\Users\dk887\Desktop\Robot\dataSet"
                cv2.imwrite(os.path.join(path ,img_item),roi_color)
                count+=1
        
    
    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
    if(count==25):
        break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
