import cv2,os
import datetime;
# ct stores current time
  
import numpy as np
from PIL import Image
from gtts import gTTS
from playsound import playsound
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create() 
language = 'en'
rec.read('recognizer/trainingdata.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
cred = credentials.Certificate("criminal-identification-db2c8-firebase-adminsdk-pmn5o-067bd8ea2d.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'Detected').document(u'record')
record=0
flag1=0
flag2=0
flag3=0
while True:
    ret,img =cam.read()
    

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(id==1):
                id='Dilkhush'
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                if(flag1==0):
                    ct = datetime.datetime.now()
                    flag1=flag1+1
                    a='record_'+str(record)
                    record=record+1
                    doc_ref = db.collection(u'criminal').document()
                    doc_ref.set({
                        u'name': u'Name of person',
                        u'age': u'22',
                        u'address': u'address ' ,
                        u'recordId': u'1211' ,
                        u'detectedOn ' : u'%s'%(ct)

})
                
                    
      
        else:
            id='unkown'
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            
	    
        print(conf)   
        cv2.putText(img,str(id),(x,y-10), font,1,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow('Face',img)
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
            
cam.release()
cv2.destroyAllWindows()
