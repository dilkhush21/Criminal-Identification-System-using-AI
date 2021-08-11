
import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create() 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    print(imagePaths) 
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faceSamples.append(imageNp)
        Ids.append(Id)
        cv2.imshow("training",imageNp)
        cv2.waitKey(10)
    
        
    return Ids,faceSamples


Ids,faces = getImagesAndLabels(r'C:\Users\dk887\Desktop\robot\dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
