import cv2
import os
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-8520d-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognition-8520d.appspot.com"
})


# to retrieve images of students
folderpath = 'images'
pathlist = os.listdir(folderpath)
imglist = []
customerIDs = []

for path in pathlist:
    
    imglist.append(cv2.imread(os.path.join(folderpath,path)))
    # this will split .png from the id name and it to the list of customerID
    customerIDs.append(os.path.splitext(path)[0])
    
    # to send images to database
    filename = f'{folderpath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)
    
    # defining a function which will find encodes of images  
    def findencodings(list):
        encodelist = []
        for img in list:
            # to set the default color of image during recognition
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodelist.append(encode)
        return encodelist
# to store encodes
knownEncodeList = findencodings(imglist) 
# to store encodes with respective images 
knownEncodeListWIthIDs = [knownEncodeList,customerIDs]

# providing the files to an external storage
file = open('Encodes.p','wb')
pickle.dump(knownEncodeListWIthIDs,file)
file.close()
