import cv2
import os
import pickle
import face_recognition
import cvzone
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db
from datetime import date
from datetime import datetime

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H:%M:%S")

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-8520d-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognition-8520d.appspot.com"
})

bucket = storage.bucket()
# calling webcam
video=cv2.VideoCapture(0)

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(3))
height = int(video.get(4))
output = cv2.VideoWriter(".mp4",
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            fps=fps*7, frameSize=(width, height))

# adjusting the detection box dimensions
video.set(3,640)
video.set(4,480)

# take bg image
imgBackground=cv2.imread('Resources/background.png')

# to get images of output display
foldermodepath = 'Resources/Modes'
modepathlist = os.listdir(foldermodepath)
imgmodelist = []
    
# adding images to our list
for path in modepathlist:
    # adding path to our image
    imgmodelist.append(cv2.imread(os.path.join(foldermodepath,path)))
    
# load the encode file
file = open('Encodes.p','rb')
knownEncodeListWIthIDs = pickle.load(file)   
file.close()

# extracting the file
knownEncodeList,customerIDs = knownEncodeListWIthIDs

modetype = 0
ans=0
counter = 0
id = -1
imgCustomer = []

while True:
    # read the video
    success,img=video.read()
    img = cv2.resize(img, (width, height))
    output.write(img)
        
    # to resize required image
    imgSmall = cv2.resize(img ,(0,0),None,0.25,0.25)
    
    # to maintain the defualt color
    imgSmall = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # locate the face
    faceInCurrentFrame = face_recognition.face_locations(imgSmall)
    # prepare encoodes of current face
    encodesOfCurrentFace = face_recognition.face_encodings(imgSmall,faceInCurrentFrame)

    # display webcam in img background []holds the dimensions
    imgBackground[162:162+480,55:55+640]=img
    
    # display images in output display
    imgBackground[44:44+633,808:808+414]=imgmodelist[modetype]
    
    if faceInCurrentFrame:
        for encodeFaces , faceLoc in zip(encodesOfCurrentFace,faceInCurrentFrame):
            # compares images and gives true false
            match =face_recognition.compare_faces(knownEncodeList,encodeFaces)
        
            # find face distance the lower the face distance the better the accuracy
            faceDistance = face_recognition.face_distance(knownEncodeList,encodeFaces) 
        
            # this wil, give the index of image whose img got matched with highest accuracy
            matchIndex = np.argmin(faceDistance) 
        
            # to put a box if face is detected
            if match[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                boundingbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, boundingbox, rt=0)
                id = customerIDs[matchIndex]
            
                if counter==0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Facial Authentication", imgBackground)
                    # once detection is done show active and make the counter 1
                    counter=1
                    modetype=1                
                
        if counter!=0:
            # if counter is 1 it means we have to load the data from the database and show it 
            if counter==1:
                # get the data from database
                customerInfo = db.reference(f'Customers/{id}').get()
            
                # get the image from database
                blob = bucket.get_blob(f'images/{id}.png')
                array = np.frombuffer(blob.download_as_string(),np.uint8)
                imgCustomer = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
            
                # checking the time elapsed
                datetimeObject = datetime.strptime(customerInfo['last_login_dnt'],
                                                "%Y-%m-%d %H:%M:%S")
            
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
            
                if secondsElapsed>60:
                    # update the database
                    ref = db.reference(f'Customers/{id}')
                    ref.child('last_login_dnt').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    ref.child('last_login_time').set(current_time)
                    ref.child('last_login_date').set(current_date)
            
                else:
                    modetype=3
                    counter = 0
                    imgBackground[44:44+633,808:808+414]=imgmodelist[modetype]
                    
            if modetype!=3:
                if 10<counter<30:
                    modetype = 2
                    ans = 1
                    
                imgBackground[44:44+633,808:808+414]=imgmodelist[modetype]
                if counter<=10 :
                    # to align the name according to the length
                    (w, h), _ = cv2.getTextSize(customerInfo['customer_name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w)//2
        
                    cv2.putText(imgBackground, str(customerInfo['customer_name']), (890+offset, 363),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
                            
                    cv2.putText(imgBackground, str(customerInfo['customerid']), (1022, 413),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    
                    cv2.putText(imgBackground, str(customerInfo['last_login_time']), (1053, 465),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    
                    cv2.putText(imgBackground, str(customerInfo['last_login_date']), (1036, 517),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    
                    imgBackground[93:93+216,904:904+216] = imgCustomer
                
                counter+=1
        
                if counter >= 30:
                    counter = 0
                    modeType = 0
                    customerInfo = []
                    imgCustomer = []
                    imgBackground[44:44+633,808:808+414]=imgmodelist[modetype]
    
    else:
        modeType = 0
        counter = 0
        
    # show webcame img
    # cv2.imshow("Webcam",img)

    # show img bg
    cv2.imshow("Facial Authentication",imgBackground)

    cv2.waitKey(1)
