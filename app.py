from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import cvzone
import os
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db
from datetime import datetime

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-8520d-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognition-8520d.appspot.com"
})

# Load encodes
file = open('Encodes.p', 'rb')
knownEncodeListWithIDs = pickle.load(file)
file.close()

knownEncodeList, customerIDs = knownEncodeListWithIDs

# Set up video capture
video = cv2.VideoCapture(0)

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(3))
height = int(video.get(4))

output = cv2.VideoWriter(".mp4",
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                         fps=fps * 7, frameSize=(width, height))

# adjusting the detection box dimensions
video.set(3, 640)
video.set(4, 480)

# take bg image
imgBackground = cv2.imread('Resources/background.png')

# to get images of output display
foldermodepath = 'Resources/Modes'
modepathlist = os.listdir(foldermodepath)
imgmodelist = []

# adding images to our list
for path in modepathlist:
    # adding path to our image
    imgmodelist.append(cv2.imread(os.path.join(foldermodepath, path)))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    global knownEncodeList, customerIDs, imgBackground, video, width, height, output

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # retrieve image from the request
    image = request.files['image']
    image_np = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image_np = cv2.resize(image_np, (width, height))

    # Write the image to the video file
    output.write(image_np)

    # to resize required image
    imgSmall = cv2.resize(image_np, (0, 0), None, 0.25, 0.25)

    # to maintain the default color
    imgSmall = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # locate the face
    faceInCurrentFrame = face_recognition.face_locations(imgSmall)
    # prepare encodes of current face
    encodesOfCurrentFace = face_recognition.face_encodings(imgSmall, faceInCurrentFrame)

    # display webcam in img background []holds the dimensions
    imgBackground[162:162 + height, 55:55 + width] = image_np

    if faceInCurrentFrame:
        for encodeFaces, faceLoc in zip(encodesOfCurrentFace, faceInCurrentFrame):
            # compares images and gives true false
            match = face_recognition.compare_faces(knownEncodeList, encodeFaces)

            # find face distance the lower the face distance the better the accuracy
            faceDistance = face_recognition.face_distance(knownEncodeList, encodeFaces)

            # this will give the index of image whose img got matched with highest accuracy
            matchIndex = np.argmin(faceDistance)

            # to put a box if face is detected
            if match[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                boundingbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, boundingbox, rt=0)
                id = customerIDs[matchIndex]

                # add your logic for processing the matched face here

    # convert the image to bytes for sending to the frontend
    _, img_encoded = cv2.imencode('.jpg', imgBackground)
    img_bytes = img_encoded.tobytes()

    return jsonify({'image': img_bytes.decode('utf-8')})


if __name__ == '__main__':
    app.run(debug=True)
