import cv2
import numpy as np
import asyncio
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import os

# MongoDB Connection
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["marketing_insights"]
collection = db["viewer_data"]

#load 'opencv_face_detector.pbtxt' and 'opencv_face_detector_uint8.pb'
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'

#initialize age and gender protocol buffer and model
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'

genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'


#load the networks
faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

#create age_list and gender_list and also model _mean_value 
# as this is required while creating blobs from an image.
age_list = ['(0-2)','(4-6)', '(8-12)', '(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
gender_list = ['Male', 'Female']
model_mean_value = (78.4263377603,87.7689143744, 114.895847746)

#Open Webcam and start detecting Gender and Age of person present in frame
video  = cv2.VideoCapture(0)
padding = 20
    
def faceBox(faceNet,frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0 , (227,227), [104,117,123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3]*frame_width)
            y1 = int(detection[0,0,i,4]*frame_height)
            x2 = int(detection[0,0,i,5]*frame_width)
            y2 = int(detection[0,0,i,6]*frame_height)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs

async def analyze_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot access the webcam")
        return

    while True:
        ret,frame= video.read()
        # frame,bboxs = faceBox(faceNet,frame)
        for bbox in bboxs:
            # face= frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            blob = cv2.dnn.blobFromImage(face,1.0 , (227,227),model_mean_value,swapRB = False)
            
            
            genderNet.setInput(blob)
            gender_pred = genderNet.forward()
            gender = gender_list[gender_pred[0].argmax()]
            
            
            ageNet.setInput(blob)
            age_pred = ageNet.forward()
            age = age_list[age_pred[0].argmax()]
            
            label = "{},{}".format(gender,age)
            cv2.rectangle(frame, (bbox[0], bbox[1]-10), (bbox[2], bbox[1]),(0,255,0),-1)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2, cv2.LINE_AA)

            # Log Data
            timestamp = datetime.datetime.now().isoformat()
            data = {"timestamp": timestamp, "gender": gender, "age": age}
            await collection.insert_one(data)
            
            
        cv2.imshow('ProjectPraise Age-Gender', frame)
        k = cv2.waitKey(1)
        if k==ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Run async function
asyncio.run(analyze_webcam())
