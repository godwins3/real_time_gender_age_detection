import cv2
import asyncio
import datetime
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB Connection
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["marketing_insights"]
collection = db["viewer_data"]

# Load models
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Age and gender lists
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']
model_mean_value = (78.4263377603, 87.7689143744, 114.895847746)

def faceBox(faceNet, frame):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame, bboxs

async def analyze_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break

        frame, bboxs = faceBox(faceNet, frame)

        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue  # Skip if no face detected

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_value, swapRB=False)

            # Gender prediction
            genderNet.setInput(blob)
            gender_pred = genderNet.forward()
            gender = gender_list[gender_pred[0].argmax()]

            # Age prediction
            ageNet.setInput(blob)
            age_pred = ageNet.forward()
            age = age_list[age_pred[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Log Data
            timestamp = datetime.datetime.now().isoformat()
            data = {"timestamp": timestamp, "gender": gender, "age": age}
            await collection.insert_one(data)

        cv2.imshow('ProjectPraise Age-Gender', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run async function
asyncio.run(analyze_webcam())
