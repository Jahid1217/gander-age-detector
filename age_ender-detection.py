import cv2
import os

# Paths to model files
faceProto = "deploy.prototxt"
faceModel = "res10_300x300_ssd_iter_140000.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Verify model files exist
required_files = [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]
for f in required_files:
    if not os.path.exists(f):
        print(f"Error: {f} not found. Please download it and place it in the script directory.")
        exit()

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Use laptop camera
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x1, y1, x2, y2) = box.astype(int)

            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]

            # Age and gender prediction
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age Gender Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()