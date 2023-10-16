import os
import random

import cv2

from ultralytics import YOLO


from tracker import Tracker


video_path = os.path.join('.', 'data', 'file.video') #add name of own video here

cap = cv2.VideoCapture(video_path)

#read frames from video
ret, frame = cap.read()

#pretrained model
model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for j in range(10)]

#initialise number of people to begin count
numPeople = 0

while ret:

    #list containing all detections
    results = model(frame)

    for result in results:
        #place to save all detections
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)


            if class_id == 0:

                numPeople += 1;
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

            cv2.putText(frame, f"Number of People: {numPeople}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

