import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Load model and cascade
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
except:
    print("❌ Could not load trainer.yml. Please run TrainImages.py first.")
    exit()

if not os.path.exists("haarcascade_frontalface_default.xml"):
    print("❌ Haar cascade file missing.")
    exit()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load student details
try:
    df = pd.read_csv("StudentDetails.csv")
    id_name_map = dict(zip(df['ID'], df['Name']))
except:
    print("❌ Error loading StudentDetails.csv.")
    id_name_map = {}

# Setup camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Could not open webcam.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
attendance = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
marked_ids = set()

print("📡 Recognizing... Press 'Q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ Cannot read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            id_, conf = recognizer.predict(face)
        except:
            continue

        name = id_name_map.get(id_, "Unknown")
        label = f"{name} ({int(conf)}%)" if conf < 85 else "Unknown"

        # Mark attendance
        if conf < 85 and id_ not in marked_ids:
            now = datetime.now()
            attendance.loc[len(attendance)] = [id_, name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')]
            marked_ids.add(id_)
            print(f"✔ Marked: {name}")

        # Draw
        color = (0, 255, 0) if conf < 85 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), font, 0.6, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# Save attendance
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

filename = f"Attendance/Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
attendance.to_csv(filename, index=False)
print(f"✅ Attendance saved to: {filename}")
