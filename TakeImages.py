import cv2
import os
import csv
import time

# Ensure folders exist
if not os.path.exists("TrainingImage"):
    os.makedirs("TrainingImage")
if not os.path.exists("StudentDetails.csv"):
    with open("StudentDetails.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name"])

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Input student info
while True:
    Id = input("Enter ID (numeric): ")
    if Id.isdigit():
        break
    print("❌ ID must be numeric!")
name = input("Enter Name: ").strip()

cam = cv2.VideoCapture(0)
sampleNum = 0
maxSamples = 20

print("\n📸 Capturing images... look at the camera.")
time.sleep(2)

while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to access camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        face_img = gray[y:y+h, x:x+w]
        filename = f"TrainingImage/{name}.{Id}.{sampleNum}.jpg"
        cv2.imwrite(filename, face_img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"Image {sampleNum}/{maxSamples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(f"✅ Saved: {filename}")
        time.sleep(2)
        break

    cv2.imshow('Capturing Faces', img)
    if cv2.waitKey(1) & 0xFF == ord('q') or sampleNum >= maxSamples:
        break

cam.release()
cv2.destroyAllWindows()

# Append to CSV if new ID
with open("StudentDetails.csv", 'r') as f:
    if str(Id) not in f.read():
        with open("StudentDetails.csv", 'a', newline='') as f_append:
            writer = csv.writer(f_append)
            writer.writerow([Id, name])

print(f"\n✅ {sampleNum} images saved for ID={Id}, Name={name}")
