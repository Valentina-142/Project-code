import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    faces, ids = [], []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        detected = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in detected:
            faces.append(imageNp[y:y+h, x:x+w])
            ids.append(Id)
    return faces, ids

print("🔄 Training faces...")
faces, ids = getImagesAndLabels("TrainingImage")
if len(faces) == 0:
    print("❌ No faces found.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")
print(f"✅ Training complete. Model saved as trainer.yml")
