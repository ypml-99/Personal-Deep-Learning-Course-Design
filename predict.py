import os
from ultralytics import YOLO

model = YOLO("2022337621104Deep Learning.pt")

image_folder = "val"
images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))])

for image in images:
    model.predict(image, save=True)
