# Standard Libraries
import os

# External Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from networks.dan import DAN

"""
    This script detects real-time emotion from cam 
"""


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']#, 'contempt']

        self.model = DAN(num_head=4, num_class=7, pretrained=False)

        # FOR PRETRAINED MODELS #
        # checkpoint = torch.load('./checkpoints/rafdb_epoch21_acc0.897_bacc0.8275.pth',
        #    map_location=self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        #                       #

        # FOR PRUNED MODELS     #
        self.model.load_state_dict(
            torch.load('./checkpoints/rafdb_epoch21_acc0.897_bacc0.8275_pruned_0.7.pth', map_location=self.device),
            strict=True)
        #                       #

        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def fer(self, cv_img):

        img0 = Image.fromarray(cv_img)

        faces = self.detect(img0)

        if len(faces) == 0:
           return 'null'

        #  single face detection
        x, y, w, h = faces[0]

        img = img0.crop((x, y, x + w, y + h))

        img = self.data_transforms(img)
        img = img.view(1, 3, 224, 224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out, 1)
            index = int(pred)
            label = self.labels[index]

            return label


MAX_FPS=30
FPS=5
# start the webcam feed
cap = cv2.VideoCapture(0)
model = Model()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames
    for i in range(int(MAX_FPS / FPS)):
        cap.grab()
    _, image = cap.retrieve()

    label = model.fer(image)

    print(f'emotion label: {label}')
    cv2.imshow('Video', cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
