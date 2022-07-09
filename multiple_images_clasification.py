# Standard Libraries
import os
import csv

# External Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import glob

import torch
from torchvision import transforms
from networks.dan import DAN


"""

    This scripts is used to classify images into classes
    We give the path of the folder containing images for classification and we save the results (emotion)
    on a (already created) results.csv file
    For the prediction we use a pretrained (pruned or not) model (set path below)
    
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

        """
            Depending on the model, you can comment/uncomment the label list
            You have also to change the num_class argument (7 or 8) on DAN() call
        """

        # Labels the AffectNet dataset is trained
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        # Labels the RAF dataset is trained
        #self.labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)


        """
            You have to comment / uncomment the code below for pretrained or pruned model evaluation
        """
        # FOR PRETRAINED MODELS #
        #checkpoint = torch.load('./checkpoints/affecnet8_epoch5_acc0.6209.pth',
        #                         map_location=self.device)
        #self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        #                       #

        # FOR PRUNED MODELS #
        self.model.load_state_dict(
            torch.load('./checkpoints/affecnet8_epoch5_acc0.6209_pruned_0.2.pth', map_location=self.device),
            strict=True)
        #                   #

        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def fer(self,cv_img):

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


##################################################################################################################
##################################################################################################################

model = Model()

# Folder containing the images
list = glob.glob("/home/zachos/Desktop/AffectNet HQ/test_set/*.jpg")
list.sort()

# Write the results in results.csv
with open('/home/zachos/Desktop/AffectNet HQ/results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['pth', 'label'])
    for img in list:
        n= cv2.imread(img)

        label = model.fer(n)
        print(f'Emotion on {img}  -->   label: {label}')
        writer.writerow([img,label])

'''
# Open the csv files as arrays to compare true or false predictions
df1 = pd.read_csv('/home/zachos/Desktop/AffectNet HQ/labels.csv')
df2 = pd.read_csv('/home/zachos/Desktop/AffectNet HQ/results.csv')
array1 = np.array(df1)
array2 = np.array(df2)

true_pred = 0
false_pred = 0

for i in range (array2.shape[0]) :
    if array1[i][1] == array2[i][1]:
        true_pred += 1
    else:
        false_pred += 1


print()
print('true predicted: ',true_pred)
print('false predicted: ',false_pred)

'''

