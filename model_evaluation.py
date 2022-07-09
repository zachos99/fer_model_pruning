# Standard Libraries
import os
import csv

# External Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import glob

import torch
from torchvision import transforms, datasets

from networks.dan import DAN


"""

    This script evaluates a model (whole or pruned) and prints its accuracy 
    For our project we used the AffectNet-HQ dataset ( https://www.kaggle.com/datasets/tom99763/affectnethq ) 
    We also used the pretrained AffectNet-8 model from https://github.com/yaoing/dan based on the 
    Distract Your Attention (DAN) network ( https://paperswithcode.com/paper/distract-your-attention-multi-head-cross )
    You can also use (with a few changes) the other pretrained models or your own model
    The images of the dataset have to be separated in folders depending on the class ( set the path below )
     
"""



class customTransform:
    """
    This function takes the images from the dataset loader and return them cropped to the face.
    For the face recognition we use the haarcascade-frontalface from cv2.

    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def __call__(self, img: PIL.Image) -> PIL.Image:
        #img= Image.fromarray(img,'RGB')

        faces = self.detect(img)

        if len(faces) == 0:
            return img

        #  single face detection
        x, y, w, h = faces[0]
        img = img.crop((x, y, x + w, y + h))

        return img


##################################################################################################################
##################################################################################################################
##################################################################################################################


class customTargetTransform:
    """
    The targets in the DataLoader are taken in alphabetical order like the order of the folders
    but the labels the dataset is trained have a specific order
    So we transform them in the correct order

    """

    def __init__(self):

        # Order the AffectNet dataset is trained
        self.classes_in_order = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        # Order the RAF dataset is trained
        #self.classes_in_order = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']

        self.classes_alphabetical = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise' ]

    def __call__(self, target):

        label = self.classes_alphabetical[target]
        class_index = self.classes_in_order.index(label)

        return class_index


##################################################################################################################
##################################################################################################################
##################################################################################################################

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
            You have also to change the num_class argument on DAN() call
        """
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        #self.labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)


        """
            You have to comment / uncomment the code below for pretrained or pruned model evaluation
        """
        # FOR PRETRAINED MODELS #
        #checkpoint = torch.load('./checkpoints/affecnet8_epoch5_acc0.6209.pth',
        #                        map_location=self.device)
        #self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        #                       #

        # FOR PRUNED MODELS     #
        self.model.load_state_dict(torch.load('./checkpoints/affecnet8_epoch5_acc0.6209_pruned_0.2.pth',map_location=self.device), strict=True)
        #                       #

        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def fer(self, val_loader):
        """
            Function for face emotion recognition (fer)
            We predict the emotion on the image and we compare with the target value
            We count the true predictions and we return the accuracy
        """
        with torch.no_grad():
            bingo_cnt = 0
            sample_cnt = 0
            for imgs, targets in val_loader:
                imgs = imgs.to(model.device)
                targets = targets.to(model.device)
                out, feat, heads = self.model(imgs)
                _, pred = torch.max(out, 1)             # Variable pred contains the indices of the max value for every (batch_size) tensor

                correct_num = torch.eq(pred, targets)   # Compares the two lists per element
                bingo_cnt += correct_num.sum().cpu()    # Counts the true matches
                sample_cnt += out.size(0)               # Counts total data

                label = [self.labels[i] for i in pred]  # contains the labels for every tensor
                print(label)
                print("sample count: ",sample_cnt)
                all_labels.append(label)
            return all_labels, bingo_cnt/sample_cnt


##################################################################################################################
##################################################################################################################
##################################################################################################################

model = Model()
# Choose batch size for the data loader
batch_size = 128
img_dir= "/home/zachos/Desktop/AffectNet HQ/AffectNetFixed"

data_transforms = transforms.Compose([
    customTransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

target_transforms = transforms.Compose([
    customTargetTransform()])

val_dataset = datasets.ImageFolder(img_dir, transform=data_transforms, target_transform=target_transforms)

print('Validation set size:', val_dataset.__len__())

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True)
all_labels = []
out, acc = model.fer(val_loader)


labels_array = np.array(out)
print(labels_array)

print('Accuracy: ', acc)
