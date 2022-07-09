import os
import csv
import glob
import pandas as pd
import numpy as np
from shutil import copyfile

"""

    Script made for AffectNet HQ dataset ( link: https://www.kaggle.com/datasets/tom99763/affectnethq )
    in order to rearrange images according to the labels.csv file ('results.csv' here)
    You have to create an AffectNetFixed folder with a subfolder for each emotion ( like the original dataset )
    Run for each emotion separately (change emotion in two lines below)
    
"""

df = pd.read_csv('/home/zachos/Desktop/AffectNet HQ/results.csv')
labels = np.array(df)
print('\nLabels:\n',labels)


data_dir = "/home/zachos/Desktop/AffectNet HQ/AffectNetDataset/anger"  # Which folder to fix (change for each emotion)
src_dir = "/home/zachos/Desktop/AffectNet HQ/AffectNetDataset/"
dest_dir = "/home/zachos/Desktop/AffectNet HQ/AffectNetFixed/"

clc_files = os.listdir(data_dir)
files = os.listdir(data_dir)

for i in range(len(files)):
    files[i]= 'anger/'+clc_files[i]                                    # Which folder to fix (change for each emotion)
print('\nFiles:\n',files)


for k in range(len(labels)):
    for i in range(len(files)):
        if labels[k][0] == files[i]:
            curr_img_path = os.path.join(src_dir, labels[k][0])
            new_img_path = os.path.join(dest_dir, labels[k][1],clc_files[i])
            copyfile(curr_img_path, new_img_path)

