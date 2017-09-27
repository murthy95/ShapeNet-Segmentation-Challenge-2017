import glob
import numpy as np
from kd_helpers import read_labels

# Checking the number of points in each model

max_pts = 0

for main_folder in ["./data/train_data/*","./data/val_data/*","./data/test_data/*"]:
    print(main_folder)
    folders = glob.glob(main_folder)
    model_files = []
    for folder in folders:
        model_files.extend(glob.glob(folder + '/*'))

    for model_file in model_files:
        with open(model_file,'r') as myfile:
            num_pts = len(myfile.readlines())
            if num_pts>max_pts:
                max_pts = num_pts
            if num_pts > 4096 or num_pts <= 512:
                print(num_pts)
                print("Out of range")
    print('Total models : '),
    print(len(model_files))
print("Highest number of points: " + str(max_pts))

# Checking the number of parts in each class

folders = glob.glob("./data/train_label/*")
for folder in folders:
    label_files = glob.glob(folder + '/*')
    print(folder)
    max_part = 0
    for label_file in label_files:
        labels = read_labels(label_file)
        if max(labels) > max_part:
            max_part = max(labels)
    print(max_part)
