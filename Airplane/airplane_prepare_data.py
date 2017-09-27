'''
To-do : If points <=1024, add random noise
for points >2048, split into two

in both cases, make sure the labels and indices are also taken care
'''

from airplane_kd_helpers import *
import glob
import numpy as np
import os

class_names = {
"02691156":"Airplane_02691156",
"02773838":"Bag_02773838",
"02954340":"Cap_02954340",
"02958343":"Car_02958343",
"03001627":"Chair_03001627",
"03261776":"Earphone_03261776",
"03467517":"Guitar_03467517",
"03624134":"Knife_03624134",
"03636649":"Lamp_03636649",
"03642806":"Laptop_03642806",
"03790512":"Motorbike_03790512",
"03797390":"Mug_03797390",
"03948459":"Pistol_03948459",
"04099429":"Rocket_04099429",
"04225987":"Skateboard_04225987",
"04379243":"Table_04379243"}

INP_SZ_1 = 1024
INP_SZ_2 = 2048
INP_SZ_3 = 4096

NUM_PTS = 2996

def get_fname(folder_name,suffix):
    global class_names
    words = folder_name.split('/')
    return class_names[words[len(words)-1]] + '_' + suffix

def check_class_equality(c1,c2):
    c1_split = c1.split('/')
    c2_split = c2.split('/')
    return c1_split[len(c1_split)-1] == c2_split[len(c2_split)-1]

def check_file_equality(f1,f2):
    f1_split = f1.split('/')
    f2_split = f2.split('/')
    return (f1_split[len(f1_split)-1].split('.'))[0] == (f2_split[len(f2_split)-1].split('.'))[0]

# Collecting all of the data
data_folders = ["./data/train_data/*","./data/val_data/*","./data/test_data/*"]
label_folders = ["./data/train_label/*","./data/val_label/*"]

data_fnames = ["X_train.npy","X_val.npy","X_test.npy"]
ind_map_fnames = ["ind_map_train.npy","ind_map_val.npy","ind_map_test.npy"]
label_fnames = ["y_train.npy","y_val.npy"]
print("Processing data..")

for i in range(2):  #iterating over train, val

    main_data_folder = data_folders[i]
    main_label_folder = label_folders[i]
    data_classes = sorted(glob.glob(main_data_folder))
    label_classes = sorted(glob.glob(main_label_folder))
    for data_class,label_class in zip(data_classes,label_classes):
        print(data_class)
        # if os.path.exists(get_fname(data_class,data_fnames[i])):
        #     continue
        if(check_class_equality(data_class,label_class) != True):
            print("Glob picks up in different order. Re-write code!")
            exit()
        model_files = sorted(glob.glob(data_class + '/*'))
        label_files = sorted(glob.glob(label_class + '/*'))
        data = []
        ind_maps = []
        labels = []
        for model_file,label_file in zip(model_files,label_files):
            print(model_file)
            if(check_file_equality(model_file,label_file) != True):
                print("Glob picks up in different order. Re-write code!")
                exit()
            pts = read_pts(model_file)
            lbls = read_labels(label_file)
            kd_leaves,kd_inds = create_kd_tree(pts)

            kd_leaves_placed = np.zeros((NUM_PTS,4))
            ind_start = (NUM_PTS - len(kd_leaves))/2
            kd_leaves_placed[ind_start:ind_start + len(kd_leaves), 0:3] = kd_leaves
            kd_leaves_placed[ind_start:ind_start + len(kd_leaves), 3] = np.ones(len(kd_leaves))

            kd_inds_placed = (-1)*np.zeros(NUM_PTS)
            kd_inds_placed[ind_start:ind_start + len(kd_leaves)] = kd_inds

            inds_for_lbls = [int(f) for f in kd_inds]
            kd_labels_placed = (-1)*np.zeros((NUM_PTS,1))
            kd_labels_placed[ind_start:ind_start + len(kd_leaves)] = lbls[inds_for_lbls]

            data.append(kd_leaves_placed)
            ind_maps.append(kd_inds_placed)
            labels.append(kd_labels_placed)
        np.save(get_fname(data_class,data_fnames[i]),data)
        np.save(get_fname(data_class,ind_map_fnames[i]),ind_maps)
        np.save(get_fname(label_class,label_fnames[i]),labels)


# Processing the test set (only points)
# print("Processing test data...")
# test_folder = data_folders[2]
# data_classes = sorted(glob.glob(test_folder))
# for data_class in data_classes:
#     print(data_class)
#     model_files = sorted(glob.glob(data_class + '/*'))
#     data = []
#     ind_maps = []
#     for model_file in model_files:
#         print(model_file)
#         pts = read_pts(model_file)
#         kd_leaves,kd_inds = create_kd_tree(pts)
#         kdl,kdi = augment_kd(kd_leaves,kd_inds)
#         for l,i in zip(kdl,kdi):
#             data.append(l)
#             ind_maps.append(i)
#     np.save(get_fname(data_class,data_fnames[2]),data)
#     np.save(get_fname(data_class,ind_map_fnames[2]),ind_maps)
