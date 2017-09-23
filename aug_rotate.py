
# coding: utf-8

# In[1]:

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import glob
from shutil import copyfile


def show_data(pts, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    colors = {1:'red', 2:'green', 3:'blue', 4:'cyan', 5:'black', 6:'yellow'}
    xs, ys, zs = pts[:,0],pts[:,1],pts[:,2]
    for i in np.unique(labels):
        indices = np.where(labels==i)
        ax.scatter(xs[indices],ys[indices],zs[indices],color=colors[i],marker='.')
    plt.show()


def rotate(points, alpha, beta, gamma):
    '''((x,y,z), alpha, beta, gamma) -> (x1,y1,z1)
        yaw, pitch, roll'''
    R_a = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    R_b = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    R_c = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    return np.matmul(np.matmul(R_a, np.matmul(R_b, R_c)), points.reshape(3,1))


def rotate_dataset(points_data_paths, points_labels_path):
    for each_data_path, each_labels_path in zip(points_data_paths, points_labels_path):
        aug_count = 0
        pts = np.loadtxt(each_data_path)
        temp_path1 = each_data_path.split("/")
        temp_path2 = each_labels_path.split("/")
        sampl_a = np.random.uniform(low=0, high=2*np.pi, size=(7,))
        sampl_b = np.random.uniform(low=0, high=2*np.pi, size=(7,))
        sampl_c = np.random.uniform(low=0, high=2*np.pi, size=(7,))
        for alpha, beta, gamma in zip(sampl_a, sampl_b, sampl_c):
            temp2 = []
            for pt in pts:
                temp2.append(rotate(pt, alpha, beta, gamma))
            pts = np.array(temp2).reshape(-1,3)
            aug_count+=1

            #<train/valid>_data/class_id/aug_<num>
            print(temp_path1[0]+"/"+temp_path1[1]+"/"+temp_path1[2].split(".")[0]+"aug"+str(aug_count)+".pts")
            np.savetxt(temp_path1[0]+"/"+temp_path1[1]+"/"+temp_path1[2].split(".")[0]+"aug"+str(aug_count)+".pts", pts)
            copyfile(each_labels_path, temp_path2[0]+"/"+temp_path2[1]+"/"+temp_path2[2].split(".")[0]+"aug"+str(aug_count)+".seg")


'''
Usecase:

points_xyz = list(glob.iglob("valid_data/*/*.pts", recursive=True))
points_label = list(glob.iglob("valid_label/*/*.seg", recursive=True))


# In[6]:

# creates 7 sets of augmented data by passing dataset paths and folders
# Ex: (valid_data/*/*.pts, valid_label/*/*.seg)
rotate_dataset(points_xyz, points_label)


# In[7]:

points_xyz = list(glob.iglob("valid_data/*/*.pts", recursive=True))
points_label = list(glob.iglob("valid_label/*/*.seg", recursive=True))

pts = np.loadtxt(points_xyz[7])
labels = np.loadtxt(points_label[7])

show_data(np.loadtxt(points_xyz[0]), np.loadtxt(points_label[0]))
show_data(pts, labels)
'''
