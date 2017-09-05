
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
from kd_helpers import *

if len(sys.argv)>1:
    pts = read_pts(sys.argv[1])
    labels = read_labels(sys.argv[2])
    kd_leaves,kd_inds = create_kd_tree(pts)
    kd_leaves = kd_leaves[:,:,0].reshape((len(kd_inds),3))
    print(len(kd_leaves))

    labels = read_pts(sys.argv[2])
    colors = {1:'red', 2:'green', 3:'blue', 4:'cyan', 5:'black', 6:'yellow'}

    p1_x = []
    p1_y = []
    p1_z = []
    p2_x = []
    p2_y = []
    p2_z = []

    ind_1 = []
    ind_2 = []

    #splitting the leaves into two
    for i in range(2047):
        p1_x.append(kd_leaves[2*i,0])
        p1_y.append(kd_leaves[2*i,1])
        p1_z.append(kd_leaves[2*i,2])

        p2_x.append(kd_leaves[2*i + 1,0])
        p2_y.append(kd_leaves[2*i + 1,1])
        p2_z.append(kd_leaves[2*i + 1,2])

        ind_1.append(kd_inds[2*i])
        ind_2.append(kd_inds[2*i + 1])

    p1_x = np.array(p1_x)
    p1_y = np.array(p1_y)
    p1_z = np.array(p1_z)
    p2_x = np.array(p2_x)
    p2_y = np.array(p2_y)
    p2_z = np.array(p2_z)
    labels_1 = np.array(labels[ind_1])
    labels_2 = np.array(labels[ind_2])

    fig = plt.figure()

    s1 = fig.add_subplot(121,projection='3d')
    for i in np.unique(labels):
        indices = np.where(labels_1==i)
        s1.scatter(p1_x[indices],p1_y[indices],p1_z[indices],color=colors[i],marker='.')

    s2 = fig.add_subplot(122,projection='3d')
    for i in np.unique(labels):
        indices = np.where(labels_2==i)
        s2.scatter(p2_x[indices],p2_y[indices],p2_z[indices],color=colors[i],marker='.')

    plt.show()
    exit()
