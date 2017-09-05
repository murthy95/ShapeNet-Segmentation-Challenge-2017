from kd_helpers import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import time

if len(sys.argv)>1:
    pts = read_pts(sys.argv[1])
    labels = read_labels(sys.argv[2])
    kd_leaves,kd_inds = create_kd_tree(pts)
    # it returns two layers, only first has points
    kd_leaves = kd_leaves[:,:,0].reshape((len(kd_inds),3))

    p1_x = []
    p1_y = []
    p1_z = []
    p2_x = []
    p2_y = []
    p2_z = []

    #splitting the leaves into two
    for i in range(2047):
        p1_x.append(kd_leaves[2*i,0])
        p1_y.append(kd_leaves[2*i,1])
        p1_z.append(kd_leaves[2*i,2])

        p2_x.append(kd_leaves[2*i + 1,0])
        p2_y.append(kd_leaves[2*i + 1,1])
        p2_z.append(kd_leaves[2*i + 1,2])

    plt.ion()
    fig = plt.figure()
    splot = fig.add_subplot(111,projection = '3d')
    splot.scatter(p1_x,p1_y,p1_z,color='yellow',marker = '.')
    plt.pause(2)
    # plt.show()

    stride = int(sys.argv[3])
    for i in range(int(2048/stride - 1)):
        for j in range(stride):
            splot.scatter(p1_x[stride*i+j],p1_y[stride*i+j],p1_z[stride*i+j],color='black',marker = 'o')
        plt.pause(0.000001)
    #     plt.draw()
