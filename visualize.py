
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from kd_helpers import read_pts,read_labels


if len(sys.argv)>1:
    pts = read_pts(sys.argv[1])
    labels = read_pts(sys.argv[2])

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    colors = {1:'red', 2:'green', 3:'blue', 4:'cyan', 5:'black', 6:'yellow'}
    xs, ys, zs = pts[:,0],pts[:,1],pts[:,2]
    for i in np.unique(labels):
        indices = np.where(labels==i)
        ax.scatter(xs[indices],ys[indices],zs[indices],color=colors[i],marker='.')
    plt.show()
    exit()
