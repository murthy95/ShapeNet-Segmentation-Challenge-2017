
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

def read_pts(fname):
    with open(fname,'r') as myfile:
        data = myfile.readlines()
    return np.loadtxt(data)

def read_labels(fname):
    with open(fname,'r') as myfile:
        data = myfile.readlines()
    return np.loadtxt(data).reshape((len(data),1))

def create_kd_tree(pts):
    pt_cld = []
    # find the kd-tree leafs
    kd_orients = []
    nodes = [pt_cld]
    temp_list = []
    while len(nodes[0]) != 1:
        temp_list.clear()
        for node in nodes:
            ranges = np.amax(node,axis = 0) - np.amin(node,axis = 0)
            split_dir = np.argmax(ranges)
            kd_orients.append(split_dir)
            sorted_node = node[node[:,split_dir].argsort()]
            num_pts = len(sorted_node)
            temp_list.append(sorted_node[0:np.int(num_pts/2),:])
            temp_list.append(sorted_node[np.int(num_pts/2):num_pts,:])
        nodes.clear()
        nodes.extend(temp_list)

    kd_leaves = np.array(nodes)
    kd_orients = np.array(kd_orients)
    # np.save('kd_tree.npy',kd_tree)
    print(str(time.time()-start_t) + 's elapsed.')
    return (kd_leaves,kd_orients)

if len(sys.argv)>1:
    pts = read_pts(sys.argv[1])
    labels = read_pts(sys.argv[2])
    num_parts = len(np.unique(labels))

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    colors = {1:'red', 2:'green', 3:'blue', 4:'cyan', 5:'black', 6:'yellow'}
    markers = ['o','^','+','x']
    xs, ys, zs = pts[:,0],pts[:,1],pts[:,2]
    for i in np.unique(labels):
        indices = np.where(labels==i)
        ax.scatter(xs[indices],ys[indices],zs[indices],color=colors[i],marker='.')
    plt.show()
    exit()
